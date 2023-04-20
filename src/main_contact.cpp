#include "ContactFEM.h"
#include "ElmerReader.h"
#include "Materials.h"
#include "fieldNames.h"
#include <vtkUnstructuredGrid.h>
#include <vtkPoints.h>
#include <vtkDoubleArray.h>
#include <vtkIntArray.h>
#include <vtkPointData.h>

#include <vtkXMLUnstructuredGridReader.h>

using namespace MyFEM;

#include "SensitivityAnalysis.h"
#include "DirectSensitivity.h"
#include "AdjointSensitivity.h"
#include "ElasticMaterialParameterHandler.h"
#include "InitialConditionParameterHandler.h"
#include "FrictionParameterHandler.h"
#include "ViscousMaterialParameterHandler.h"
#include "TemporalInterpolationField.h"
#include "BoundaryFieldObjectiveFunction.h"
#include "PointCloudObjectiveFunction.h"
#include "../LBFGSpp/LBFGS.h"
#include "../LBFGSpp/Newton.h"
#include "Remesher.h"

#include "cmaes/CMAMinimizer.h"

#ifdef _USE_OPTIMLIB
#define OPTIM_DONT_USE_OPENMP
#include "optim.hpp" // testing optimlib from https://github.com/kthohr/optim
Eigen::VectorXd optim_previousQ, optim_previousG; double optim_previousPhi;
double optim_wrapper(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data){
	double phi;
	if( opt_data==NULL ) return -1.0;
	SensitivityAnalysis *sa = (SensitivityAnalysis*) opt_data;
	Eigen::VectorXd q( vals_inp.size() ), g( vals_inp.size() );
	for(unsigned int i=0; i<vals_inp.size(); ++i){ q(i)=vals_inp(i); }
	if( optim_previousQ.size()==q.size() && (optim_previousQ-q).cwiseAbs().maxCoeff()==0.0 ){
		phi = optim_previousPhi;
		g = optim_previousG;
		printf("\n%% optim_wrapper: same params, returning old values");
	}else{
		phi = (*sa)(q,g);
		optim_previousPhi = phi;
		optim_previousQ = q;
		optim_previousG = g;
	}
	if( grad_out!=NULL){ for(unsigned int i=0; i<vals_inp.size(); ++i){ (*grad_out)(i)=g(i); }}
	else{ printf("\n%% optim_wrapper: skipping gradient"); }
	return phi;
}
#endif


class PlaneFieldFromVTK : public DiffableScalarField{
protected:
	Eigen::Vector3d n;
	double c;
public:
	PlaneFieldFromVTK(){ n = Eigen::Vector3d::UnitZ(); c=0.0; }
	virtual double eval(Eigen::Vector3d& n_, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const { n_=n; return n.dot(x)-c; }
	virtual void fitToPointCloud(std::string vtuPointCouldFileName);
	virtual void flip(){ n=-n;c=-c; }
	virtual Eigen::Vector3d getNormal(){ return n; }
	virtual double getOffset(){ return c; }
};
void PlaneFieldFromVTK::fitToPointCloud(std::string vtuPointCouldFileName){
	vtkSmartPointer<vtkXMLUnstructuredGridReader>  reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
	reader->GlobalWarningDisplayOff(); reader->SetFileName( vtuPointCouldFileName.c_str() ); reader->Update();
	vtkSmartPointer<vtkUnstructuredGrid> data = reader->GetOutput();
	if( data->GetNumberOfPoints()==0 ){
		printf("\n%% failed to read file \"%s\"!\n", vtuPointCouldFileName.c_str() );
		return;
	}
	printf("\n%% read \"%s\", have %d points.\n", vtuPointCouldFileName.c_str(), data->GetPoints()->GetNumberOfPoints() );
	printf("\n%% type %d (double==%d, float==%d)\n", data->GetPoints()->GetDataType(),VTK_DOUBLE,VTK_FLOAT);
	assert( data->GetPoints()->GetDataType()==VTK_FLOAT, "point coords not single floats" );

	Eigen::Map<Eigen::MatrixXf> points((float*)(data->GetPoints()->GetVoidPointer(0)), 3, data->GetPoints()->GetNumberOfPoints() );
	//std::cout << std::endl << points.transpose() << std::endl;

	Eigen::Vector3f centroid(points.row(0).mean(), points.row(1).mean(), points.row(2).mean());
	points.row(0).array() -= centroid(0);
	points.row(1).array() -= centroid(1);
	points.row(2).array() -= centroid(2);

	Eigen::JacobiSVD<Eigen::MatrixXf> svd = points.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::Vector3f normal = svd.matrixU().rightCols<1>();
	c = normal.dot(centroid);
	n = normal.cast<double>();

	if( c>0.0 ) flip(); // make c negative, which means the normal is oriented such that it points from the plane "towards" the origin -- note the projection of the origin onto the plane is (n*c)

	std::cout << std::endl << "n = [ " << n.transpose() << " ]; c = " << c << ";";
	std::cout << std::endl << "centroid = [ " << centroid.transpose() << " ]; normalPlusCentroid = [ " << (normal+centroid).transpose() << " ];" << std::endl;

	printf("\n\n%% Done.\n");
}


int main_contact_opt_toPoint(int argc, char* argv[]){
	printf("\n%% === main_contact_opt === \n\n");
	std::stringstream fileName;
	std::string outDir="../_out/";

	ContactFEM fem;
	if( argc>2 ){
		if(       std::string(argv[2]).find("qp")!=std::string::npos ){ fem.method=fem.CONTACT_IGNORE;//fem.CONTACT_QP; outDir.append("qp_");
		}else if( std::string(argv[2]).find("tanh")!=std::string::npos ){ fem.method=fem.CONTACT_TANH_PENALTY; outDir.append("tanh_");
		}else if( std::string(argv[2]).find("clamp")!=std::string::npos ){ fem.method=fem.CONTACT_CLAMP_PENALTY; outDir.append("clamp_");
		}else if( std::string(argv[2]).find("hyb")!=std::string::npos ){ fem.method=fem.CONTACT_HYBRID; outDir.append("hybrid_");
		}else printf("\n%% unknown option %s - ignored \n", argv[2]);
	}

#ifdef _WINDOWS
	// argv[1] contains the input mesh file name - we'll write outputs to outDir/inputMeshDir/inputMeshName... make sure inputMeshDir exists ...
	std::cout << "% output dir \"" << outDir << argv[1] << " ... ";
	std::string sc("@for %f in (\""); sc.append(outDir).append(argv[1]).append("\") do @mkdir %~dpf"); std::replace( sc.begin(), sc.end(), '/', '\\');
	int sr = system(sc.c_str());
	printf("\n\n");
#endif // _WINDOWS

	fem.useBDF2=true;
	fem.MAX_SOLVER_ITERS = 99;
	fem.FORCE_BALANCE_EPS = 1e-10;
	fem.FORCE_BALANCE_DELTA = 1e-12;
	fem.setPenaltyFactors(1e3, 1e3);
	if( fem.method == ContactFEM::CONTACT_HYBRID ) fem.setSlipDirectionRegularizer(1e-8);
	fem.doPrintResiduals = true;

	bool enablePreview = false;
	double
		timestep = 0.001  ,//*0.5
		stoptime = 0.25,
		frictionCoeff = 0.4,//0.0,//
		viscosity = 0.0125, //0.1,
		lameLambda = 1e4,
		lameMu = 1e5,
		density = 150; // FlexFoam has roughly lambda = 1e4 Pa, mu = 1e5 Pa, rho = 150 kg/m^3
	unsigned int nSteps = (unsigned int)((stoptime+0.5*timestep)/timestep);
	//lameLambda = 1e1; lameMu = 1e3; // soft works to end-point target with dt .001 and visc .1 (clamp,tanh(step1,m5) and hyb(init step .01,m50), zero init guess) (tanh init step .01, m=50, backwd init guess)
	lameLambda = 1e1; lameMu = 1e4; // medium-stiff works to end-point target with dt .001 and visc .1 (clamp,hyb~ backwd init step.01 m50, clamp,tanh~ zero init step .1 m50)

	fem.loadMeshFromElmerFiles(argv[1]);
	unsigned int bodyID = fem.bodyId.minCoeff();
	unsigned int bndryID = fem.bndId.minCoeff();
	MeshPreviewWindow* preview = NULL;
	if( enablePreview ) preview = new MeshPreviewWindow(fem.mesh.Get());
	fem.initializeMaterialModel(bodyID, LinearFEM::ISOTROPIC_NEOHOOKEAN, lameLambda, lameMu, density); printf("\n%% Neohookean material (la = %.4lg, mu = %.4lg, rho = %.4lg) ", lameLambda, lameMu, density);

	/**/  //default setup
	vtkSmartPointer<RotationInvariantViscosityModel> viscMdl = vtkSmartPointer<RotationInvariantViscosityModel>::New();
	fem.initializeViscosityModel(bodyID, viscMdl,  &viscosity ); printf("\n%% Rotation invariant viscosity (nu = %.4lg) ", viscosity);
	/*/  //BDF1 instead of viscosity
	fem.useBDF2=false;
	/**/

	class GravityField : public VectorField{ public: virtual void eval(Eigen::Vector3d& g, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		g.setZero(); g[2]=-9.81;
		g = (Eigen::AngleAxisd(-0.0/180.0*M_PI, Eigen::Vector3d::UnitY())*g).eval();
	} } g;
	fem.setExternalAcceleration(bodyID,g);

	class FloorField : public DiffableScalarField{ public: virtual double eval(Eigen::Vector3d& n, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		double floorHeight = -0.05;
		n = (Eigen::AngleAxisd(0.0/180.0*M_PI, Eigen::Vector3d::UnitY())*Eigen::Vector3d::UnitZ()).eval().normalized(); // outward unit normal
		return n.dot(x)-floorHeight; // flat floor at (n'x)==floorHeight (positive outside, negative inside)
	} } floor;
	fem.addRigidObstacle(floor, frictionCoeff); printf("\n%% floor at -0.05, z-normal (cf = %.4lg) ", frictionCoeff);
	
	InitialConditionParameterHandler icQ( bodyID ); icQ.setVelocity=true; icQ.velocityGradients=true;
	icQ.setAngularVelocity=true; icQ.angularVelocityScale=1.0; //icQ.angularVelocityGradients=true; 
	Eigen::VectorXd initCnds( icQ.getNumberOfParams(fem) ); initCnds.setZero();
	printf("\n%% total mass %.6lg ", density*fem.computeBodyVolume() );

	if(0){
		if( icQ.setVelocity )               initCnds.segment<3>(icQ.getVelocityIndex()) += 0.3*Eigen::Vector3d::UnitZ();
		if( icQ.setVelocity )               initCnds.segment<3>(icQ.getVelocityIndex()) += -0.5*Eigen::Vector3d::UnitX();
		if( icQ.setAngularVelocity ) initCnds.segment<3>(icQ.getAngularVelocityIndex()) += -4.0*Eigen::Vector3d::UnitY(); //10.0*Eigen::Vector3d::UnitY(); //
	}else initCnds << 0.45, -0,  0.4, 0, -4.0, 0; //initCnds << 0.496368, -0.0039957,  0.1601881, 0, -4.0, 0; //initCnds << 0.483, 0.00374511, 0.075, 0, -4.0, 0; //<< 0.48755, 0.00374511, 0.172329, -0.00590274, -4.00782, -0.00118485; // check solutions ...
	cout << endl << "% initial conditions (" << initCnds.transpose() << ") ";
	icQ.setNewParams(initCnds,fem); icQ.applyInitialConditions(fem); fem.setResetPoseFromCurrent(); fem.setResetVelocitiesFromCurrent(); // slightly cumbersome way to set persistent initial conditions if the parameter handler is not used for optimization ...

	// optimize initial conditions
	ParameterHandler& theQ = icQ; Eigen::VectorXd& q = initCnds;

	AverageBoundaryValueObjectiveFunction phi;
	class TargetField : public VectorField{ public: virtual void eval(Eigen::Vector3d& g, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		g=x;
		if( t>=endtime ) g = Eigen::Vector3d( 0.1, 0.0, -0.025 );
	}	TargetField(double t_end){endtime=t_end;} double endtime;
	} finalTarget(stoptime-0.5*timestep);
	phi.addTargetField(bndryID, &finalTarget); printf("\n%% Target at (0.1, 0.0, -0.025) from t = %.4lf ", finalTarget.endtime );

	/** /
	DirectSensitivity analysis(phi,theQ,fem); //AdjointSensitivity analysis(phi,theQ,fem, false );//expect non-symmetric matrices   //
	fileName.str(""); fileName.clear();
	fileName << outDir << argv[1];
	analysis.setupDynamicSim( timestep, nSteps, false, fileName.str(), preview );
	double objValue;
	
	if( 0 ){ //FD test
		icQ.angularVelocityGradients=true; 
		Eigen::VectorXd objGradient, adjGradient;

		DirectSensitivity analysis(phi,theQ,fem);
		analysis.setupDynamicSim( timestep, nSteps, false, fileName.str(), preview );
		objValue = analysis( q, objGradient );
		//return 1;

		fem.doPrintResiduals=false;
		AdjointSensitivity adjoint(phi,theQ,fem); adjoint.setAssumeSymmetricMatrices(false); // friction penalty forces will have asymmetric stiffness
		adjoint.setupDynamicSim( timestep, nSteps, false, fileName.str(), preview );
		double tmp = adjoint( q, adjGradient );

		printf("\n%% Direct v adjoint objective function evaluation ...\n");
		printf("\n%% ... value %.2lg =?= %.2lg  (abs diff %.2lg) \n", objValue,tmp, std::abs(objValue-tmp));
		printf("\n%% ... gradients\n");
		cout << "% ... [ " << objGradient.transpose() << " ]" << endl;
		cout << "% ... [ " << adjGradient.transpose() << " ]" << endl;
		printf("\n%% ... grad abs diff\n"); cout << "% ... [ " << (objGradient-adjGradient).cwiseAbs().transpose() << " ]" << endl;

		if( 1 ){ //finite-difference check dPhi/dq
			AdjointSensitivity& analysis = adjoint;
			double err_8 = analysis.finiteDifferenceTest(q, 1e-8);
			double err_10= analysis.finiteDifferenceTest(q, 1e-10);
			double err_6 = analysis.finiteDifferenceTest(q, 1e-6);
			double err_12= analysis.finiteDifferenceTest(q, 1e-12);
			
			printf("\n%% max errors (direct-adjoint, FDe-6, FDe-8, FDe-10, FDe-12) ... ");
			printf("\n maxAbsErr = [ %.4lg , %.4lg , %.4lg , %.4lg , %.4lg ];", (objGradient-adjGradient).cwiseAbs().maxCoeff(), err_6, err_8, err_10, err_12);
			printf("\n maxFDerrDiff = [ 000 , %.4lg , %.4lg , %.4lg , 000 ];", std::abs(err_6-err_12), std::abs(err_8-err_12), std::abs(err_10-err_12));
		}

		return 0;
	}
	fem.doPrintResiduals = false;

	LBFGSpp::LBFGSParam<double> optimOptions = defaultLBFGSoptions<LBFGSpp::LBFGSParam<double> >();
	optimOptions.init_step=0.1; optimOptions.m = 50;
	LBFGSpp::LBFGSSolver<double> solver(optimOptions); //minimize with LBFGS
	//LBFGSpp::NewtonSolver<double,Eigen::ColPivHouseholderQR<Eigen::MatrixXd> > solver(optimOptions); //why so slow???
	int r = solver.minimize(analysis, q, objValue);
	q = analysis.bestRunParameters;

	//if( 0 ){ // optimize with optimlib
	//	arma::vec iovals( q.size() );
	//	for(unsigned int i=0; i<q.size(); ++i){ iovals(i)=q(i); }
	//	optim::algo_settings_t optimsets; optimsets.iter_max=2000;
	//	bool ok = optim::de_prmm(iovals,optim_wrapper,&analysis,optimsets);
	//	//bool ok = optim::pso_dv(iovals,optim_wrapper,&analysis,optimsets);
	//	if( ok ){
	//		printf("\n%% optimlib success - params found: \n%%"); cout << iovals.t() << endl;
	//	}else printf("\n%% optimlib failed\n");
	//	q = analysis.bestRunParameters;
	//}else{ // optimize with CMA from SCP
	//	cout << endl << "% CMA optimization (5000,16,0.05,1e-16,1e-8) bound +/-1e2" << endl;
	//	std::vector<double> fixedValues; std::vector<unsigned int> fixedIDs;
	//	if( icQ.setAngularVelocity ){
	//		fixedValues.push_back( q[icQ.getAngularVelocityIndex()  ]);
	//		fixedValues.push_back( q[icQ.getAngularVelocityIndex()+1]);
	//		fixedValues.push_back( q[icQ.getAngularVelocityIndex()+2]);
	//		fixedIDs.push_back( icQ.getAngularVelocityIndex()  );
	//		fixedIDs.push_back( icQ.getAngularVelocityIndex()+1);
	//		fixedIDs.push_back( icQ.getAngularVelocityIndex()+2);
	//	}
	//	CMAObjectiveFunction cmaobj(analysis,fixedValues,fixedIDs);
	//	CMAMinimizer cmaopt(5000,16,0.05,1e-16,1e-8);
	//	double phiVal; bool ret;
	//	Eigen::VectorXd q_lower(q.size()), q_upper(q.size());
	//	q_lower.setConstant(-1e2); q_upper.setConstant(1e2);
	//	cmaopt.setBounds(q_lower,q_upper);
	//	ret = cmaopt.minimize(&cmaobj,q,phiVal);
	//	cout << endl << "% CMA returned " << ret; 
	//}

	cout << endl << "best_q = [ " << analysis.bestRunParameters.transpose() << "]; ";
	cout << endl << "best_grad_phi = [ " << analysis.bestRunGradient.transpose() << "]; best_phi = " << analysis.bestRunPhiValue << "; ";

	/*/

	fem.doPrintResiduals=false;

	AdjointSensitivity analysis(phi,theQ,fem, false);//DirectSensitivity analysis(phi,theQ,fem); //

	fileName.str(""); fileName.clear();
	fileName << outDir << argv[1];
	analysis.setupDynamicSim( timestep, nSteps , false, fileName.str() );
	Eigen::VectorXd objGradient;
	// sphere clamp lameMu=1e4 solution: best_q = [     0.48755  0.00374511    0.172329 -0.00590274    -4.00782 -0.00118485];
	// same with fine mesh (2 bounces):  q = [   0.516368 -0.0039957  0.0601881          0         -4          0 ];  phi =  4.483e-12;
	// and  with fine mesh (1 bounce ):  q = [  0.505533 -0.003451  0.175441         0        -4         0 ];  phi =  1.296e-12; 
	q <<  0.48755, 0.00374511, 0.172329, -0.00590274, -4.00782, -0.00118485;
	//q <<   0.4, 0.00409134,   0.161977   ,       0 ,        -4  ,        0; // for frictionless test
	// sphere tanh lameMu=1e3 (soft invisc) solution (no ang vel opt)
	//q << 0.525813, 0.00866866, 0.0280971, 0.0, -4.0, 0;
	// BDF1 no viscosity, no angular velocity optimization solution:
	//q <<  0.458029, 0.00153451, 0.422859, 0.0, -4, 0.0;

	// sample objective function along x-z initial velocity ...
	Eigen::VectorXd xVel,zVel;
	xVel.resize(51); zVel.resize(71); //xVel.resize(15); zVel.resize(17); //xVel.resize(5); zVel.resize(7); //

	// also use soft penalty factor
	fem.setPenaltyFactors(1e2, 1e2);

	//xVel.setLinSpaced( q(0) -0.1,0.1+ q(0) ); zVel.setLinSpaced( q(2) -0.2,0.1+ q(2) ); // range 0.2/0.3 (broad)
	//fileName.str(""); fileName.clear(); fileName << outDir << argv[1] << "_objFcnSamples_broad_k1e2.m";

	xVel.setLinSpaced( 0.47, 0.51  ); zVel.setLinSpaced( 0.14, 0.20 ); // range 0.04/0.06 (narrow)
	fileName.str(""); fileName.clear(); fileName << outDir << argv[1] << "_objFcnSamples_narrow_k1e2.m";


	Eigen::MatrixXd phiVals;  phiVals.resize( xVel.size(), zVel.size() );
	Eigen::MatrixXd dphi_dvx; dphi_dvx.resize( xVel.size(), zVel.size() );
	Eigen::MatrixXd dphi_dvz; dphi_dvz.resize( xVel.size(), zVel.size() );
	printf("\n%% Sampling objective function: x = [%lf : %lf : %lf], z = [%lf : %lf : %lf]", xVel.minCoeff(), xVel(1)-xVel(0), xVel.maxCoeff(), zVel.minCoeff(), zVel(1)-zVel(0), zVel.maxCoeff());
	for(unsigned int i=0; i<xVel.size(); ++i) for(unsigned int j=0; j<zVel.size(); ++j){
		printf("\n%% %5d/%d ... ", zVel.size()*i+j+1, xVel.size()*zVel.size());
		q(0) = xVel(i);
		q(2) = zVel(j);
		phiVals(i,j) = analysis( q, objGradient );
		dphi_dvx(i,j) = objGradient(0);
		dphi_dvz(i,j) = objGradient(2);
	}
	ofstream outFile(fileName.str());
	outFile << endl << " xVel = [ " << xVel.transpose() << " ];" << endl;
	outFile << endl << " zVel = [ " << zVel.transpose() << " ];" << endl;
	outFile << endl << " phiVals = [ "  << endl << phiVals  << " ];" << endl;
	outFile << endl << " dphi_dvx = [ " << endl << dphi_dvx << " ];" << endl;
	outFile << endl << " dphi_dvz = [ " << endl << dphi_dvz << " ];" << endl;

	//// sample objective function along x-ry initial velocity ...
	//q.setZero(); q(2)=0.3; // ground truth is [0.2 0 0.3 0 -10 0] // 0.233012  0.00518844    0.309836  0.00164286 -0.00671362
	//Eigen::VectorXd xVel,ryVel;
	//xVel.resize(11); ryVel.resize(13);  //xVel.resize(31); ryVel.resize(31);
	//xVel.setLinSpaced( 0.0 , 0.4 ); ryVel.setLinSpaced(-0.3, 0.1); // assuming angularVelocityScale=100.0
	//theQ.angularVelocityScale=100.0;
	//Eigen::MatrixXd phiVals;  phiVals.resize( xVel.size(), ryVel.size() );
	//Eigen::MatrixXd dphi_dvx; dphi_dvx.resize( xVel.size(), ryVel.size() );
	//Eigen::MatrixXd dphi_dvry; dphi_dvry.resize( xVel.size(), ryVel.size() );
	//printf("\n%% Sampling objective function: x = [%lf : %lf : %lf], ry = [%lf : %lf : %lf]", xVel.minCoeff(), xVel(1)-xVel(0), xVel.maxCoeff(), ryVel.minCoeff(), ryVel(1)-ryVel(0), ryVel.maxCoeff());
	//for(unsigned int i=0; i<xVel.size(); ++i) for(unsigned int j=0; j<ryVel.size(); ++j){
	//	printf("\n%% %5d/%d ... ", ryVel.size()*i+j+1, xVel.size()*ryVel.size());
	//	q(0) = xVel(i);
	//	q(4) = ryVel(j);
	//	phiVals(i,j) = analysis( q, objGradient );
	//	dphi_dvx(i,j) = objGradient(0);
	//	dphi_dvry(i,j) = objGradient(4);
	//}
	//fileName.str(""); fileName.clear(); fileName << outDir << argv[1] << "_objFcnSamplesRot.m";
	//ofstream outFile(fileName.str());
	//outFile << endl << " xVel = [ " << xVel.transpose() << " ];" << endl;
	//outFile << endl << " ryVel = [ " << ryVel.transpose() << " ];" << endl;
	//outFile << endl << " phiVals = [ "  << endl << phiVals  << " ];" << endl;
	//outFile << endl << " dphi_dvx = [ " << endl << dphi_dvx << " ];" << endl;
	//outFile << endl << " dphi_dvry = [ " << endl << dphi_dvry << " ];" << endl;


	//fileName.str(""); fileName.clear();
	//fileName << outDir << argv[1] << "_" << std::setfill ('0') << std::setw(5) << (0);
	//fem.saveMeshToVTUFile(fileName.str());
	//theQ.setNewParams( q , fem);
	//theQ.applyInitialConditions(fem);
	//fem.assembleMassAndExternalForce();
	//for(unsigned int step=0; step<nSteps; ++step){
	//	printf("\n%% (%5d/%d) ", step+1,nSteps);
	//	fem.updateAllBoundaryData();
	//	fem.dynamicImplicitTimestep(timestep);
	//	fileName.str(""); fileName.clear();
	//	fileName << outDir << argv[1] << "_" << std::setfill ('0') << std::setw(5) << (step+1);
	//	fem.saveMeshToVTUFile(fileName.str());
	//	// update preview
	//	preview.render();
	//}	printf("\n");
	//Eigen::VectorXd phi_x,phi_v,phi_f,phi_q;
	//phi.evaluate(fem,phi_x,phi_v,phi_f,phi_q);
	//vtkSmartPointer<vtkDoubleArray> vtkPhiX = vtkSmartPointer<vtkDoubleArray>::New();
	//vtkPhiX->SetName("phi_x"); vtkPhiX->SetNumberOfComponents(3); vtkPhiX->SetNumberOfTuples(fem.getNumberOfNodes());
	//for(unsigned int i=0; i<fem.getNumberOfNodes(); ++i) vtkPhiX->SetTuple3(i, phi_x(fem.getNodalDof(i,fem.X_DOF)), phi_x(fem.getNodalDof(i,fem.Y_DOF)), phi_x(fem.getNodalDof(i,fem.Z_DOF)) );
	//fem.mesh->GetPointData()->AddArray(vtkPhiX);
	//fem.saveMeshToVTUFile(fileName.str());
	/**/

	if( preview != NULL ) delete preview;
	return 0;
}

int main_contact_opt_toPointWithWall(int argc, char* argv[]){
	printf("\n%% === main_contact_opt_toPointWithWall === \n\n");
	std::stringstream fileName;
	std::string outDir="../_out/";

	ContactFEM fem;
	if( argc>2 ){
		if(       std::string(argv[2]).find("qp")!=std::string::npos ){ fem.method=fem.CONTACT_QP; outDir.append("qp_");
		}else if( std::string(argv[2]).find("class")!=std::string::npos ){ fem.method=92; outDir.append("class_"); // we'll probably drop the first two options here eventually ...
		}else if( std::string(argv[2]).find("tanh")!=std::string::npos ){ fem.method=fem.CONTACT_TANH_PENALTY; outDir.append("tanh_");
		}else if( std::string(argv[2]).find("clamp")!=std::string::npos ){ fem.method=fem.CONTACT_CLAMP_PENALTY; outDir.append("clamp_");
		}else if( std::string(argv[2]).find("hyb")!=std::string::npos ){ fem.method=fem.CONTACT_HYBRID; outDir.append("hybrid_");
		}else printf("\n%% unknown option %s - ignored \n", argv[2]);
	}

#ifdef _WINDOWS
	// argv[1] contains the input mesh file name - we'll write outputs to outDir/inputMeshDir/inputMeshName... make sure inputMeshDir exists ...
	std::cout << "% output dir \"" << outDir << argv[1] << " ... ";
	std::string sc("@for %f in (\""); sc.append(outDir).append(argv[1]).append("\") do @mkdir %~dpf"); std::replace( sc.begin(), sc.end(), '/', '\\');
	int sr = system(sc.c_str());
	printf("\n\n");
#endif // _WINDOWS

	fem.useBDF2=true;
	fem.MAX_SOLVER_ITERS = 99;
	fem.FORCE_BALANCE_EPS = 1e-10;
	fem.FORCE_BALANCE_DELTA = 1e-12;
	fem.setPenaltyFactors(1e3, 1e3);
	if( fem.method == ContactFEM::CONTACT_HYBRID ) fem.setSlipDirectionRegularizer(1e-8);
	fem.doPrintResiduals = false;

	bool enablePreview = false;
	double
		timestep = 0.001,
		stoptime = 0.35,
		frictionCoeff = 0.4,
		viscosity = 0.1/8.0,
		lameLambda = 1e4,
		lameMu = 1e5,
		density = 150;
	unsigned int nSteps = (unsigned int)((stoptime+0.5*timestep)/timestep);
	lameLambda = 1e1; lameMu = 1e4;

	fem.loadMeshFromElmerFiles(argv[1]);
	unsigned int bodyID = fem.bodyId.minCoeff();
	unsigned int bndryID = fem.bndId.minCoeff();
	MeshPreviewWindow* preview = NULL;
	if( enablePreview ) preview = new MeshPreviewWindow(fem.mesh.Get());
	fem.initializeMaterialModel(bodyID, LinearFEM::ISOTROPIC_NEOHOOKEAN, lameLambda, lameMu, density); printf("\n%% Neohookean material (la = %.4lg, mu = %.4lg, rho = %.4lg) ", lameLambda, lameMu, density);

	vtkSmartPointer<RotationInvariantViscosityModel> viscMdl = vtkSmartPointer<RotationInvariantViscosityModel>::New();
	fem.initializeViscosityModel(bodyID, viscMdl,  &viscosity ); printf("\n%% Rotation invariant viscosity (nu = %.4lg) ", viscosity);

	class GravityField : public VectorField{ public: virtual void eval(Eigen::Vector3d& g, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		g.setZero(); g[2]=-9.81;
		g = (Eigen::AngleAxisd(-0.0/180.0*M_PI, Eigen::Vector3d::UnitY())*g).eval();
	} } g;
	fem.setExternalAcceleration(bodyID,g);

	class FloorField : public DiffableScalarField{ public: virtual double eval(Eigen::Vector3d& n, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		double floorHeight = -0.05;
		n = (Eigen::AngleAxisd(0.0/180.0*M_PI, Eigen::Vector3d::UnitY())*Eigen::Vector3d::UnitZ()).eval().normalized(); // outward unit normal
		return n.dot(x)-floorHeight; // flat floor at (n'x)==floorHeight (positive outside, negative inside)
	} } floor;
	fem.addRigidObstacle(floor, frictionCoeff); printf("\n%% floor at -0.05, z-normal (cf = %.4lg) ", frictionCoeff);
	
	class WallField : public DiffableScalarField{ public: virtual double eval(Eigen::Vector3d& n, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		double wallX = 0.1;
		n = -Eigen::Vector3d::UnitY();
		return n.dot(x)+wallX;
	} } wall;
	fem.addRigidObstacle(wall, 2.0*frictionCoeff);

	InitialConditionParameterHandler icQ( bodyID ); icQ.setVelocity=true; icQ.velocityGradients=true;
	//icQ.setAngularVelocity=true; icQ.angularVelocityScale=1.0; //icQ.angularVelocityGradients=true; 
	Eigen::VectorXd initCnds( icQ.getNumberOfParams(fem) ); initCnds.setZero();

	if(1){
		if( icQ.setVelocity )               initCnds.segment<3>(icQ.getVelocityIndex()) += 0.2*Eigen::Vector3d::UnitX();
		if( icQ.setVelocity )               initCnds.segment<3>(icQ.getVelocityIndex()) += 0.5*Eigen::Vector3d::UnitY();
		if( icQ.setVelocity )               initCnds.segment<3>(icQ.getVelocityIndex()) += 1.4*Eigen::Vector3d::UnitZ(); //new version: -0.6,0.6,2.8,clamp ok 0.6,0.8,*1.4*,tanh ok; old old version: 0.3+clamp and -0.6+tanh works
	}
	cout << endl << "% initial conditions (" << initCnds.transpose() << ") ";
	icQ.setNewParams(initCnds,fem); icQ.applyInitialConditions(fem); fem.setResetPoseFromCurrent(); fem.setResetVelocitiesFromCurrent(); // slightly cumbersome way to set persistent initial conditions if the parameter handler is not used for optimization ...

	// optimize initial conditions
	ParameterHandler& theQ = icQ; Eigen::VectorXd& q = initCnds;

	AverageBoundaryValueObjectiveFunction phi;
	//class TargetField : public VectorField{ public: virtual void eval(Eigen::Vector3d& g, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
	//	g=x;
	//	if( t>=endtime ) g = Eigen::Vector3d( 0.1, 0.0, -0.025 );
	//}	TargetField(double t_end){endtime=t_end;} double endtime;
	//} finalTarget(stoptime-0.5*timestep);
	//printf("\n%% Target at (0.1, 0.0, -0.025) from t = %.4lf ", finalTarget.endtime ); // old version for stoptime 0.25
	class TargetField : public VectorField{ public: virtual void eval(Eigen::Vector3d& g, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		g=x;
		if( t>=endtime ) g = Eigen::Vector3d( 0.3, 0.0, 0.0 );
	}	TargetField(double t_end){endtime=t_end;} double endtime;
	} finalTarget(stoptime-0.5*timestep);
	printf("\n%% Target at (0.3, 0.0, 0.0) from t = %.4lf ", finalTarget.endtime );
	phi.addTargetField(bndryID, &finalTarget); // new version for stoptime 0.35

	AdjointSensitivity analysis(phi,theQ,fem, false );//expect non-symmetric matrices   //DirectSensitivity analysis(phi,theQ,fem); //
	fileName.str(""); fileName.clear();
	fileName << outDir << argv[1];
	analysis.setupDynamicSim( timestep, nSteps, false, fileName.str(), preview );
	double objValue;

	LBFGSpp::LBFGSParam<double> optimOptions = defaultLBFGSoptions<LBFGSpp::LBFGSParam<double> >();
	optimOptions.init_step=0.01; //optimOptions.m = 50;
	LBFGSpp::LBFGSSolver<double> solver(optimOptions); //minimize with LBFGS
	/**/
	int r = solver.minimize(analysis, q, objValue);
	q = analysis.bestRunParameters;
	/*/ // try penalty continuation
	for(double pfCont=1e2; pfCont<1e3; pfCont*=1.2){
		optimOptions.delta=1e-5; optimOptions.epsilon=1e-8;
		fem.setPenaltyFactors(pfCont,pfCont);
		int r = solver.minimize(analysis, q, objValue);
		q = analysis.bestRunParameters;
	}
	optimOptions.delta=1e-8; optimOptions.epsilon=1e-16;
	fem.setPenaltyFactors(1e3,1e3);
	int r = solver.minimize(analysis, q, objValue);
	q = analysis.bestRunParameters;
	/**/

	cout << endl << "best_q = [ " << analysis.bestRunParameters.transpose() << "]; ";
	cout << endl << "best_grad_phi = [ " << analysis.bestRunGradient.transpose() << "]; best_phi = " << analysis.bestRunPhiValue << "; ";

	// verify against hybrid contacts ...
	if( fem.method==fem.CONTACT_TANH_PENALTY || fem.method==fem.CONTACT_CLAMP_PENALTY ){
		fem.method =fem.CONTACT_HYBRID;
		fileName.str(""); fileName.clear();
		fileName << outDir << argv[1] << "_hybCompare";
		analysis.setupDynamicSim( timestep, nSteps, false, fileName.str(), preview );
		Eigen::VectorXd grad;
		objValue = analysis(q,grad);
		cout << endl << "hyb_grad_phi = [ " << grad.transpose() << "]; hyb_phi = " << objValue << "; ";
	}

	if( preview != NULL ) delete preview;
	return 0;
}

int main_contact_opt_toLine(int argc, char* argv[]){
	printf("\n%% === main_contact_opt === \n\n");
	std::stringstream fileName;
	std::string outDir="../_out/";

	ContactFEM fem;
	if( argc>2 ){
		if(       std::string(argv[2]).find("qp")!=std::string::npos ){ fem.method=fem.CONTACT_QP; outDir.append("qp_");
		}else if( std::string(argv[2]).find("class")!=std::string::npos ){ fem.method=92; outDir.append("class_"); // we'll probably drop the first two options here eventually ...
		}else if( std::string(argv[2]).find("tanh")!=std::string::npos ){ fem.method=fem.CONTACT_TANH_PENALTY; outDir.append("tanh_");
		}else if( std::string(argv[2]).find("clamp")!=std::string::npos ){ fem.method=fem.CONTACT_CLAMP_PENALTY; outDir.append("clamp_");
		}else if( std::string(argv[2]).find("hyb")!=std::string::npos ){ fem.method=fem.CONTACT_HYBRID; outDir.append("hybrid_");
		}else printf("\n%% unknown option %s - ignored \n", argv[2]);
	}

#ifdef _WINDOWS
	// argv[1] contains the input mesh file name - we'll write outputs to outDir/inputMeshDir/inputMeshName... make sure inputMeshDir exists ...
	std::cout << "% output dir \"" << outDir << argv[1] << " ... ";
	std::string sc("@for %f in (\""); sc.append(outDir).append(argv[1]).append("\") do @mkdir %~dpf"); std::replace( sc.begin(), sc.end(), '/', '\\');
	int sr = system(sc.c_str());
	printf("\n\n");
#endif // _WINDOWS

	fem.useBDF2=true;
	fem.MAX_SOLVER_ITERS = 99;
	fem.FORCE_BALANCE_EPS = 1e-10;
	fem.FORCE_BALANCE_DELTA = 1e-12;
	fem.setPenaltyFactors(1e3, 1e3);
	if( fem.method == ContactFEM::CONTACT_HYBRID ) fem.setSlipDirectionRegularizer(1e-8);
	fem.doPrintResiduals = false;

	bool enablePreview = false;
	double
		timestep = 0.001,
		stoptime = 0.25,
		frictionCoeff = 0.4,
		viscosity = 0.1,
		lameLambda = 1e4,
		lameMu = 1e5,
		density = 150; // FlexFoam has roughly lambda = 1e4 Pa, mu = 1e5 Pa, rho = 150 kg/m^3
	unsigned int nSteps = (unsigned int)((stoptime+0.5*timestep)/timestep);
	lameLambda = 1e1; lameMu = 1e4; // throw with backspin (target at stoptime/2=0.125 on line along z-axis, soft mat, clamp, dt.001 visc.1 initstep.01 m50, angvelscale2pi, all methods with LBFGS->ADAM(200)->LBFGS)

	fem.loadMeshFromElmerFiles(argv[1]);
	unsigned int bodyID = fem.bodyId.minCoeff();
	unsigned int bndryID = fem.bndId.minCoeff();
	MeshPreviewWindow* preview = NULL;
	if( enablePreview ) preview = new MeshPreviewWindow(fem.mesh.Get());
	fem.initializeMaterialModel(bodyID, LinearFEM::ISOTROPIC_NEOHOOKEAN, lameLambda, lameMu, density); printf("\n%% Neohookean material (la = %.4lg, mu = %.4lg, rho = %.4lg) ", lameLambda, lameMu, density);

	vtkSmartPointer<RotationInvariantViscosityModel> viscMdl = vtkSmartPointer<RotationInvariantViscosityModel>::New();
	fem.initializeViscosityModel(bodyID, viscMdl,  &viscosity ); printf("\n%% Rotation invariant viscosity (nu = %.4lg) ", viscosity);

	class GravityField : public VectorField{ public: virtual void eval(Eigen::Vector3d& g, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		g.setZero(); g[2]=-9.81;
		g = (Eigen::AngleAxisd(-0.0/180.0*M_PI, Eigen::Vector3d::UnitY())*g).eval();
	} } g;
	fem.setExternalAcceleration(bodyID,g);

	class FloorField : public DiffableScalarField{ public: virtual double eval(Eigen::Vector3d& n, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		double floorHeight = -0.05;
		n = (Eigen::AngleAxisd(0.0/180.0*M_PI, Eigen::Vector3d::UnitY())*Eigen::Vector3d::UnitZ()).eval().normalized(); // outward unit normal
		return n.dot(x)-floorHeight; // flat floor at (n'x)==floorHeight (positive outside, negative inside)
	} } floor;
	fem.addRigidObstacle(floor, frictionCoeff); printf("\n%% floor at -0.05, z-normal (cf = %.4lg) ", frictionCoeff);
	
	InitialConditionParameterHandler icQ( bodyID ); icQ.setVelocity=true; icQ.velocityGradients=true;
	icQ.setAngularVelocity=true; icQ.angularVelocityGradients=true; icQ.angularVelocityScale=2.0*M_PI;//1.0; //
	Eigen::VectorXd initCnds( icQ.getNumberOfParams(fem) ); initCnds.setZero();

	if( icQ.setVelocity )               initCnds.segment<3>(icQ.getVelocityIndex()) += 0.3*Eigen::Vector3d::UnitZ();
	if( icQ.setVelocity )               initCnds.segment<3>(icQ.getVelocityIndex()) += -0.5*Eigen::Vector3d::UnitX();
	if( icQ.setAngularVelocity ) initCnds.segment<3>(icQ.getAngularVelocityIndex()) += -4.0*Eigen::Vector3d::UnitY(); //10.0*Eigen::Vector3d::UnitY(); //

	cout << endl << "% initial conditions (" << initCnds.transpose() << ") ";
	icQ.setNewParams(initCnds,fem); icQ.applyInitialConditions(fem); fem.setResetPoseFromCurrent(); fem.setResetVelocitiesFromCurrent(); // slightly cumbersome way to set persistent initial conditions if the parameter handler is not used for optimization ...

	// optimize initial conditions
	ParameterHandler& theQ = icQ; Eigen::VectorXd& q = initCnds;

	AverageBoundaryValueObjectiveFunction phi;
	class TargetField : public VectorField{ public: virtual void eval(Eigen::Vector3d& g, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		g=x;
		if( t>=endtime ){g[0] = 0.1; g[1]=0.0;}
	}	TargetField(double t_end){endtime=t_end;} double endtime;
	} finalTarget(stoptime*0.5-0.5*timestep);
	phi.addTargetField(bndryID, &finalTarget); printf("\n%% Target at (0.1, 0.0, *) from t = %.4lf ", finalTarget.endtime );

	AdjointSensitivity analysis(phi,theQ,fem, false );//expect non-symmetric matrices   //DirectSensitivity analysis(phi,theQ,fem); //
	fileName.str(""); fileName.clear();
	fileName << outDir << argv[1];
	analysis.setupDynamicSim( timestep, nSteps, false, fileName.str(), preview );
	double objValue;
	
	if(0){
		fem.doPrintResiduals=false; //fem.useBDF2=false;
		Eigen::VectorXd objGradient, adjGradient;

		DirectSensitivity analysis(phi,theQ,fem);
		analysis.setupDynamicSim( timestep, nSteps, false, fileName.str(), preview );
		objValue = analysis( q, objGradient );

		AdjointSensitivity adjoint(phi,theQ,fem); adjoint.setAssumeSymmetricMatrices(false); // friction penalty forces will have asymmetric stiffness
		adjoint.setupDynamicSim( timestep, nSteps, false, fileName.str(), preview );
		double tmp = adjoint( q, adjGradient );

		printf("\n%% Direct v adjoint objective function evaluation ...\n");
		printf("\n%% ... value %.2lg =?= %.2lg  (abs diff %.2lg) \n", objValue,tmp, std::abs(objValue-tmp));
		printf("\n%% ... gradients\n");
		cout << "% ... [ " << objGradient.transpose() << " ]" << endl;
		cout << "% ... [ " << adjGradient.transpose() << " ]" << endl;
		printf("\n%% ... grad abs diff\n"); cout << "% ... [ " << (objGradient-adjGradient).cwiseAbs().transpose() << " ]" << endl;

		if( 1 ){ //finite-difference check dPhi/dq
			//AdjointSensitivity& analysis = adjoint;
			double err_6 = analysis.finiteDifferenceTest(q, 1e-6);
			double err_8 = analysis.finiteDifferenceTest(q, 1e-8);
			double err_10= analysis.finiteDifferenceTest(q, 1e-10);
			double err_12= analysis.finiteDifferenceTest(q, 1e-12);
			
			printf("\n%% max errors (direct-adjoint, FDe-6, FDe-8, FDe-10, FDe-12) ... ");
			printf("\n maxAbsErr = [ %.4lg , %.4lg , %.4lg , %.4lg , %.4lg ];", (objGradient-adjGradient).cwiseAbs().maxCoeff(), err_6, err_8, err_10, err_12);
			printf("\n maxFDerrDiff = [ 000 , %.4lg , %.4lg , %.4lg , 000 ];", std::abs(err_6-err_12), std::abs(err_8-err_12), std::abs(err_10-err_12));
		}

		return 0;
	}

	LBFGSpp::LBFGSParam<double> optimOptions = defaultLBFGSoptions<LBFGSpp::LBFGSParam<double> >();
	optimOptions.init_step=0.01; optimOptions.m = 50; // optimizing initial conditions for ballistic marker trajectories seems to benefit quite significantly from increased LBFGS memory
	LBFGSpp::LBFGSSolver<double> solver(optimOptions); //minimize with LBFGS
	int r;
	r = solver.minimize(analysis, q, objValue);
	q = analysis.bestRunParameters;

#ifdef _USE_OPTIMLIB
	// optimize with optimlib
	arma::vec iovals( q.size() );
	for(unsigned int i=0; i<q.size(); ++i){ iovals(i)=q(i); }
	//optim::algo_settings_t optimsets; optimsets.cg_method=6;
	//bool ok = optim::cg(iovals,optim_wrapper,&analysis,optimsets);
	optim::algo_settings_t optimsets; optimsets.gd_method = 7; optimsets.iter_max=200;
	bool ok = optim::gd(iovals,optim_wrapper,&analysis,optimsets);
	if( ok ){
		printf("\n%% optimlib success - params found: \n%%"); cout << iovals.t() << endl;
	}else printf("\n%% optimlib failed\n");
	q = analysis.bestRunParameters;
#else
	printf("\n%%!!! This example is supposed to use OPTIMLIB - results may differ if built without it !!!");
#endif

	r = solver.minimize(analysis, q, objValue);

	cout << endl << "best_q = [ " << analysis.bestRunParameters.transpose() << "]; ";
	cout << endl << "best_grad_phi = [ " << analysis.bestRunGradient.transpose() << "]; best_phi = " << analysis.bestRunPhiValue << "; ";

	if( preview != NULL ) delete preview;
	return 0;
}

int main_SphereMarkers_opt(int argc, char* argv[]){
	printf("\n%% === main_SphereMarkers_opt === \n\n");
	std::stringstream fileName;
	std::string outDir="../_out/";

	if( argc>3 ){
		std::string s(argv[3]);
		size_t lastPathSep = s.find_last_of("/\\")+1;
		size_t lastExtSep = s.find_last_of(".");
		//printf("\n%% lastPathSep %d, lastExtSep %d, substr \"%s\"", lastPathSep,lastExtSep, s.substr(lastPathSep, lastExtSep-lastPathSep).c_str());
		outDir.append(s.substr(lastPathSep, lastExtSep-lastPathSep).c_str()).append("/");
	}

	ContactFEM fem;
	if( argc>2 ){
		if(       std::string(argv[2]).find("qp")!=std::string::npos ){ fem.method=fem.CONTACT_QP; outDir.append("qp_");
		}else if( std::string(argv[2]).find("class")!=std::string::npos ){ fem.method=92; outDir.append("class_"); // we'll probably drop the first two options here eventually ...
		}else if( std::string(argv[2]).find("tanh")!=std::string::npos ){ fem.method=fem.CONTACT_TANH_PENALTY; outDir.append("tanh_");
		}else if( std::string(argv[2]).find("clamp")!=std::string::npos ){ fem.method=fem.CONTACT_CLAMP_PENALTY; outDir.append("clamp_");
		}else if( std::string(argv[2]).find("hyb")!=std::string::npos ){ fem.method=fem.CONTACT_HYBRID; outDir.append("hybrid_");
		}else printf("\n%% unknown option %s - ignored \n", argv[2]);
	}

#ifdef _WINDOWS
	// argv[1] contains the input mesh file name - we'll write outputs to outDir/inputMeshDir/inputMeshName... make sure inputMeshDir exists ...
	std::cout << "% output dir \"" << outDir << argv[1] << " ... ";
	std::string sc("@for %f in (\""); sc.append(outDir).append(argv[1]).append("\") do @mkdir %~dpf"); std::replace( sc.begin(), sc.end(), '/', '\\');
	int sr = system(sc.c_str());
	printf("\n\n");
#endif // _WINDOWS

	fem.useBDF2=true; //false; //
	fem.MAX_SOLVER_ITERS = 99;
	fem.FORCE_BALANCE_EPS = 1e-8;
	fem.FORCE_BALANCE_DELTA = 1e-12;
	fem.setPenaltyFactors(1e3, 1e3);
	if( fem.method == ContactFEM::CONTACT_HYBRID ) fem.setSlipDirectionRegularizer(1e-8);

	bool   enablePreview = false;
	double addNoise=0.0010; //negative or zero == no noise
	double
		timestep = 0.005, //0.001, //
		stoptime = 0.25, fulltime=0.0,
		part1_stoptime = 0.125,
		frictionCoeff = 0.4,
		viscosity = 0.0,//0.2,//
		lameLambda = 1e4,
		lameMu = 1e5,
		density = 150; // FlexFoam has roughly lambda = 1e4 Pa, mu = 1e5 Pa, rho = 150 kg/m^3
	unsigned int nSteps = (unsigned int)((stoptime+0.5*timestep)/timestep);
	
	fem.loadMeshFromElmerFiles(argv[1]);
	MeshPreviewWindow* preview = NULL;
	if( enablePreview ) preview = new MeshPreviewWindow(fem.mesh.Get());
	unsigned int bodyID = fem.bodyId.minCoeff();
	unsigned int bndryID = fem.bndId.minCoeff();
	if( 1 ){ // use different initial values for optimization test
		lameLambda = 1e3; lameMu = 1e4; frictionCoeff = 0.2; viscosity = 0.4;
	}else{
		density = 43e-3/fem.computeBodyVolume(); printf("\n%% density %.2lg (mass %.2lg) ", density, density*fem.computeBodyVolume() ); //adjust density for actual specimen
	}
	fem.initializeMaterialModel(bodyID, LinearFEM::ISOTROPIC_NEOHOOKEAN, lameLambda, lameMu, density);

	vtkSmartPointer<RotationInvariantViscosityModel> viscMdl = vtkSmartPointer<RotationInvariantViscosityModel>::New();
	viscMdl->setTimestep((fem.useBDF2?2.0/3.0:1.0)*timestep);
	fem.initializeViscosityModel(bodyID, viscMdl,  &viscosity );
	
	InitialConditionParameterHandler icQ( bodyID ); icQ.angularVelocityScale=1.0;
	icQ.setPostion=true; icQ.positionGradients=true; icQ.setOrientation=true; icQ.orientationGradients=true;
	icQ.setVelocity=true; icQ.velocityGradients=true; icQ.setAngularVelocity=true; icQ.angularVelocityGradients=true;
	Eigen::VectorXd initCnds( icQ.getNumberOfParams(fem) ); initCnds.setZero();

	if( 0 ){ // set ground truth initial condition data for comparison setup data generation
		initCnds << 0.0, 0.1, 0.05,	M_PI_4, 0.0, M_PI_4, -0.5, 0.0, 0.3, 0.0, -4.0,	0.0;
	}

	icQ.setNewParams(initCnds,fem); icQ.applyInitialConditions(fem); fem.setResetPoseFromCurrent(); fem.setResetVelocitiesFromCurrent(); // slightly cumbersome way to set persistent initial conditions if the parameter handler is not used for optimization ...
	
	GlobalElasticMaterialParameterHandler elastQ(bodyID);
	FrictionParameterHandler frictionQ(bodyID, (fem.useBDF2?2.0/3.0:1.0)*timestep);
	GlobalRotationInvariantViscosityMaterialParameterHandler viscQ( bodyID );
	CombinedParameterHandler elastFrictQ( elastQ, frictionQ ); //elastFrictQ.useLogOfParams=true;
	CombinedParameterHandler viscElastFrictQ( viscQ, elastFrictQ ); viscElastFrictQ.useLogOfParams=true;

	AverageBoundaryValueObjectiveFunction phi; double phiValue;
	std::vector<vtkSmartPointer<TemporalInterpolationField> > trackedTargets; std::vector<unsigned int> trackedIDs;
	std::vector<vtkSmartPointer<TemporalInterpolationField> > markerData; std::vector<std::string> markerNames;
	std::vector<vtkSmartPointer<PositionRotationInterpolationField> > rbData; std::vector<std::string> rbNames;
	bool optToMocapData=false;

	fileName.str(""); fileName.clear();
	if( argc>3 ) fileName << argv[3]; else fileName << argv[1] << "_targetFields.txt";
	if( fileName.str().find(".csv") != std::string::npos){ // Motion capture csv file
		printf("\n%% Loading mocap data file \"%s\" ...", fileName.str().c_str() );
		loadMocapDataFromCSV(fileName.str(), rbData,rbNames,markerData,markerNames, 1.0 /*file units assumed in metres*/);

		// assign labelled data to mesh boundary IDs
		for(unsigned int i=0; i<markerNames.size(); ++i){
			if( markerNames[i].find("s:xp")!=std::string::npos ){
				trackedIDs.push_back(1); printf("\n%% found marker data for s:xp ");
				trackedTargets.push_back(markerData[i]);
			}else
			if( markerNames[i].find("s:xm")!=std::string::npos ){
				trackedIDs.push_back(2); printf("\n%% found marker data for s:xm ");
				trackedTargets.push_back(markerData[i]);
			}else
			if( markerNames[i].find("s:yp")!=std::string::npos ){
				trackedIDs.push_back(3); printf("\n%% found marker data for s:yp ");
				trackedTargets.push_back(markerData[i]);
			}else
			if( markerNames[i].find("s:ym")!=std::string::npos ){
				trackedIDs.push_back(4); printf("\n%% found marker data for s:ym ");
				trackedTargets.push_back(markerData[i]);
			}else
			if( markerNames[i].find("s:zp")!=std::string::npos ){
				trackedIDs.push_back(5); printf("\n%% found marker data for s:zp ");
				trackedTargets.push_back(markerData[i]);
			}else
			if( markerNames[i].find("s:zm")!=std::string::npos ){
				trackedIDs.push_back(6); printf("\n%% found marker data for s:zm ");
				trackedTargets.push_back(markerData[i]);
			}
		}

		optToMocapData=true;
		// adjust the time-range of the simulation to cover the data
		double trange[2], ttmp[2];
		for(int i=0; i<trackedTargets.size(); ++i){
			trackedTargets[i]->getRange(ttmp);
			if( i==0 ){ trange[0]=ttmp[0]; trange[1]=ttmp[1]; }
			else{
				if( ttmp[0]<trange[0] ) trange[0]=ttmp[0];
				if( ttmp[1]>trange[1] ) trange[1]=ttmp[1];
			}
		}
		for(int i=0; i<trackedTargets.size(); ++i) trackedTargets[i]->t_shift = trange[0];

		//ToDo: allow setting part1_stoptime from input somewhere ...
		part1_stoptime = 0.25;
		stoptime=0.75;//
		fulltime=trange[1]-trange[0];//
		nSteps = (stoptime+0.5*timestep)/timestep; // also adjust the simulated time accordingly
		printf("\n%% Data time range (%.2lg %.2lg), simulated range (0 %.2lg), time step %.2lg (%u steps) ",trange[0],trange[1], stoptime, timestep, nSteps );

		if( icQ.setPostion ) initCnds.segment<3>(icQ.getPositionIndex()) += Eigen::Vector3d(0.0, 0.5, 0.5); // start well away from floor and wall
		icQ.setNewParams(initCnds,fem); icQ.applyInitialConditions(fem); fem.setResetPoseFromCurrent(); fem.setResetVelocitiesFromCurrent(); // slightly cumbersome way to set persistent initial conditions if the parameter handler is not used for optimization ...

	}else{
		fulltime=stoptime;
		if( addNoise > 0.0 ) std::srand(std::time(0));
		if( TemporalInterpolationField::buildFieldsFromTextFile(fileName.str(), trackedTargets, trackedIDs, addNoise/*add uniform noise*/) >= 0){
			printf("\n%% Target file \"%s\"", fileName.str().c_str());
			if( addNoise > 0.0 ) printf(" ... adding uniform noise in %.4lg*[-1;1] ", addNoise);
		}else return -1;
	}
	for(int i = 0; i < trackedIDs.size(); ++i) phi.addTargetField(trackedIDs[i], trackedTargets[i]);
	fileName.str(""); fileName.clear(); fileName << outDir << argv[1] << "_targetFields";
	TemporalInterpolationField::writeFieldsToVTU(fileName.str(), nSteps+1, trackedTargets, trackedIDs);

	class GravityField : public VectorField{ public: virtual void eval(Eigen::Vector3d& g, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const{
		g.setZero(); g[2]=-9.81;
		g = (Eigen::AngleAxisd(-0.0/180.0*M_PI, Eigen::Vector3d::UnitY())*g).eval();
	} } g;
	class GravityFieldY : public VectorField{ public: virtual void eval(Eigen::Vector3d& g, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const{
		g = -9.81*Eigen::Vector3d::UnitY();
	} } gY;
	class FloorField : public DiffableScalarField{ public: virtual double eval(Eigen::Vector3d& n, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const{
		double floorHeight = -0.05;//-0.05;//
		n = (Eigen::AngleAxisd(0.0/180.0*M_PI, Eigen::Vector3d::UnitY())*Eigen::Vector3d::UnitZ()).eval().normalized(); // outward unit normal
		return n.dot(x)-floorHeight; // flat floor at (n'x)==floorHeight (positive outside, negative inside)
	} } floor;
	class FloorFieldY : public DiffableScalarField{ public: virtual double eval(Eigen::Vector3d& n, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const{
		double floorHeight = 0.012; // mocap calibration puts the origin about 1cm below the calibration marker triangle
		n = Eigen::Vector3d::UnitY(); // outward unit normal
		return n.dot(x)-floorHeight; // flat floor at (n'x)==floorHeight (positive outside, negative inside)
	} } floorY;
	class WallFieldY : public DiffableScalarField{ public: virtual double eval(Eigen::Vector3d& n, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const{
		double wallLocation = 0.0;
		n = Eigen::Vector3d::UnitZ(); // outward unit normal
		return n.dot(x)-wallLocation; // flat floor at (n'x)==floorHeight (positive outside, negative inside)
	} } wallY;


	if( optToMocapData ){ // Y-up world with floor and wall
		fem.setExternalAcceleration(bodyID, gY);
		fem.addRigidObstacle(floorY, frictionCoeff);
		fem.addRigidObstacle(wallY, frictionCoeff);
	
	}else{ // basic test use Z-up world with floor
		fem.setExternalAcceleration(bodyID, g);
		fem.addRigidObstacle(floor, frictionCoeff);
	}

	if( 0 /*output initial guess*/){
		CombinedParameterHandler theQ( icQ, viscElastFrictQ );
		Eigen::VectorXd q( theQ.getNumberOfParams(fem) );
		theQ.getCurrentParams( q, fem );
		AdjointSensitivity analysis(phi,theQ,fem, false );//expect non-symmetric matrices
		fileName.str(""); fileName.clear();
		fileName << outDir << argv[1] << "_initialGuess";
		unsigned int nSteps = (fulltime+0.5*timestep)/timestep;
		analysis.setupDynamicSim( timestep, nSteps, false, fileName.str() );
		analysis(q,q); //return 0;
	}else if(0){ /*output previously computed solution*/
		CombinedParameterHandler theQ( icQ, viscElastFrictQ ); viscElastFrictQ.useLogOfParams=false;
		Eigen::VectorXd q( theQ.getNumberOfParams(fem) );

		// Solution for SphereMarkers_vid_take1_part2_11s BDF1 dt=0.001 hybrid
		//q <<  0.0616309,   0.567733,    1.22756,  -0.256048 , -0.835765 , -0.142514, 0.00665409, -0.0588751 , -0.818502  , -2.77761,  -0.509287,   0.407135, 0.00838225 ,   2479.96  ,  43342.9  ,  0.11363   ,     0.4;
		//theQ.setNewParams( q, fem );

		// Initial conditions for SphereMarkers_vid_take1_part2_11s BDF2 dt=0.005 
		initCnds << 0.0616114 ,  0.567807  ,  1.22752,  -0.257449 , -0.835156,  -0.143717 ,0.00687853, -0.0872608 , -0.818142 ,  -2.79854,  -0.506008 ,   0.39346 ; 
		icQ.setNewParams( initCnds, fem ); theQ.getCurrentParams( q, fem );

		AdjointSensitivity analysis(phi,theQ,fem, false );//expect non-symmetric matrices
		fileName.str(""); fileName.clear();
		fileName << outDir << argv[1] << "_solution";
		unsigned int nSteps = (fulltime+0.5*timestep)/timestep;
		analysis.setupDynamicSim( timestep, nSteps, false, fileName.str() );
		analysis(q,q);
		return 0;
	}

	fem.doPrintResiduals=false;

	bool doCompareSolution=false;
	if( 0 /*set material parameters to previously computed solution for comparison*/){
		// Solution for SphereMarkers_vid_take1_part2_11s clamp BDF2 dt=0.005 -- for comparing to take1_part1
		// q_all_opt = [ 0.0629641   0.563307    1.23009  -0.288068  -0.872725  -0.131378  0.0119076 -0.0785354  -0.807579   -2.79318  -0.502276   0.390664	49.8006  259237 11187.7 0.46607     0.4 ];
		Eigen::VectorXd sol( viscElastFrictQ.getNumberOfParams(fem) );
		sol << 49.8006,  259237, 11187.7, 0.46607,     0.4;
		bool tmp = viscElastFrictQ.useLogOfParams; viscElastFrictQ.useLogOfParams=false;
		viscElastFrictQ.setNewParams(sol, fem); // write material parameters to FEM object
		viscElastFrictQ.useLogOfParams=tmp;
		doCompareSolution=true;
		part1_stoptime = 0.4; stoptime=fulltime; nSteps = (unsigned int)((stoptime+0.5*timestep)/timestep);
	}

	if( 1 /*part 1: optimize initial conditions*/){
		ParameterHandler& theQ = icQ; Eigen::VectorXd& q = initCnds;
		unsigned int part1_nSteps = (unsigned int)((part1_stoptime+0.5*timestep)/timestep); //over reduced time range

		AdjointSensitivity analysis(phi,theQ,fem, false );//expect non-symmetric matrices   //DirectSensitivity analysis(phi,theQ,fem); //
		fileName.str(""); fileName.clear();
		fileName << outDir << argv[1] << "_ic";
		analysis.setupDynamicSim( timestep, part1_nSteps, false, fileName.str(), preview );
	
		LBFGSpp::LBFGSParam<double> optimOptions = defaultLBFGSoptions<LBFGSpp::LBFGSParam<double> >();
		optimOptions.init_step=0.01; optimOptions.m = 50; // optimizing initial conditions for ballistic marker trajectories seems to benefit quite significantly from increased LBFGS memory
		LBFGSpp::LBFGSSolver<double> solver(optimOptions); //minimize with LBFGS
		int r = solver.minimize(analysis, q, phiValue);
		fem.reset(); icQ.setNewParams(analysis.bestRunParameters,fem); icQ.applyInitialConditions(fem); fem.setResetPoseFromCurrent(); fem.setResetVelocitiesFromCurrent(); // fix initial conditions for following optimizations

		fileName.str(""); fileName.clear();
		fileName << outDir << argv[1] << "_result_ic";
		analysis.resetEvalCounter();
		analysis.setupDynamicSim( timestep, nSteps, false, fileName.str() );
		phiValue = analysis( analysis.bestRunParameters , analysis.bestRunGradient );

		printf("\n%% *** Part 1: initial conditions \n n_iters_ic = %d; n_sims_ic = %d; phi_ic = %.4lg; grad_phi_ic_norm = %.4lg \n%% *** \n", r,analysis.getEvalCounter(),phiValue,analysis.bestRunGradient.norm() );
		cout << " q_ic_opt = [ " << analysis.bestRunParameters.transpose() << " ]; " << endl << endl;
	}
	
	if( 1 && doCompareSolution /*part 1c: optimize initial conditions on full trajectory*/){
		ParameterHandler& theQ = icQ; Eigen::VectorXd& q = initCnds;

		AdjointSensitivity analysis(phi,theQ,fem, false );//expect non-symmetric matrices   //DirectSensitivity analysis(phi,theQ,fem); //
		fileName.str(""); fileName.clear();
		fileName << outDir << argv[1] << "_icfull";
		analysis.setupDynamicSim( timestep, nSteps, false, fileName.str(), preview );
	
		LBFGSpp::LBFGSParam<double> optimOptions = defaultLBFGSoptions<LBFGSpp::LBFGSParam<double> >();
		optimOptions.init_step=0.01; optimOptions.m = 50; // optimizing initial conditions for ballistic marker trajectories seems to benefit quite significantly from increased LBFGS memory
		LBFGSpp::LBFGSSolver<double> solver(optimOptions); //minimize with LBFGS
		int r = solver.minimize(analysis, q, phiValue);
		fem.reset(); icQ.setNewParams(analysis.bestRunParameters,fem); icQ.applyInitialConditions(fem); fem.setResetPoseFromCurrent(); fem.setResetVelocitiesFromCurrent(); // fix initial conditions for following optimizations

		fileName.str(""); fileName.clear();
		fileName << outDir << argv[1] << "_result_icfull";
		analysis.resetEvalCounter();
		analysis.setupDynamicSim( timestep, nSteps, false, fileName.str() );
		phiValue = analysis( analysis.bestRunParameters , analysis.bestRunGradient );

		printf("\n%% *** Part 1c: initial conditions \n n_iters_icf = %d; n_sims_icf = %d; phi_icf = %.4lg; grad_phi_icf_norm = %.4lg \n%% *** \n", r,analysis.getEvalCounter(),phiValue,analysis.bestRunGradient.norm() );
		cout << " q_icf_opt = [ " << analysis.bestRunParameters.transpose() << " ]; " << endl << endl;
	}

	if( 1 && !doCompareSolution /*part 2: optimize friction coefficient and elastic material parameters*/){
		ParameterHandler& theQ = viscElastFrictQ; //elastFrictQ;// frictionQ;//
		Eigen::VectorXd q( theQ.getNumberOfParams(fem) );
		theQ.getCurrentParams( q, fem );

		AdjointSensitivity analysis(phi,theQ,fem, false );//expect non-symmetric matrices   //DirectSensitivity analysis(phi,theQ,fem); //
		fileName.str(""); fileName.clear();
		fileName << outDir << argv[1] << "_mp";
		analysis.setupDynamicSim( timestep, nSteps, false, fileName.str(), preview );
	
		if(1){
#ifdef _USE_OPTIMLIB
			// optimize with optimlib
			arma::vec iovals( q.size() );
			for(unsigned int i=0; i<q.size(); ++i){ iovals(i)=q(i); }
			optim::algo_settings_t optimsets; optimsets.gd_method = 7; optimsets.iter_max=75;
			bool ok = optim::gd(iovals,optim_wrapper,&analysis,optimsets);
			if( ok ){
				printf("\n%% optimlib success - params found: \n%%"); cout << iovals.t() << endl;
			}else printf("\n%% optimlib failed\n");
			q = analysis.bestRunParameters;
			printf("\n%% starting LBFGS after %d ADAM sim runs. ",analysis.getEvalCounter());
#else
			printf("\n%%!!! This example is supposed to use OPTIMLIB - results may differ if built without it !!!");
#endif
		}

		LBFGSpp::LBFGSParam<double> optimOptions = defaultLBFGSoptions<LBFGSpp::LBFGSParam<double> >();
		optimOptions.init_step=0.01; optimOptions.m = 50; // optimizing initial conditions for ballistic marker trajectories seems to benefit quite significantly from increased LBFGS memory
		LBFGSpp::LBFGSSolver<double> solver(optimOptions); //minimize with LBFGS
		int r = solver.minimize(analysis, q, phiValue);

		fileName.str(""); fileName.clear();
		fileName << outDir << argv[1] << "_result_mp";
		analysis.resetEvalCounter();
		analysis.setupDynamicSim( timestep, nSteps, false, fileName.str() );
		phiValue = analysis( analysis.bestRunParameters , analysis.bestRunGradient );

		printf("\n%% *** Part 2: material parameters \n n_iters_mp = %d; n_sims_mp = %d; phi_mp = %.4lg; grad_phi_mp_norm = %.4lg \n%% *** \n", r,analysis.getEvalCounter(),phiValue,analysis.bestRunGradient.norm() );
		cout << " q_mp_opt = [ " << analysis.bestRunParameters.array().exp().transpose() << " ]; " << endl << endl;

	}

	if( 1 && !doCompareSolution  /*part 3: optimize everything*/){
		CombinedParameterHandler theQ( icQ, viscElastFrictQ ); viscElastFrictQ.useLogOfParams=true; // Note: initial cond. handler must come first
		nSteps = (fulltime+0.5*timestep)/timestep;
		Eigen::VectorXd q( theQ.getNumberOfParams(fem) );
		theQ.getCurrentParams( q, fem );

		AdjointSensitivity analysis(phi,theQ,fem, false );//expect non-symmetric matrices   //DirectSensitivity analysis(phi,theQ,fem); //
		fileName.str(""); fileName.clear();
		fileName << outDir << argv[1] << "_all";
		analysis.setupDynamicSim( timestep, nSteps, false, fileName.str(), preview );
	
		LBFGSpp::LBFGSParam<double> optimOptions = defaultLBFGSoptions<LBFGSpp::LBFGSParam<double> >();
		optimOptions.init_step=0.01; optimOptions.m = 50; // optimizing initial conditions for ballistic marker trajectories seems to benefit quite significantly from increased LBFGS memory
		LBFGSpp::LBFGSSolver<double> solver(optimOptions); //minimize with LBFGS
		int r = solver.minimize(analysis, q, phiValue);

		fileName.str(""); fileName.clear();
		fileName << outDir << argv[1] << "_result_all";
		analysis.resetEvalCounter();
		nSteps = (fulltime+0.5*timestep)/timestep;
		analysis.setupDynamicSim( timestep, nSteps, false, fileName.str() );
		phiValue = analysis( analysis.bestRunParameters , analysis.bestRunGradient );

		printf("\n%% *** Part 3: all parameters \n n_iters_all = %d; n_sims_all = %d; phi_all = %.4lg; grad_phi_all_norm = %.4lg \n%% *** \n", r,analysis.getEvalCounter(),phiValue,analysis.bestRunGradient.norm() );
		cout << " q_all_opt = [ " << analysis.bestRunParameters.segment(0,icQ.getNumberOfParams(fem)).transpose() << "\t" << analysis.bestRunParameters.segment(icQ.getNumberOfParams(fem),analysis.bestRunParameters.size()-icQ.getNumberOfParams(fem)).array().exp().transpose() << " ]; " << endl << endl;

	}
		
	/*/

	fem.doPrintResiduals=false;
	DirectSensitivity analysis(phi,theQ,fem); //
	analysis.setupDynamicSim( timestep, nSteps );
	Eigen::VectorXd objGradient;

	//// sample objective function along x-z initial velocity ...
	//Eigen::VectorXd xVel,zVel;
	//xVel.resize(51); zVel.resize(81); //xVel.resize(15); zVel.resize(17); //xVel.resize(5); zVel.resize(7); //
	////xVel.setLinSpaced( 0.45 , 0.65 ); zVel.setLinSpaced(-0.4, 0.3);
	////xVel.setLinSpaced( 0.522 , 0.527 ); zVel.setLinSpaced(-0.312, -0.307); // solution for lameMu=1e4 is roughly v0=[0.525026 0.00503517  -0.309689]
	//xVel.setLinSpaced( 0.575 , 0.578 ); zVel.setLinSpaced(-0.151, -0.145); // solution for lameMu=1e3 is roughly v0=[0.57612 0.0135116 -0.148542]
	//Eigen::MatrixXd phiVals;  phiVals.resize( xVel.size(), zVel.size() );
	//Eigen::MatrixXd dphi_dvx; dphi_dvx.resize( xVel.size(), zVel.size() );
	//Eigen::MatrixXd dphi_dvz; dphi_dvz.resize( xVel.size(), zVel.size() );
	//printf("\n%% Sampling objective function: x = [%lf : %lf : %lf], z = [%lf : %lf : %lf]", xVel.minCoeff(), xVel(1)-xVel(0), xVel.maxCoeff(), zVel.minCoeff(), zVel(1)-zVel(0), zVel.maxCoeff());
	//for(unsigned int i=0; i<xVel.size(); ++i) for(unsigned int j=0; j<zVel.size(); ++j){
	//	printf("\n%% %5d/%d ... ", zVel.size()*i+j+1, xVel.size()*zVel.size());
	//	q.setZero();
	//	q(0) = xVel(i);
	//	q(2) = zVel(j);
	//	phiVals(i,j) = analysis( q, objGradient );
	//	dphi_dvx(i,j) = objGradient(0);
	//	dphi_dvz(i,j) = objGradient(2);
	//}
	//fileName.str(""); fileName.clear(); fileName << outDir << argv[1] << "_objFcnSamples.m";
	//ofstream outFile(fileName.str());
	//outFile << endl << " xVel = [ " << xVel.transpose() << " ];" << endl;
	//outFile << endl << " zVel = [ " << zVel.transpose() << " ];" << endl;
	//outFile << endl << " phiVals = [ "  << endl << phiVals  << " ];" << endl;
	//outFile << endl << " dphi_dvx = [ " << endl << dphi_dvx << " ];" << endl;
	//outFile << endl << " dphi_dvz = [ " << endl << dphi_dvz << " ];" << endl;

	// sample objective function along x-ry initial velocity ...
	q.setZero(); q(2)=0.3; // ground truth is [0.2 0 0.3 0 -10 0] // 0.233012  0.00518844    0.309836  0.00164286 -0.00671362
	Eigen::VectorXd xVel,ryVel;
	xVel.resize(11); ryVel.resize(13);  //xVel.resize(31); ryVel.resize(31);
	xVel.setLinSpaced( 0.0 , 0.4 ); ryVel.setLinSpaced(-0.3, 0.1); // assuming angularVelocityScale=100.0
	theQ.angularVelocityScale=100.0;
	Eigen::MatrixXd phiVals;  phiVals.resize( xVel.size(), ryVel.size() );
	Eigen::MatrixXd dphi_dvx; dphi_dvx.resize( xVel.size(), ryVel.size() );
	Eigen::MatrixXd dphi_dvry; dphi_dvry.resize( xVel.size(), ryVel.size() );
	printf("\n%% Sampling objective function: x = [%lf : %lf : %lf], ry = [%lf : %lf : %lf]", xVel.minCoeff(), xVel(1)-xVel(0), xVel.maxCoeff(), ryVel.minCoeff(), ryVel(1)-ryVel(0), ryVel.maxCoeff());
	for(unsigned int i=0; i<xVel.size(); ++i) for(unsigned int j=0; j<ryVel.size(); ++j){
		printf("\n%% %5d/%d ... ", ryVel.size()*i+j+1, xVel.size()*ryVel.size());
		q(0) = xVel(i);
		q(4) = ryVel(j);
		phiVals(i,j) = analysis( q, objGradient );
		dphi_dvx(i,j) = objGradient(0);
		dphi_dvry(i,j) = objGradient(4);
	}
	fileName.str(""); fileName.clear(); fileName << outDir << argv[1] << "_objFcnSamplesRot.m";
	ofstream outFile(fileName.str());
	outFile << endl << " xVel = [ " << xVel.transpose() << " ];" << endl;
	outFile << endl << " ryVel = [ " << ryVel.transpose() << " ];" << endl;
	outFile << endl << " phiVals = [ "  << endl << phiVals  << " ];" << endl;
	outFile << endl << " dphi_dvx = [ " << endl << dphi_dvx << " ];" << endl;
	outFile << endl << " dphi_dvry = [ " << endl << dphi_dvry << " ];" << endl;


	//fileName.str(""); fileName.clear();
	//fileName << outDir << argv[1] << "_" << std::setfill ('0') << std::setw(5) << (0);
	//fem.saveMeshToVTUFile(fileName.str());
	//theQ.setNewParams( q , fem);
	//theQ.applyInitialConditions(fem);
	//fem.assembleMassAndExternalForce();
	//for(unsigned int step=0; step<nSteps; ++step){
	//	printf("\n%% (%5d/%d) ", step+1,nSteps);
	//	fem.updateAllBoundaryData();
	//	fem.dynamicImplicitTimestep(timestep);
	//	fileName.str(""); fileName.clear();
	//	fileName << outDir << argv[1] << "_" << std::setfill ('0') << std::setw(5) << (step+1);
	//	fem.saveMeshToVTUFile(fileName.str());
	//	// update preview
	//	preview.render();
	//}	printf("\n");
	//Eigen::VectorXd phi_x,phi_v,phi_f,phi_q;
	//phi.evaluate(fem,phi_x,phi_v,phi_f,phi_q);
	//vtkSmartPointer<vtkDoubleArray> vtkPhiX = vtkSmartPointer<vtkDoubleArray>::New();
	//vtkPhiX->SetName("phi_x"); vtkPhiX->SetNumberOfComponents(3); vtkPhiX->SetNumberOfTuples(fem.getNumberOfNodes());
	//for(unsigned int i=0; i<fem.getNumberOfNodes(); ++i) vtkPhiX->SetTuple3(i, phi_x(fem.getNodalDof(i,fem.X_DOF)), phi_x(fem.getNodalDof(i,fem.Y_DOF)), phi_x(fem.getNodalDof(i,fem.Z_DOF)) );
	//fem.mesh->GetPointData()->AddArray(vtkPhiX);
	//fem.saveMeshToVTUFile(fileName.str());
	/**/

	if( preview != NULL ) delete preview;
	return 0;
}

int main_CubeMarkers_opt(int argc, char* argv[]){
	printf("\n%% === main_CubeMarkers_opt === \n\n");
	std::stringstream fileName;
	std::string outDir="../_out/";

	if( argc>3 ){
		std::string s(argv[3]);
		size_t lastPathSep = s.find_last_of("/\\")+1;
		size_t lastExtSep = s.find_last_of(".");
		//printf("\n%% lastPathSep %d, lastExtSep %d, substr \"%s\"", lastPathSep,lastExtSep, s.substr(lastPathSep, lastExtSep-lastPathSep).c_str());
		outDir.append(s.substr(lastPathSep, lastExtSep-lastPathSep).c_str()).append("/");
	}

	ContactFEM fem;
	if( argc>2 ){
		if(       std::string(argv[2]).find("qp")!=std::string::npos ){ fem.method=fem.CONTACT_QP; outDir.append("qp_");
		}else if( std::string(argv[2]).find("class")!=std::string::npos ){ fem.method=92; outDir.append("class_"); // we'll probably drop the first two options here eventually ...
		}else if( std::string(argv[2]).find("tanh")!=std::string::npos ){ fem.method=fem.CONTACT_TANH_PENALTY; outDir.append("tanh_");
		}else if( std::string(argv[2]).find("clamp")!=std::string::npos ){ fem.method=fem.CONTACT_CLAMP_PENALTY; outDir.append("clamp_");
		}else if( std::string(argv[2]).find("hyb")!=std::string::npos ){ fem.method=fem.CONTACT_HYBRID; outDir.append("hybrid_");
		}else printf("\n%% unknown option %s - ignored \n", argv[2]);
	}

#ifdef _WINDOWS
	// argv[1] contains the input mesh file name - we'll write outputs to outDir/inputMeshDir/inputMeshName... make sure inputMeshDir exists ...
	std::cout << "% output dir \"" << outDir << argv[1] << " ... ";
	std::string sc("@for %f in (\""); sc.append(outDir).append(argv[1]).append("\") do @mkdir %~dpf"); std::replace( sc.begin(), sc.end(), '/', '\\');
	int sr = system(sc.c_str());
	printf("\n\n");
#endif // _WINDOWS

	fem.useBDF2=true;
	fem.MAX_SOLVER_ITERS = 99;
	fem.FORCE_BALANCE_EPS = 1e-8;
	fem.FORCE_BALANCE_DELTA = 1e-12;
	fem.setPenaltyFactors(1e3, 1e3);
	if( fem.method == ContactFEM::CONTACT_HYBRID ) fem.setSlipDirectionRegularizer(1e-8);

	bool   enablePreview = false;
	double addNoise=0.0; //negative or zero == no noise
	double
		timestep = 0.005, //0.002, //
		stoptime = 0.25,
		part1_stoptime = 0.18,// 0.125,
		frictionCoeff = 0.4,
		viscosity = 4.2, //0.2, //
		lameLambda = 1e4,
		lameMu = 1e5,
		density = 150; // FlexFoam has roughly lambda = 1e4 Pa, mu = 1e5 Pa, rho = 150 kg/m^3
	unsigned int nSteps = (unsigned int)((stoptime+0.5*timestep)/timestep);
	lameLambda = 1e2; lameMu = 2e4; // start with softer initial guess -- cube is well expanded type III foam -- quite soft
	
	fem.loadMeshFromElmerFiles(argv[1]); density = 24e-3/fem.computeBodyVolume(); printf("\n%% density %.2lg (mass %.2lg) ", density, density*fem.computeBodyVolume() ); //adjust density for actual specimen
	MeshPreviewWindow* preview = NULL;
	if( enablePreview ) preview = new MeshPreviewWindow(fem.mesh.Get());
	unsigned int bodyID = fem.bodyId.minCoeff();
	unsigned int bndryID = fem.bndId.minCoeff();
	fem.initializeMaterialModel(bodyID, LinearFEM::ISOTROPIC_NEOHOOKEAN, lameLambda, lameMu, density);

	vtkSmartPointer<RotationInvariantViscosityModel> viscMdl = vtkSmartPointer<RotationInvariantViscosityModel>::New();
	viscMdl->setTimestep((fem.useBDF2?2.0/3.0:1.0)*timestep);
	fem.initializeViscosityModel(bodyID, viscMdl,  &viscosity );
	
	InitialConditionParameterHandler icQ( bodyID ); icQ.angularVelocityScale=1.0;
	icQ.setPostion=true; icQ.positionGradients=true; icQ.setOrientation=true; icQ.orientationGradients=true;
	icQ.setVelocity=true; icQ.velocityGradients=true; icQ.setAngularVelocity=true; icQ.angularVelocityGradients=true;
	Eigen::VectorXd initCnds( icQ.getNumberOfParams(fem) ); initCnds.setZero();
	icQ.setNewParams(initCnds,fem); icQ.applyInitialConditions(fem); fem.setResetPoseFromCurrent(); fem.setResetVelocitiesFromCurrent(); // slightly cumbersome way to set persistent initial conditions if the parameter handler is not used for optimization ...
	
	GlobalElasticMaterialParameterHandler elastQ(bodyID);
	FrictionParameterHandler frictionQ(bodyID, (fem.useBDF2?2.0/3.0:1.0)*timestep);
	GlobalRotationInvariantViscosityMaterialParameterHandler viscQ( bodyID );
	CombinedParameterHandler elastFrictQ( elastQ, frictionQ ); //elastFrictQ.useLogOfParams=true;
	CombinedParameterHandler viscElastFrictQ( viscQ, elastFrictQ ); viscElastFrictQ.useLogOfParams=true;
	CombinedParameterHandler theQ( icQ, viscElastFrictQ ); // Note: when combining an InitialConditionParameterHandler with another CombinedParameterHandler (which does not include initial conditions, but derives from InitialConditionParameterHandler) the InitialConditionParameterHandler must be listed FIRST !!!


	AverageBoundaryValueObjectiveFunction phi; double phiValue;
	std::vector<vtkSmartPointer<TemporalInterpolationField> > trackedTargets; std::vector<unsigned int> trackedIDs;
	std::vector<vtkSmartPointer<TemporalInterpolationField> > markerData; std::vector<std::string> markerNames;
	std::vector<vtkSmartPointer<PositionRotationInterpolationField> > rbData; std::vector<std::string> rbNames;
	bool optToMocapData=false;

	fileName.str(""); fileName.clear();
	if( argc>3 ) fileName << argv[3]; else fileName << argv[1] << "_targetFields.txt";
	if( fileName.str().find(".csv") != std::string::npos){ // Motion capture csv file
		printf("\n%% Loading mocap data file \"%s\" ...", fileName.str().c_str() );
		loadMocapDataFromCSV(fileName.str(), rbData,rbNames,markerData,markerNames, 1.0 /*file units assumed in metres*/);

		// assign labelled data to mesh boundary IDs
		for(unsigned int i=0; i<markerNames.size(); ++i){
			if( markerNames[i].find("s:xp")!=std::string::npos ){
				trackedIDs.push_back(1); printf("\n%% found marker data for s:xp ");
				trackedTargets.push_back(markerData[i]);
			}else
			if( markerNames[i].find("s:xm")!=std::string::npos ){
				trackedIDs.push_back(2); printf("\n%% found marker data for s:xm ");
				trackedTargets.push_back(markerData[i]);
			}else
			if( markerNames[i].find("s:yp")!=std::string::npos ){
				trackedIDs.push_back(3); printf("\n%% found marker data for s:yp ");
				trackedTargets.push_back(markerData[i]);
			}else
			if( markerNames[i].find("s:ym")!=std::string::npos ){
				trackedIDs.push_back(4); printf("\n%% found marker data for s:ym ");
				trackedTargets.push_back(markerData[i]);
			}else
			if( markerNames[i].find("s:zp")!=std::string::npos ){
				trackedIDs.push_back(5); printf("\n%% found marker data for s:zp ");
				trackedTargets.push_back(markerData[i]);
			}else
			if( markerNames[i].find("s:zm")!=std::string::npos ){
				trackedIDs.push_back(6); printf("\n%% found marker data for s:zm ");
				trackedTargets.push_back(markerData[i]);
			}
		}

		optToMocapData=true;
		// adjust the time-range of the simulation to cover the data
		double trange[2], ttmp[2];
		for(int i=0; i<trackedTargets.size(); ++i){
			trackedTargets[i]->getRange(ttmp);
			if( i==0 ){ trange[0]=ttmp[0]; trange[1]=ttmp[1]; }
			else{
				if( ttmp[0]<trange[0] ) trange[0]=ttmp[0];
				if( ttmp[1]>trange[1] ) trange[1]=ttmp[1];
			}
		}
		for(int i=0; i<trackedTargets.size(); ++i) trackedTargets[i]->t_shift = trange[0];

		//ToDo: allow setting part1_stoptime from input somewhere ...
		part1_stoptime = 0.25;
		stoptime=trange[1]-trange[0];//
		nSteps = (stoptime+0.5*timestep)/timestep; // also adjust the simulated time accordingly
		printf("\n%% Data time range (%.2lg %.2lg), simulated range (0 %.2lg), time step %.2lg (%u steps) ",trange[0],trange[1], stoptime, timestep, nSteps );

		if( icQ.setPostion ) initCnds.segment<3>(icQ.getPositionIndex()) += Eigen::Vector3d(0.0, 0.5, 0.5); // start well away from floor and wall
		icQ.setNewParams(initCnds,fem); icQ.applyInitialConditions(fem); fem.setResetPoseFromCurrent(); fem.setResetVelocitiesFromCurrent(); // slightly cumbersome way to set persistent initial conditions if the parameter handler is not used for optimization ...

	}else{
		return -1;
	}
	for(int i = 0; i < trackedIDs.size(); ++i) phi.addTargetField(trackedIDs[i], trackedTargets[i]);
	fileName.str(""); fileName.clear(); fileName << outDir << argv[1] << "_targetFields";
	TemporalInterpolationField::writeFieldsToVTU(fileName.str(), nSteps+1, trackedTargets, trackedIDs);

	class GravityFieldY : public VectorField{ public: virtual void eval(Eigen::Vector3d& g, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const{
		g = -9.81*Eigen::Vector3d::UnitY();
	} } gY;
	class FloorFieldY : public DiffableScalarField{ public: virtual double eval(Eigen::Vector3d& n, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const{
		double floorHeight = 0.012; // mocap calibration puts the origin about 1cm below the calibration marker triangle
		n = Eigen::Vector3d::UnitY(); // outward unit normal
		return n.dot(x)-floorHeight; // flat floor at (n'x)==floorHeight (positive outside, negative inside)
	} } floorY;
	class WallFieldY : public DiffableScalarField{ public: virtual double eval(Eigen::Vector3d& n, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const{
		double wallLocation = 0.0;
		n = Eigen::Vector3d::UnitZ(); // outward unit normal
		return n.dot(x)-wallLocation; // flat floor at (n'x)==floorHeight (positive outside, negative inside)
	} } wallY;

	fem.setExternalAcceleration(bodyID, gY);
	fem.addRigidObstacle(floorY, frictionCoeff);
	fem.addRigidObstacle(wallY, frictionCoeff);
	
	bool doCompareSolution=false;
	if( 1 /*set material parameters to previously computed solution for comparison*/){
		// Solution for t1p1 clamp BDF2 dt=0.005 -- for comparing to t1p3
		// q_13_opt = [   0.102394   0.265965   0.632113   0.201116   0.895828   0.446169   0.291025  -0.296414  -0.388836 -0.0280618 -0.0708823  -0.047844	 4.27518  99.9976   9526.1 0.398808      0.4 ]; 
		Eigen::VectorXd sol( viscElastFrictQ.getNumberOfParams(fem) );
		//sol << 49.8006,  259237, 11187.7, 0.46607,     0.4; //SphereMarkers solution
		//sol << 4.27518,  99.9976,   9526.1, 0.398808,    0.4; // old version CubeMarkers
		//sol <<  0.0861192,  7359.23,   81559.4,  0.344478,  0.984287; // another attempt
		sol << 5.74652, 13369.4, 5612.53, 1.16386,     0.4; //maybe final version
		bool tmp = viscElastFrictQ.useLogOfParams; viscElastFrictQ.useLogOfParams=false;
		viscElastFrictQ.setNewParams(sol, fem); // write material parameters to FEM object
		viscElastFrictQ.useLogOfParams=tmp;
		doCompareSolution=true;
		part1_stoptime = 0.15;
	}

	if( 1 /*output initial guess*/){
		CombinedParameterHandler theQ( viscElastFrictQ, icQ );
		Eigen::VectorXd q( theQ.getNumberOfParams(fem) );
		theQ.getCurrentParams( q, fem );
		AdjointSensitivity analysis(phi,theQ,fem, false );//expect non-symmetric matrices
		fileName.str(""); fileName.clear();
		fileName << outDir << argv[1] << "_initialGuess";
		analysis.setupDynamicSim( timestep, nSteps, false, fileName.str() );
		analysis(q,q);
	}

	fem.doPrintResiduals=false;

	if( 0 && !doCompareSolution /* continuation in stoptime for all parameters */){
		unsigned int nStepsAll = nSteps, nParts=stoptime/0.15; // parts in 0.15s increments
		unsigned int nStepsInc = nStepsAll/nParts;
		for(unsigned int part=0; part<nParts; ++part){
			nSteps=nStepsInc*(part+1);

			Eigen::VectorXd q( theQ.getNumberOfParams(fem) );
			theQ.getCurrentParams( q, fem );

			AdjointSensitivity analysis(phi,theQ,fem, false );//expect non-symmetric matrices   //DirectSensitivity analysis(phi,theQ,fem); //
			fileName.str(""); fileName.clear();
			fileName << outDir << argv[1] << "_" << part;
			analysis.setupDynamicSim( timestep, nSteps, false, fileName.str(), preview );
	
			LBFGSpp::LBFGSParam<double> optimOptions = defaultLBFGSoptions<LBFGSpp::LBFGSParam<double> >();
			optimOptions.init_step=0.01; optimOptions.m = 50; // optimizing initial conditions for ballistic marker trajectories seems to benefit quite significantly from increased LBFGS memory
			if( part<(nParts-2) ){ optimOptions.delta = 1e-4; } //reduce accuracy for partial solutions
			LBFGSpp::LBFGSSolver<double> solver(optimOptions); //minimize with LBFGS
			int r = solver.minimize(analysis, q, phiValue);

			fileName.str(""); fileName.clear();
			fileName << outDir << argv[1] << "_result_" << part;
			analysis.resetEvalCounter();
			analysis.setupDynamicSim( timestep, nStepsAll, false, fileName.str() );
			phiValue = analysis( analysis.bestRunParameters , analysis.bestRunGradient );
			theQ.setNewParams( analysis.bestRunParameters, fem );

			printf("\n%% *** Part %d\n n_iters_%d = %d; n_sims_%d = %d; phi_%d = %.4lg; grad_phi_%d_norm = %.4lg \n%% *** \n", part,part, r,part,analysis.getEvalCounter(),part,phiValue,part,analysis.bestRunGradient.norm() );
			if( viscElastFrictQ.useLogOfParams ){
				cout << " q_" << part << "_opt = [ " << analysis.bestRunParameters.segment(0,icQ.getNumberOfParams(fem)).transpose() << "\t" << analysis.bestRunParameters.segment(icQ.getNumberOfParams(fem),analysis.bestRunParameters.size()-icQ.getNumberOfParams(fem)).array().exp().transpose() << " ]; " << endl << endl;
			}else{
				cout << " q_" << part << "_opt = [ " << analysis.bestRunParameters.transpose() << " ]; " << endl << endl;
			}
		}
	}

	if( 1 || doCompareSolution /*part 1: optimize initial conditions*/){
		ParameterHandler& theQ = icQ; Eigen::VectorXd& q = initCnds;
		unsigned int part1_nSteps = (unsigned int)((part1_stoptime+0.5*timestep)/timestep); //over reduced time range

		AdjointSensitivity analysis(phi,theQ,fem, false );//expect non-symmetric matrices   //DirectSensitivity analysis(phi,theQ,fem); //
		fileName.str(""); fileName.clear();
		fileName << outDir << argv[1] << "_ic";
		analysis.setupDynamicSim( timestep, part1_nSteps, false, fileName.str(), preview );
	
		LBFGSpp::LBFGSParam<double> optimOptions = defaultLBFGSoptions<LBFGSpp::LBFGSParam<double> >();
		optimOptions.init_step=0.01; optimOptions.m = 50; // optimizing initial conditions for ballistic marker trajectories seems to benefit quite significantly from increased LBFGS memory
		LBFGSpp::LBFGSSolver<double> solver(optimOptions); //minimize with LBFGS
		int r = solver.minimize(analysis, q, phiValue);
		fem.reset(); icQ.setNewParams(analysis.bestRunParameters,fem); icQ.applyInitialConditions(fem); fem.setResetPoseFromCurrent(); fem.setResetVelocitiesFromCurrent(); // fix initial conditions for following optimizations

		fileName.str(""); fileName.clear();
		fileName << outDir << argv[1] << "_result_ic";
		analysis.resetEvalCounter();
		analysis.setupDynamicSim( timestep, nSteps, false, fileName.str() );
		phiValue = analysis( analysis.bestRunParameters , analysis.bestRunGradient );

		printf("\n%% *** Part 1: initial conditions \n n_iters_ic = %d; n_sims_ic = %d; phi_ic = %.4lg; grad_phi_ic_norm = %.4lg \n%% *** \n", r,analysis.getEvalCounter(),phiValue,analysis.bestRunGradient.norm() );
		cout << " q_ic_opt = [ " << analysis.bestRunParameters.transpose() << " ]; " << endl << endl;
	}

	if( 0 && doCompareSolution /*part 1c: optimize initial conditions on full trajectory*/){
		ParameterHandler& theQ = icQ; Eigen::VectorXd& q = initCnds;

		AdjointSensitivity analysis(phi,theQ,fem, false );//expect non-symmetric matrices   //DirectSensitivity analysis(phi,theQ,fem); //
		fileName.str(""); fileName.clear();
		fileName << outDir << argv[1] << "_icfull";
		analysis.setupDynamicSim( timestep, nSteps, false, fileName.str(), preview );
	
		LBFGSpp::LBFGSParam<double> optimOptions = defaultLBFGSoptions<LBFGSpp::LBFGSParam<double> >();
		optimOptions.init_step=0.01; optimOptions.m = 50; // optimizing initial conditions for ballistic marker trajectories seems to benefit quite significantly from increased LBFGS memory
		LBFGSpp::LBFGSSolver<double> solver(optimOptions); //minimize with LBFGS
		int r = solver.minimize(analysis, q, phiValue);
		fem.reset(); icQ.setNewParams(analysis.bestRunParameters,fem); icQ.applyInitialConditions(fem); fem.setResetPoseFromCurrent(); fem.setResetVelocitiesFromCurrent(); // fix initial conditions for following optimizations

		fileName.str(""); fileName.clear();
		fileName << outDir << argv[1] << "_result_icfull";
		analysis.resetEvalCounter();
		analysis.setupDynamicSim( timestep, nSteps, false, fileName.str() );
		phiValue = analysis( analysis.bestRunParameters , analysis.bestRunGradient );

		printf("\n%% *** Part 1c: initial conditions \n n_iters_icf = %d; n_sims_icf = %d; phi_icf = %.4lg; grad_phi_icf_norm = %.4lg \n%% *** \n", r,analysis.getEvalCounter(),phiValue,analysis.bestRunGradient.norm() );
		cout << " q_icf_opt = [ " << analysis.bestRunParameters.transpose() << " ]; " << endl << endl;
	}

	if( 1 && !doCompareSolution /*part 2: optimize friction coefficient and elastic material parameters*/){
		unsigned int nStepsP2 = (0.4+0.5*timestep)/timestep; // stop after first bounce (for take1_part1)
		ParameterHandler& theQ = viscElastFrictQ; //elastFrictQ;// frictionQ;//
		Eigen::VectorXd q( theQ.getNumberOfParams(fem) );
		theQ.getCurrentParams( q, fem );

		AdjointSensitivity analysis(phi,theQ,fem, false );//expect non-symmetric matrices   //DirectSensitivity analysis(phi,theQ,fem); //
		fileName.str(""); fileName.clear();
		fileName << outDir << argv[1] << "_mp";
		analysis.setupDynamicSim( timestep, nStepsP2, false, fileName.str(), preview );
	
		if(1){
#ifdef _USE_OPTIMLIB
			// optimize with optimlib
			arma::vec iovals( q.size() );
			for(unsigned int i=0; i<q.size(); ++i){ iovals(i)=q(i); }
			optim::algo_settings_t optimsets; optimsets.gd_method = 7; optimsets.iter_max=75;
			bool ok = optim::gd(iovals,optim_wrapper,&analysis,optimsets);
			if( ok ){
				printf("\n%% optimlib success - params found: \n%%"); cout << iovals.t() << endl;
			}else printf("\n%% optimlib failed\n");
			q = analysis.bestRunParameters;
			printf("\n%% starting LBFGS after %d ADAM sim runs. ",analysis.getEvalCounter());
#else
			printf("\n%%!!! This example is supposed to use OPTIMLIB - results may differ if built without it !!!");
#endif
		}

		LBFGSpp::LBFGSParam<double> optimOptions = defaultLBFGSoptions<LBFGSpp::LBFGSParam<double> >();
		optimOptions.init_step=0.01; optimOptions.m = 50; // optimizing initial conditions for ballistic marker trajectories seems to benefit quite significantly from increased LBFGS memory
		LBFGSpp::LBFGSSolver<double> solver(optimOptions); //minimize with LBFGS
		int r = solver.minimize(analysis, q, phiValue);

		fileName.str(""); fileName.clear();
		fileName << outDir << argv[1] << "_result_mp";
		analysis.resetEvalCounter();
		analysis.setupDynamicSim( timestep, nSteps, false, fileName.str() );
		phiValue = analysis( analysis.bestRunParameters , analysis.bestRunGradient );

		printf("\n%% *** Part 2: material parameters \n n_iters_mp = %d; n_sims_mp = %d; phi_mp = %.4lg; grad_phi_mp_norm = %.4lg \n%% *** \n", r,analysis.getEvalCounter(),phiValue,analysis.bestRunGradient.norm() );
		cout << " q_mp_opt = [ " << analysis.bestRunParameters.array().exp().transpose() << " ]; " << endl << endl;

	}
	
	if( 1 && !doCompareSolution /*part 3: optimize everything*/){
		Eigen::VectorXd q( theQ.getNumberOfParams(fem) );
		theQ.getCurrentParams( q, fem );

		AdjointSensitivity analysis(phi,theQ,fem, false );//expect non-symmetric matrices   //DirectSensitivity analysis(phi,theQ,fem); //
		fileName.str(""); fileName.clear();
		fileName << outDir << argv[1] << "_all";
		analysis.setupDynamicSim( timestep, nSteps, false, fileName.str(), preview );
	
		LBFGSpp::LBFGSParam<double> optimOptions = defaultLBFGSoptions<LBFGSpp::LBFGSParam<double> >();
		optimOptions.init_step=0.01; optimOptions.m = 50; // optimizing initial conditions for ballistic marker trajectories seems to benefit quite significantly from increased LBFGS memory
		LBFGSpp::LBFGSSolver<double> solver(optimOptions); //minimize with LBFGS
		int r = solver.minimize(analysis, q, phiValue);

		fileName.str(""); fileName.clear();
		fileName << outDir << argv[1] << "_result_all";
		analysis.resetEvalCounter();
		analysis.setupDynamicSim( timestep, nSteps, false, fileName.str() );
		phiValue = analysis( analysis.bestRunParameters , analysis.bestRunGradient );

		printf("\n%% *** Part 3: all parameters \n n_iters_all = %d; n_sims_all = %d; phi_all = %.4lg; grad_phi_all_norm = %.4lg \n%% *** \n", r,analysis.getEvalCounter(),phiValue,analysis.bestRunGradient.norm() );
		cout << " q_all_opt = [ " << analysis.bestRunParameters.segment(0,icQ.getNumberOfParams(fem)).transpose() << "\t" << analysis.bestRunParameters.segment(icQ.getNumberOfParams(fem),analysis.bestRunParameters.size()-icQ.getNumberOfParams(fem)).array().exp().transpose() << " ]; " << endl << endl;

	}

	if( preview != NULL ) delete preview;
	return 0;
}

int main_contact_opt_Bunny_OLD(int argc, char* argv[]){
	printf("\n%% === main_contact_opt_Bunny === \n\n");
	std::stringstream fileName;
	std::string outDir="../_out/";

	ContactFEM fem;
	if( argc>2 ){
		if(       std::string(argv[2]).find("qp")!=std::string::npos ){ fem.method=fem.CONTACT_QP; outDir.append("qp_");
		}else if( std::string(argv[2]).find("tanh")!=std::string::npos ){ fem.method=fem.CONTACT_TANH_PENALTY; outDir.append("tanh_");
		}else if( std::string(argv[2]).find("clamp")!=std::string::npos ){ fem.method=fem.CONTACT_CLAMP_PENALTY; outDir.append("clamp_");
		}else if( std::string(argv[2]).find("hyb")!=std::string::npos ){ fem.method=fem.CONTACT_HYBRID; outDir.append("hybrid_");
		}else printf("\n%% unknown option %s - ignored \n", argv[2]);
	}

#ifdef _WINDOWS
	// argv[1] contains the input mesh file name - we'll write outputs to outDir/inputMeshDir/inputMeshName... make sure inputMeshDir exists ...
	std::cout << "% output dir \"" << outDir << argv[1] << " ... ";
	std::string sc("@for %f in (\""); sc.append(outDir).append(argv[1]).append("\") do @mkdir %~dpf"); std::replace( sc.begin(), sc.end(), '/', '\\');
	int sr = system(sc.c_str());
	printf("\n\n");
#endif // _WINDOWS

	fem.useBDF2=false;
	fem.MAX_SOLVER_ITERS = 99;
	fem.FORCE_BALANCE_EPS = 1e-10;
	fem.FORCE_BALANCE_DELTA = 1e-12;
	fem.setPenaltyFactors(1e3, 1e3);
	if( fem.method == ContactFEM::CONTACT_HYBRID ) fem.setSlipDirectionRegularizer(1e-8);
	fem.doPrintResiduals = false;

	bool enablePreview = false;
	double
		timestep = 0.005,
		stoptime = 1.0,
		frictionCoeff = 0.2,
		viscosity = -0.1, // zero or negative to disable
		lameLambda = 1e2,
		lameMu = 1e4,
		density = 90;
	unsigned int nSteps = (unsigned int)((stoptime+0.5*timestep)/timestep);

	fem.loadMeshFromElmerFiles(argv[1]);
	unsigned int bodyID = fem.bodyId.minCoeff();
	
	Remesher remesh; remesh.setTargetEdgeLengthFromMesh(fem.mesh); remesh.dOpt[22]=1e5;
	remesh.targetEdgeLength *= 2.0; lameMu = 1.2e4;//5.0; //  //use stiffer material on higher resolution mesh
	remesh.remesh(fem); printf("\n%% remeshed to %d elems \n", fem.getNumberOfElems()); 

	MeshPreviewWindow* preview = NULL;
	if( enablePreview ) preview = new MeshPreviewWindow(fem.mesh.Get());
	fem.initializeMaterialModel(bodyID, LinearFEM::ISOTROPIC_NEOHOOKEAN, lameLambda, lameMu, density); printf("\n%% Neohookean material (la = %.4lg, mu = %.4lg, rho = %.4lg) ", lameLambda, lameMu, density);

	vtkSmartPointer<RotationInvariantViscosityModel> viscMdl = vtkSmartPointer<RotationInvariantViscosityModel>::New();
	if( viscosity>0.0 ){ fem.initializeViscosityModel(bodyID, viscMdl,  &viscosity ); printf("\n%% Rotation invariant viscosity (nu = %.4lg) ", viscosity);}

	class GravityField : public VectorField{ public: virtual void eval(Eigen::Vector3d& g, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		g.setZero(); g[2]=-9.81;
		g = (Eigen::AngleAxisd(-0.0/180.0*M_PI, Eigen::Vector3d::UnitY())*g).eval();
	} } g;
	fem.setExternalAcceleration(bodyID,g);

	class FloorField : public DiffableScalarField{ public: virtual double eval(Eigen::Vector3d& n, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		double floorHeight = -0.05;
		n = (Eigen::AngleAxisd(0.0/180.0*M_PI, Eigen::Vector3d::UnitY())*Eigen::Vector3d::UnitZ()).eval().normalized(); // outward unit normal
		return n.dot(x)-floorHeight; // flat floor at (n'x)==floorHeight (positive outside, negative inside)
	} } floor;
	fem.addRigidObstacle(floor, frictionCoeff); printf("\n%% floor at -0.05, z-normal (cf = %.4lg) ", frictionCoeff);
	
	class WallField : public DiffableScalarField{ public: virtual double eval(Eigen::Vector3d& n, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		double wallX = 0.1;
		n = -Eigen::Vector3d::UnitY();
		return n.dot(x)+wallX;
	} } wall;
	fem.addRigidObstacle(wall, 2.0*frictionCoeff); printf("\n%% wall at 0.1, -y-normal (cf = %.4lg) ", 2.0*frictionCoeff); //second example: with wall

	InitialConditionParameterHandler icQ( bodyID ); icQ.setVelocity=true; icQ.velocityGradients=true;

	ObjectiveFunction unused;
	DirectSensitivity firstRun(unused,icQ,fem); //ToDo: Adjoint + hybrid contacts has a strange bug here ... seems to only affect this particular sim though
	Eigen::VectorXd initCndsZero( icQ.getNumberOfParams(fem) ); initCndsZero.setZero();
	firstRun.setupDynamicSim( timestep, nSteps );
	firstRun(initCndsZero,initCndsZero); initCndsZero.setZero(); // single run
	fileName.str(""); fileName.clear();
	fileName << outDir << argv[1];
	fem.saveMeshToVTUFile( fileName.str()+"_target" );
	class NodalObjectiveFunction : public ObjectiveFunction{ public:
		Vect3Map::PlainMatrix targetX; double fromTime; bool uprightOnly;
		NodalObjectiveFunction(double fromTime_) : fromTime(fromTime_) {uprightOnly=false;}
		double evaluate( LinearFEM& fem, Eigen::VectorXd& phi_x, Eigen::VectorXd& phi_v, Eigen::VectorXd& phi_f, Eigen::VectorXd& phi_q){
			double phi=0.0; Eigen::Vector3d u;
			phi_x.resize( fem.getNumberOfNodes()*fem.N_DOFS ); phi_x.setZero();
			if( fem.simTime >= fromTime && (targetX.size() == fem.deformedCoords.size() )){
				for(unsigned int i=0; i<fem.getNumberOfNodes(); ++i){
					if( uprightOnly ){
						u.setZero(); u(2) = fem.getDeformedCoord(i)(2) - targetX.block<fem.N_DOFS,1>(0,i)(2);
					}else{
						u = fem.getDeformedCoord(i) - targetX.block<fem.N_DOFS,1>(0,i);
					}
					phi += 0.5*u.dot(u);
					phi_x.segment<fem.N_DOFS>(fem.N_DOFS*i) += u;
				}
			}
			return phi;
		}
	//} phi(stoptime-0.5*timestep); phi.targetX = fem.deformedCoords;
	} phi(stoptime-100.5*timestep); phi.targetX = fem.deformedCoords; phi.uprightOnly=true; // second example: not full target pose, but just upright orientation on landing

	icQ.setAngularVelocity=true; icQ.angularVelocityScale=/**/M_PI*2.0*/**/1.0; icQ.angularVelocityGradients=true;
	icQ.setPostion=true; icQ.setOrientation=true; icQ.orientationGradients=false; icQ.positionGradients=false; // set a starting position and orientation, but do not optimize for it
	Eigen::VectorXd initCnds( icQ.getNumberOfParams(fem) ); initCnds.setZero();
	
	if( icQ.setPostion ) initCnds.segment<3>(icQ.getPositionIndex()) += 0.2*Eigen::Vector3d::UnitX() ;//+ 0.3*Eigen::Vector3d::UnitZ();
	
	//first example: continuation in starting position ...
	// close to solved:       q = [          0.2            0            0            0            0            0    -0.986351   -0.0556847    -0.157683  -0.00944372     0.220151 -0.000474328 ];
	// start from there and lift starting point up a bit ... (using angularVelocityScale=M_PI*2.0)
	//initCnds <<          0.2 ,           0      ,      0.1    ,        0 ,           0    ,        0 ,   -0.986351  , -0.0556847  ,  -0.157683 , -0.00944372   ,  0.220151, -0.000474328 ;
	// close to solved:    q = [        0.2          0        0.1          0          0          0  -0.946017 -0.0398673  -0.170921 -0.0175132   0.210067  0.0115826 ];
	// start from there and move starting point forward a bit ... (using angularVelocityScale=M_PI*2.0)
	//initCnds <<       0.2     ,     -0.05    ,    0.1      ,    0.0  ,        0.0   ,       0.0,  -0.946017, -0.0398673,  -0.170921, -0.0175132  , 0.210067 , 0.0115826 ;
	// roughly solved:  q = [        0.2      -0.05        0.1          0          0          0  -0.936149   0.130532  -0.176639 -0.0693811   0.177799  0.0105728 ];
	// start from there and move starting point forward even more ... (using angularVelocityScale=M_PI*2.0)
	//initCnds <<       0.2   ,   -0.2    ,    0.1    ,      0  ,        0       ,   0 , -0.936149  , 0.130532 , -0.176639, -0.0693811  , 0.177799,  0.0105728 ; // maybe too much ...
	// got stuck (slight instability) - best params were (try restart): q = [       0.2      -0.2       0.1         0         0         0 -0.953988   1.14297 -0.423456 -0.203743  0.244628 0.0457835 ];
	//initCnds <<  0.2    ,  -0.2   ,    0.1    ,     0   ,      0    ,     0, -0.953988 ,  1.14297 ,-0.423456, -0.203743 , 0.244628 ,0.0457835 ;
	// more careful next step ...
	//initCnds <<       0.2   ,   -0.1    ,    0.1    ,      0  ,        0       ,   0 , -0.936149  , 0.130532 , -0.176639, -0.0693811  , 0.177799,  0.0105728 ;
	// close to solved: q = [       0.2      -0.1       0.1         0         0         0 -0.920776  0.286909 -0.192007 -0.108822  0.185753 0.0461324 ];
	// next step:
	//initCnds <<   0.2   ,   -0.15  ,     0.1   ,      0   ,      0  ,       0, -0.920776 , 0.286909 ,-0.192007 ,-0.108822 , 0.185753 ,0.0461324 ;
	// as good as solved:      q = [       0.2     -0.15       0.1         0         0         0 -0.892056  0.418719 -0.204244 -0.172068  0.173502  0.102268 ];
	// one more step ...
	//initCnds <<       0.2  ,   -0.2   ,    0.1  ,       0  ,       0    ,     0, -0.892056  ,0.418719, -0.204244, -0.172068,  0.173502,  0.102268 ;
	// close to solved: q = [       0.2      -0.2       0.1         0         0         0 -0.874024  0.581335 -0.185959 -0.147079  0.128764 0.0975694 ];
	// and another step ... finds an inaccurate solution with the wall ... try again without the wall ... still not great anymore?
	//initCnds <<     0.3 ,     -0.2     ,  0.1  ,       0   ,      0  ,       0 ,-0.874024  ,0.581335 ,-0.185959 ,-0.147079 , 0.128764, 0.0975694 ;

	// optimize initial conditions
	ParameterHandler& theQ = icQ; Eigen::VectorXd& q = initCnds;
	AdjointSensitivity analysis(phi,theQ,fem, false ); printf("\n%% adjoint analysis ");//expect non-symmetric matrices   //DirectSensitivity analysis(phi,theQ,fem); //

	if( 1 ){ //second example initial conditions
		initCnds.setZero();
		if( icQ.setPostion ) initCnds.segment<3>(icQ.getPositionIndex()) += Eigen::Vector3d( 0.2, -0.4, 0.3 ) ;
		if( icQ.setVelocity ) initCnds.segment<3>(icQ.getVelocityIndex()) += Eigen::Vector3d(  0.0, 0.8, 1.6 ) ;
		if( icQ.setAngularVelocity ) initCnds.segment<3>(icQ.getAngularVelocityIndex()) += Eigen::Vector3d(  0.5*M_PI, -0.25*M_PI, 0.0 ) ;

		/**/
		//output initial guess
		fileName.str(""); fileName.clear(); fileName << outDir << argv[1] << "_initialGuess";
		analysis.setupDynamicSim( timestep, nSteps, false, fileName.str(), preview );
		analysis(initCnds, analysis.bestRunGradient); //return 0;

		/*/ // sample objective function in initial spin (x,y) 10x10 step 0.02
		analysis.setupDynamicSim( timestep, nSteps );
		Eigen::VectorXd rotVx(10), rotVy(10),objGradient(initCnds);
		//rotVx.setLinSpaced( 0.5*M_PI - 0.1, 0.5*M_PI + 0.1 ); rotVy.setLinSpaced( -0.25*M_PI - 0.1, -0.25*M_PI + 0.1 );
		rotVx.setLinSpaced( 0.0, M_PI ); rotVy.setLinSpaced( -0.5*M_PI, 0.0 );


		Eigen::MatrixXd phiVals;  phiVals.resize( rotVx.size(), rotVy.size() );
		Eigen::MatrixXd dphi_dvx; dphi_dvx.resize( rotVx.size(), rotVy.size() );
		Eigen::MatrixXd dphi_dvz; dphi_dvz.resize( rotVx.size(), rotVy.size() );
		printf("\n%% Sampling objective function: rx = [%lf : %lf : %lf], ry = [%lf : %lf : %lf]", rotVx.minCoeff(), rotVx(1)-rotVx(0), rotVx.maxCoeff(), rotVy.minCoeff(), rotVy(1)-rotVy(0), rotVy.maxCoeff());
		for(unsigned int i=0; i<rotVx.size(); ++i) for(unsigned int j=0; j<rotVy.size(); ++j){
			unsigned int runID = rotVy.size()*i+j+1;
			fileName.str(""); fileName.clear(); fileName << outDir << argv[1] << "_run" << runID;
			analysis.setupDynamicSim( timestep, nSteps ,false, fileName.str() );
			printf("\n%% %5d/%d ... ", runID, rotVx.size()*rotVy.size());
			initCnds.segment<3>(icQ.getAngularVelocityIndex())(0) = rotVx(i);
			initCnds.segment<3>(icQ.getAngularVelocityIndex())(1) = rotVy(j);
			objGradient.setZero();
			phiVals(i,j) = analysis( initCnds, objGradient );
			dphi_dvx(i,j) = objGradient.segment<3>(icQ.getAngularVelocityIndex())(0);
			dphi_dvz(i,j) = objGradient.segment<3>(icQ.getAngularVelocityIndex())(1);
		}
		fileName.str(""); fileName.clear(); fileName << outDir << argv[1] << "_objFcnSamples.m";
		ofstream outFile(fileName.str());
		outFile << endl << " xVel = [ " << rotVx.transpose() << " ];" << endl;
		outFile << endl << " zVel = [ " << rotVy.transpose() << " ];" << endl;
		outFile << endl << " phiVals = [ "  << endl << phiVals  << " ];" << endl;
		outFile << endl << " dphi_dvx = [ " << endl << dphi_dvx << " ];" << endl;
		outFile << endl << " dphi_dvz = [ " << endl << dphi_dvz << " ];" << endl;
		outFile.close();
		return 0;
		/**/
	}

	cout << endl << "% initial conditions (" << initCnds.transpose() << ") ";
	//icQ.setNewParams(initCnds,fem); icQ.applyInitialConditions(fem); fem.setResetPoseFromCurrent(); fem.setResetVelocitiesFromCurrent(); // slightly cumbersome way to set persistent initial conditions if the parameter handler is not used for optimization ...

	fileName.str(""); fileName.clear();
	fileName << outDir << argv[1];
	analysis.setupDynamicSim( timestep, nSteps, false, fileName.str(), preview ); printf("\n%% timestep %.2lg, #steps %u", timestep, nSteps);
	double objValue;
	
	if( 0 ){
		LBFGSpp::LBFGSParam<double> optimOptions = defaultLBFGSoptions<LBFGSpp::LBFGSParam<double> >();
		optimOptions.init_step=0.01; optimOptions.wolfe = 1.0-1e-3;
		LBFGSpp::LBFGSSolver<double> solver(optimOptions); //minimize with LBFGS
		int r = solver.minimize(analysis, q, objValue);
		q = analysis.bestRunParameters;
	}else{// optimize with CMA from SCP
		cout << endl << "% CMA optimization (5000,16,0.05,1e-16,1e-8) bound +/-1e2" << endl;
		std::vector<double> fixedValues; std::vector<unsigned int> fixedIDs;
		if( icQ.setPostion ){
			fixedValues.push_back( q[icQ.getPositionIndex()  ]);
			fixedValues.push_back( q[icQ.getPositionIndex()+1]);
			fixedValues.push_back( q[icQ.getPositionIndex()+2]);
			fixedIDs.push_back( icQ.getPositionIndex()  );
			fixedIDs.push_back( icQ.getPositionIndex()+1);
			fixedIDs.push_back( icQ.getPositionIndex()+2);
		}
		if( icQ.setOrientation ){
			fixedValues.push_back( q[icQ.getOrientationIndex()  ]);
			fixedValues.push_back( q[icQ.getOrientationIndex()+1]);
			fixedValues.push_back( q[icQ.getOrientationIndex()+2]);
			fixedIDs.push_back( icQ.getOrientationIndex()  );
			fixedIDs.push_back( icQ.getOrientationIndex()+1);
			fixedIDs.push_back( icQ.getOrientationIndex()+2);
		}
		CMAObjectiveFunction cmaobj(analysis,fixedValues,fixedIDs);
		CMAMinimizer cmaopt(5000,16,0.05,1e-16,1e-8);
		bool ret;
		Eigen::VectorXd q_lower(q.size()), q_upper(q.size());
		q_lower.setConstant(-1e2); q_upper.setConstant(1e2);
		cmaopt.setBounds(q_lower,q_upper);
		ret = cmaopt.minimize(&cmaobj,q,objValue);
		cout << endl << "% CMA returned " << ret;
		q = analysis.bestRunParameters;
	}

	cout << endl << "best_q = [ " << analysis.bestRunParameters.transpose() << "]; ";
	cout << endl << "best_grad_phi = [ " << analysis.bestRunGradient.transpose() << "]; best_phi = " << analysis.bestRunPhiValue << "; ";
	/**/

	if( preview != NULL ) delete preview;
	return 0;
}

int main_contact_opt_Bunny(int argc, char* argv[]){
	printf("\n%% === main_contact_opt_Bunny === \n\n");
	std::stringstream fileName;
	std::string outDir="../_out/";

	ContactFEM fem;
	if( argc>2 ){
		if(       std::string(argv[2]).find("qp")!=std::string::npos ){ fem.method=fem.CONTACT_QP; outDir.append("qp_");
		}else if( std::string(argv[2]).find("tanh")!=std::string::npos ){ fem.method=fem.CONTACT_TANH_PENALTY; outDir.append("tanh_");
		}else if( std::string(argv[2]).find("clamp")!=std::string::npos ){ fem.method=fem.CONTACT_CLAMP_PENALTY; outDir.append("clamp_");
		}else if( std::string(argv[2]).find("hyb")!=std::string::npos ){ fem.method=fem.CONTACT_HYBRID; outDir.append("hybrid_");
		}else printf("\n%% unknown option %s - ignored \n", argv[2]);
	}

#ifdef _WINDOWS
	// argv[1] contains the input mesh file name - we'll write outputs to outDir/inputMeshDir/inputMeshName... make sure inputMeshDir exists ...
	std::cout << "% output dir \"" << outDir << argv[1] << " ... ";
	std::string sc("@for %f in (\""); sc.append(outDir).append(argv[1]).append("\") do @mkdir %~dpf"); std::replace( sc.begin(), sc.end(), '/', '\\');
	int sr = system(sc.c_str());
	printf("\n\n");
#endif // _WINDOWS

	fem.useBDF2=false;
	fem.MAX_SOLVER_ITERS = 99;
	fem.FORCE_BALANCE_EPS = 1e-10;
	fem.FORCE_BALANCE_DELTA = 1e-12;
	fem.setPenaltyFactors(1e3, 1e3);
	if( fem.method == ContactFEM::CONTACT_HYBRID ) fem.setSlipDirectionRegularizer(1e-8);
	fem.doPrintResiduals = false;

	bool enablePreview = false;
	double
		timestep = 0.005,
		stoptime = 1.0,
		frictionCoeff = 0.2,
		viscosity = -0.1, // zero or negative to disable
		lameLambda = 1e2,
		lameMu = 1e4,
		density = 90;
	unsigned int nSteps = (unsigned int)((stoptime+0.5*timestep)/timestep);

	fem.loadMeshFromElmerFiles(argv[1]);
	unsigned int bodyID = fem.bodyId.minCoeff();
	
	Remesher remesh; remesh.setTargetEdgeLengthFromMesh(fem.mesh); remesh.dOpt[22]=1e5;
	remesh.targetEdgeLength *= 2.0; lameMu = 1.2e4;//5.0; //  //use stiffer material on higher resolution mesh
	remesh.remesh(fem); printf("\n%% remeshed to %d elems \n", fem.getNumberOfElems()); 

	MeshPreviewWindow* preview = NULL;
	if( enablePreview ) preview = new MeshPreviewWindow(fem.mesh.Get());
	fem.initializeMaterialModel(bodyID, LinearFEM::ISOTROPIC_NEOHOOKEAN, lameLambda, lameMu, density); printf("\n%% Neohookean material (la = %.4lg, mu = %.4lg, rho = %.4lg) ", lameLambda, lameMu, density);

	vtkSmartPointer<RotationInvariantViscosityModel> viscMdl = vtkSmartPointer<RotationInvariantViscosityModel>::New();
	if( viscosity>0.0 ){ fem.initializeViscosityModel(bodyID, viscMdl,  &viscosity ); printf("\n%% Rotation invariant viscosity (nu = %.4lg) ", viscosity);}

	class GravityField : public VectorField{ public: virtual void eval(Eigen::Vector3d& g, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		g.setZero(); g[2]=-9.81;
		g = (Eigen::AngleAxisd(-0.0/180.0*M_PI, Eigen::Vector3d::UnitY())*g).eval();
	} } g;
	fem.setExternalAcceleration(bodyID,g);

	class FloorField : public DiffableScalarField{ public: virtual double eval(Eigen::Vector3d& n, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		double floorHeight = -0.05;
		n = (Eigen::AngleAxisd(0.0/180.0*M_PI, Eigen::Vector3d::UnitY())*Eigen::Vector3d::UnitZ()).eval().normalized(); // outward unit normal
		return n.dot(x)-floorHeight; // flat floor at (n'x)==floorHeight (positive outside, negative inside)
	} } floor;
	fem.addRigidObstacle(floor, frictionCoeff); printf("\n%% floor at -0.05, z-normal (cf = %.4lg) ", frictionCoeff);
	
	//class WallField : public DiffableScalarField{ public: virtual double eval(Eigen::Vector3d& n, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
	//	double wallX = 0.2; // v2 was 0.1
	//	n = -Eigen::Vector3d::UnitY();
	//	return n.dot(x)+wallX;
	//} } wall;
	//fem.addRigidObstacle(wall, 2.0*frictionCoeff); printf("\n%% wall at 0.2, -y-normal (cf = %.4lg) ", 2.0*frictionCoeff); //second example: with wall

	InitialConditionParameterHandler icQ( bodyID ); icQ.setVelocity=true; icQ.velocityGradients=true;

	ObjectiveFunction unused;
	DirectSensitivity firstRun(unused,icQ,fem); //ToDo: Adjoint + hybrid contacts has a strange bug here ... seems to only affect this particular sim though
	Eigen::VectorXd initCndsZero( icQ.getNumberOfParams(fem) ); initCndsZero.setZero();
	firstRun.setupDynamicSim( timestep, nSteps );
	firstRun(initCndsZero,initCndsZero); initCndsZero.setZero(); // single run
	fileName.str(""); fileName.clear();
	fileName << outDir << argv[1];
	fem.saveMeshToVTUFile( fileName.str()+"_target" );
	class NodalObjectiveFunction : public ObjectiveFunction{ public:
		Vect3Map::PlainMatrix targetX; double fromTime; double uprightFactor,uprightTime;
		NodalObjectiveFunction(double fromTime_) : fromTime(fromTime_) {uprightFactor=0.0;uprightTime=0.0;}
		double evaluate( LinearFEM& fem, Eigen::VectorXd& phi_x, Eigen::VectorXd& phi_v, Eigen::VectorXd& phi_f, Eigen::VectorXd& phi_q){
			double phi=0.0; Eigen::Vector3d u;
			phi_x.resize( fem.getNumberOfNodes()*fem.N_DOFS ); phi_x.setZero();
			if((fem.simTime >= fromTime || (uprightFactor > 0.0 && fem.simTime >= uprightTime)) && (targetX.size() == fem.deformedCoords.size() )){
				for(unsigned int i=0; i<fem.getNumberOfNodes(); ++i){
					if( uprightFactor > 0.0 && fem.simTime >= uprightTime ){
						u.setZero(); u(2) = fem.getDeformedCoord(i)(2) - targetX.block<fem.N_DOFS,1>(0,i)(2);
						phi += uprightFactor*0.5*u.dot(u);
						phi_x.segment<fem.N_DOFS>(fem.N_DOFS*i) += uprightFactor*u;
					}
					if( fem.simTime >= fromTime ){
						u = fem.getDeformedCoord(i) - targetX.block<fem.N_DOFS,1>(0,i);
						phi += 0.5*u.dot(u);
						phi_x.segment<fem.N_DOFS>(fem.N_DOFS*i) += u;
					}
				}
			}
			return phi;
		}
	} phi(stoptime-0.5*timestep); phi.targetX = fem.deformedCoords;
	phi.uprightFactor=0.5; // v2 was 0.005 // v3 was 0.02 // v4 was 0.5
	phi.uprightTime = (stoptime-100.5*timestep);

	icQ.setAngularVelocity=true; icQ.angularVelocityScale=/**/M_PI*2.0*/**/1.0; icQ.angularVelocityGradients=true;
	icQ.setPostion=true; icQ.setOrientation=true; icQ.orientationGradients=false; icQ.positionGradients=false; // set a starting position and orientation, but do not optimize for it
	Eigen::VectorXd initCnds( icQ.getNumberOfParams(fem) );

	// optimize initial conditions
	ParameterHandler& theQ = icQ; Eigen::VectorXd& q = initCnds;
	AdjointSensitivity analysis(phi,theQ,fem, false ); printf("\n%% adjoint analysis ");//expect non-symmetric matrices   //DirectSensitivity analysis(phi,theQ,fem); //

	initCnds.setZero();
	if( icQ.setPostion ) initCnds.segment<3>(icQ.getPositionIndex()) += Eigen::Vector3d( 0.5, -0.4, 0.3 ) ; // v1 was ( 0.2, -0.4, 0.3 ) 
	if( icQ.setVelocity ) initCnds.segment<3>(icQ.getVelocityIndex()) += Eigen::Vector3d(  0.0, 0.8, 1.6 ) ;
	if( icQ.setAngularVelocity ) initCnds.segment<3>(icQ.getAngularVelocityIndex()) += Eigen::Vector3d(  0.5*M_PI, -0.25*M_PI, 0.0 ) ;

	/**/
	//output initial guess
	fileName.str(""); fileName.clear(); fileName << outDir << argv[1] << "_initialGuess";
	analysis.setupDynamicSim( timestep, nSteps, false, fileName.str(), preview );
	analysis(initCnds, analysis.bestRunGradient); //return 0;

	/*/ // sample objective function in initial spin (x,y) 10x10 step 0.02
	analysis.setupDynamicSim( timestep, nSteps );
	Eigen::VectorXd rotVx(10), rotVy(10),objGradient(initCnds);
	//rotVx.setLinSpaced( 0.5*M_PI - 0.1, 0.5*M_PI + 0.1 ); rotVy.setLinSpaced( -0.25*M_PI - 0.1, -0.25*M_PI + 0.1 );
	rotVx.setLinSpaced( 0.0, M_PI ); rotVy.setLinSpaced( -0.5*M_PI, 0.0 );


	Eigen::MatrixXd phiVals;  phiVals.resize( rotVx.size(), rotVy.size() );
	Eigen::MatrixXd dphi_dvx; dphi_dvx.resize( rotVx.size(), rotVy.size() );
	Eigen::MatrixXd dphi_dvz; dphi_dvz.resize( rotVx.size(), rotVy.size() );
	printf("\n%% Sampling objective function: rx = [%lf : %lf : %lf], ry = [%lf : %lf : %lf]", rotVx.minCoeff(), rotVx(1)-rotVx(0), rotVx.maxCoeff(), rotVy.minCoeff(), rotVy(1)-rotVy(0), rotVy.maxCoeff());
	for(unsigned int i=0; i<rotVx.size(); ++i) for(unsigned int j=0; j<rotVy.size(); ++j){
		unsigned int runID = rotVy.size()*i+j+1;
		fileName.str(""); fileName.clear(); fileName << outDir << argv[1] << "_run" << runID;
		analysis.setupDynamicSim( timestep, nSteps ,false, fileName.str() );
		printf("\n%% %5d/%d ... ", runID, rotVx.size()*rotVy.size());
		initCnds.segment<3>(icQ.getAngularVelocityIndex())(0) = rotVx(i);
		initCnds.segment<3>(icQ.getAngularVelocityIndex())(1) = rotVy(j);
		objGradient.setZero();
		phiVals(i,j) = analysis( initCnds, objGradient );
		dphi_dvx(i,j) = objGradient.segment<3>(icQ.getAngularVelocityIndex())(0);
		dphi_dvz(i,j) = objGradient.segment<3>(icQ.getAngularVelocityIndex())(1);
	}
	fileName.str(""); fileName.clear(); fileName << outDir << argv[1] << "_objFcnSamples.m";
	ofstream outFile(fileName.str());
	outFile << endl << " xVel = [ " << rotVx.transpose() << " ];" << endl;
	outFile << endl << " zVel = [ " << rotVy.transpose() << " ];" << endl;
	outFile << endl << " phiVals = [ "  << endl << phiVals  << " ];" << endl;
	outFile << endl << " dphi_dvx = [ " << endl << dphi_dvx << " ];" << endl;
	outFile << endl << " dphi_dvz = [ " << endl << dphi_dvz << " ];" << endl;
	outFile.close();
	return 0;
	/**/
	
	cout << endl << "% initial conditions (" << initCnds.transpose() << ") ";

	fileName.str(""); fileName.clear();
	fileName << outDir << argv[1];
	analysis.setupDynamicSim( timestep, nSteps, false, fileName.str(), preview ); printf("\n%% timestep %.2lg, #steps %u", timestep, nSteps);
	double objValue;
	
	if( 1 ){
		LBFGSpp::LBFGSParam<double> optimOptions = defaultLBFGSoptions<LBFGSpp::LBFGSParam<double> >();
		optimOptions.init_step=0.01;
		LBFGSpp::LBFGSSolver<double> solver(optimOptions); //minimize with LBFGS
		if( 0 ){ // direct opt
			int r = solver.minimize(analysis, q, objValue);
			q = analysis.bestRunParameters;
		}else{ // penalty continuation
			int r = 0;
			for(double pfCont=1e2; pfCont<1e3; pfCont*=2.0){
				printf("\n%% cont. pf = %.2lf (iters done %d) ... \n", pfCont,r);
				optimOptions.delta=1e-5; optimOptions.epsilon=1e-8;
				fem.setPenaltyFactors(pfCont,pfCont);
				r += solver.minimize(analysis, q, objValue);
				q = analysis.bestRunParameters;
			}
			printf("\n%% cont. pf final opt ... \n");
			optimOptions.delta=1e-8; optimOptions.epsilon=1e-16;
			//if( fem.method==fem.CONTACT_CLAMP_PENALTY ) fem.setContactMethod( fem.CONTACT_HYBRID );
			fem.setPenaltyFactors(1e3,1e3);
			r += solver.minimize(analysis, q, objValue);
			q = analysis.bestRunParameters;
		}

	}else{// optimize with CMA from SCP
		cout << endl << "% CMA optimization (5000,16,0.05,1e-16,1e-8) bound +/-1e2" << endl;
		std::vector<double> fixedValues; std::vector<unsigned int> fixedIDs;
		if( icQ.setPostion ){
			fixedValues.push_back( q[icQ.getPositionIndex()  ]);
			fixedValues.push_back( q[icQ.getPositionIndex()+1]);
			fixedValues.push_back( q[icQ.getPositionIndex()+2]);
			fixedIDs.push_back( icQ.getPositionIndex()  );
			fixedIDs.push_back( icQ.getPositionIndex()+1);
			fixedIDs.push_back( icQ.getPositionIndex()+2);
		}
		if( icQ.setOrientation ){
			fixedValues.push_back( q[icQ.getOrientationIndex()  ]);
			fixedValues.push_back( q[icQ.getOrientationIndex()+1]);
			fixedValues.push_back( q[icQ.getOrientationIndex()+2]);
			fixedIDs.push_back( icQ.getOrientationIndex()  );
			fixedIDs.push_back( icQ.getOrientationIndex()+1);
			fixedIDs.push_back( icQ.getOrientationIndex()+2);
		}
		CMAObjectiveFunction cmaobj(analysis,fixedValues,fixedIDs);
		CMAMinimizer cmaopt(5000,16,0.05,1e-16,1e-8);
		bool ret;
		Eigen::VectorXd q_lower(q.size()), q_upper(q.size());
		q_lower.setConstant(-1e2); q_upper.setConstant(1e2);
		cmaopt.setBounds(q_lower,q_upper);
		ret = cmaopt.minimize(&cmaobj,q,objValue);
		cout << endl << "% CMA returned " << ret;
		q = analysis.bestRunParameters;
	}

	cout << endl << "best_q = [ " << analysis.bestRunParameters.transpose() << "]; ";
	cout << endl << "best_grad_phi = [ " << analysis.bestRunGradient.transpose() << "];" << endl << "best_phi = " << analysis.bestRunPhiValue << "; ";
	/**/

	if( preview != NULL ) delete preview;
	return 0;
}

int main_contact_obstaclesFromVTK(int argc, char* argv[]){

	printf("%% "); system("cd");
	printf("\n%% === main_contact_obstaclesFromVTK === \n\n");
	
	if( argc<=2 ) return -1; // the mesh file name and the data file directory as parameters
	Eigen::initParallel();
	std::stringstream fileName;
	std::string dataDir( argv[2] );
	std::string outDir="../_out/" + dataDir;
	ContactFEM fem;

#ifdef _WINDOWS
	// argv[1] contains the input mesh file name - we'll write outputs to outDir/inputMeshDir/inputMeshName... make sure inputMeshDir exists ...
	std::cout << "% output dir \"" << outDir << " ... ";
	std::string sc("@for %f in (\""); sc.append(outDir).append("\") do @mkdir %~dpf"); std::replace( sc.begin(), sc.end(), '/', '\\');
	int sr = system(sc.c_str());
	printf("\n\n");
#endif // _WINDOWS
	 outDir.append("mesh");

	fem.useBDF2=true;
	fem.MAX_SOLVER_ITERS = 99;
	fem.FORCE_BALANCE_EPS = 1e-10;
	fem.FORCE_BALANCE_DELTA = 1e-12;
	if( fem.method == ContactFEM::CONTACT_HYBRID ) fem.setSlipDirectionRegularizer(1e-8);
	fem.loadMeshFromElmerFiles(argv[1]);
	unsigned int bodyID = fem.bodyId.minCoeff();

	double timestep = 1.0/300.0, frictionCoeff = 0.4 /*0.364 ~~ tan(20°)*/;
	double viscNu = 0.1;//15.0; // above 15 more viscous damping causes artificial stiffness at this time step size and things actually bounce more
	double normalPenaltyFactor = 1e4, tangentialPenaltyFactor = 1e4;
	double lameLambda = 1e4, lameMu = 1e5, density = 58e-3 /fem.computeBodyVolume(bodyID); // weight 58g is for a tennis ball (according to Wikipedia)
	unsigned int nSteps = (unsigned int)((3.0+0.5*timestep)/timestep);
	unsigned int firstBounceTimeStep = 70; // how many time steps to simulate for initial condition fitting (before material params are optimized)
	if( argc>3 ){
		unsigned int t,c; c = sscanf(argv[3],"%u",&t);
		if( c==1 ) firstBounceTimeStep=t;
	}	printf("\n%% initial condition fit on first %u time steps.", firstBounceTimeStep);

	fem.setPenaltyFactors(normalPenaltyFactor, tangentialPenaltyFactor);
	fem.initializeMaterialModel(bodyID, LinearFEM::ISOTROPIC_NEOHOOKEAN, lameLambda, lameMu, density);
	vtkSmartPointer<RotationInvariantViscosityModel> viscMdl = vtkSmartPointer<RotationInvariantViscosityModel>::New();
	viscMdl->setTimestep((fem.useBDF2?2.0/3.0:1.0)*timestep);
	fem.initializeViscosityModel(bodyID, viscMdl,  &viscNu );

	PlaneFieldFromVTK floor, wall;
	floor.fitToPointCloud(dataDir + "floor.vtu");
	wall.fitToPointCloud( dataDir + "wall.vtu" );
	fem.addRigidObstacle(floor, frictionCoeff);
	fem.addRigidObstacle(wall , frictionCoeff);

	class GravityField : public VectorField{ public: Eigen::Vector3d g;
		GravityField(Eigen::Vector3d g_) : g(g_){}
		virtual void eval(Eigen::Vector3d& g_, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const { g_=g; }
	} g( -9.81*floor.getNormal() );
	fem.setExternalAcceleration(bodyID,g);
	fem.assembleMassAndExternalForce();

	InitialConditionParameterHandler icQ(bodyID); icQ.setPostion=true; icQ.setVelocity=true; icQ.setAngularVelocity=true;
	Eigen::VectorXd icvals; icvals.resize( icQ.getNumberOfParams(fem) ); icvals.setZero();
	//initial guess used for fist two-bounce-to-wall example
	//icvals.segment<3>(icQ.getPositionIndex()) = Eigen::Vector3d( -0.45, 0.17, 0.7 );
	//icvals.segment<3>(icQ.getVelocityIndex()) = -1.8*wall.getNormal() -1.6*floor.getNormal();
	icvals.segment<3>(icQ.getPositionIndex()) = Eigen::Vector3d( -0.5, 1.0, 0.5 );
	icvals.segment<3>(icQ.getVelocityIndex()) = 2.0*floor.getNormal();
	icQ.setNewParams(icvals, fem); icQ.applyInitialConditions(fem); fem.setResetPoseFromCurrent(); fem.setResetVelocitiesFromCurrent();

	/**/ // optimize to point cloud data
	icQ.positionGradients=icQ.setPostion; icQ.orientationGradients=icQ.setOrientation; icQ.velocityGradients=icQ.setVelocity; icQ.angularVelocityGradients=icQ.setAngularVelocity;
	TimeDependentPointCloudObjectiveFunction phi; double phiVal;
	phi.loadFromFileSeries(dataDir + "ball_", 1.0/30.0);
	// adjust final time to data range
	nSteps = (unsigned int)(( phi.timeOfFrame.rbegin()->second )/timestep);
	if( 0 ){ //optimize
		{
			fileName.str(""); fileName.clear();
			fileName << outDir << "_ic";

			DirectSensitivity analysis(phi, icQ, fem);
			analysis.setupDynamicSim( timestep, firstBounceTimeStep, false, fileName.str() );
			fem.doPrintResiduals=false;

			LBFGSpp::LBFGSParam<double> optimOptions = defaultLBFGSoptions<LBFGSpp::LBFGSParam<double> >();
			LBFGSpp::LBFGSSolver<double> solver(optimOptions); //minimize with LBFGS
			int r = solver.minimize(analysis, icvals, phiVal);

			cout << endl << "best_q = [ " << analysis.bestRunParameters.transpose() << "]; ";
			cout << endl << "best_grad_phi = [ " << analysis.bestRunGradient.transpose() << "]; ";
			cout << endl << "best_phi = " << analysis.bestRunPhiValue << "; ";

			icvals = analysis.bestRunParameters;
			fem.reset(); fem.assembleMassAndExternalForce();
			icQ.setNewParams(icvals, fem); icQ.applyInitialConditions(fem); fem.setResetPoseFromCurrent(); fem.setResetVelocitiesFromCurrent();
		}
		{
			fileName.str(""); fileName.clear();
			fileName << outDir << "_mat";

			GlobalElasticMaterialParameterHandler elastQ(bodyID);
			FrictionParameterHandler frictionQ(bodyID, (fem.useBDF2?2.0/3.0:1.0)*timestep);
			GlobalRotationInvariantViscosityMaterialParameterHandler viscQ( bodyID );
			CombinedParameterHandler elastFrictQ( elastQ, frictionQ );
			CombinedParameterHandler viscElastFrictQ( viscQ, elastFrictQ ); viscElastFrictQ.useLogOfParams=true;
			ParameterHandler& theQ=viscElastFrictQ;

			Eigen::VectorXd q; q.resize( theQ.getNumberOfParams(fem) );
			theQ.getCurrentParams(q,fem);

			DirectSensitivity analysis(phi, theQ, fem);
			analysis.setupDynamicSim( timestep, (250<nSteps)?250:nSteps, false, fileName.str() );

			LBFGSpp::LBFGSParam<double> optimOptions = defaultLBFGSoptions<LBFGSpp::LBFGSParam<double> >();
			LBFGSpp::LBFGSSolver<double> solver(optimOptions); //minimize with LBFGS
			int r = solver.minimize(analysis, q, phiVal);

			cout << endl << "best_q = [ " << analysis.bestRunParameters.transpose() << "]; ";
			cout << endl << "best_grad_phi = [ " << analysis.bestRunGradient.transpose() << "]; ";
			cout << endl << "best_phi = " << analysis.bestRunPhiValue << "; ";
			q = analysis.bestRunParameters;
			theQ.setNewParams(q, fem);
		}
		{
			fileName.str(""); fileName.clear();
			fileName << outDir << "_icmat";
			GlobalElasticMaterialParameterHandler elastQ(bodyID);
			FrictionParameterHandler frictionQ(bodyID, (fem.useBDF2?2.0/3.0:1.0)*timestep);
			GlobalRotationInvariantViscosityMaterialParameterHandler viscQ( bodyID );
			CombinedParameterHandler elastFrictQ( elastQ, frictionQ );
			CombinedParameterHandler viscElastFrictQ( viscQ, elastFrictQ ); viscElastFrictQ.useLogOfParams=true;
			CombinedParameterHandler theQ( icQ, viscElastFrictQ ); // Note: initial cond. handler must come first

			Eigen::VectorXd q; q.resize( theQ.getNumberOfParams(fem) );
			theQ.getCurrentParams(q,fem);

			DirectSensitivity analysis(phi, theQ, fem);
			analysis.setupDynamicSim( timestep, (500<nSteps)?500:nSteps, false, fileName.str() );

			LBFGSpp::LBFGSParam<double> optimOptions = defaultLBFGSoptions<LBFGSpp::LBFGSParam<double> >();
			LBFGSpp::LBFGSSolver<double> solver(optimOptions); //minimize with LBFGS

			int r = solver.minimize(analysis, q, phiVal);

			cout << endl << "best_q = [ " << analysis.bestRunParameters.transpose() << "]; ";
			cout << endl << "best_grad_phi = [ " << analysis.bestRunGradient.transpose() << "]; ";
			cout << endl << "best_phi = " << analysis.bestRunPhiValue << "; ";
			q = analysis.bestRunParameters;
			
			fem.reset(); fem.assembleMassAndExternalForce();
			theQ.setNewParams(q, fem); icQ.applyInitialConditions(fem); fem.setResetPoseFromCurrent(); fem.setResetVelocitiesFromCurrent();
		}
		{
			fileName.str(""); fileName.clear();
			fileName << outDir << "_matAllT";

			GlobalElasticMaterialParameterHandler elastQ(bodyID);
			FrictionParameterHandler frictionQ(bodyID, (fem.useBDF2?2.0/3.0:1.0)*timestep);
			GlobalRotationInvariantViscosityMaterialParameterHandler viscQ( bodyID );
			CombinedParameterHandler elastFrictQ( elastQ, frictionQ );
			CombinedParameterHandler viscElastFrictQ( viscQ, elastFrictQ ); viscElastFrictQ.useLogOfParams=true;
			ParameterHandler& theQ=viscElastFrictQ;

			Eigen::VectorXd q; q.resize( theQ.getNumberOfParams(fem) );
			theQ.getCurrentParams(q,fem);
			DirectSensitivity analysis(phi, theQ, fem);
			analysis.setupDynamicSim( timestep, nSteps, false, fileName.str() );

			LBFGSpp::LBFGSParam<double> optimOptions = defaultLBFGSoptions<LBFGSpp::LBFGSParam<double> >();
			LBFGSpp::LBFGSSolver<double> solver(optimOptions); //minimize with LBFGS
			int r = solver.minimize(analysis, q, phiVal);

			cout << endl << "best_q = [ " << analysis.bestRunParameters.transpose() << "]; ";
			cout << endl << "best_grad_phi = [ " << analysis.bestRunGradient.transpose() << "]; ";
			cout << endl << "best_phi = " << analysis.bestRunPhiValue << "; ";
			q = analysis.bestRunParameters;
			theQ.setNewParams(q, fem);
		}
		// // run forward sim
		fem.reset(); fem.assembleMassAndExternalForce();
		fem.doPrintResiduals=true;
		fileName.str(""); fileName.clear();
		fileName << outDir << "_" << std::setfill ('0') << std::setw(5) << (0);
		fem.saveMeshToVTUFile(fileName.str());

		for(int step=0; step<nSteps; ++step){
			fem.updateAllBoundaryData();
			fem.dynamicImplicitTimestep(timestep);
			printf(" %5d\n", step+1);

			fileName.str(""); fileName.clear();
			fileName << outDir << "_" << std::setfill ('0') << std::setw(5) << (step+1);
			fem.saveMeshToVTUFile(fileName.str());
		}
	}else{ //validation
		cout << endl << "% validation run (" << nSteps << " steps) ..." << endl;
		fileName.str(""); fileName.clear();
		fileName << outDir << "_validation";

		GlobalElasticMaterialParameterHandler elastQ(bodyID);
		FrictionParameterHandler frictionQ(bodyID, (fem.useBDF2?2.0/3.0:1.0)*timestep);
		GlobalRotationInvariantViscosityMaterialParameterHandler viscQ( bodyID );
		CombinedParameterHandler elastFrictQ( elastQ, frictionQ );
		CombinedParameterHandler viscElastFrictQ( viscQ, elastFrictQ ); viscElastFrictQ.useLogOfParams=true;
		CombinedParameterHandler theQ( icQ, viscElastFrictQ ); // Note: initial cond. handler must come first

		Eigen::VectorXd q, grad; q.resize( theQ.getNumberOfParams(fem) ); q.setZero();
		
		// YuMi tennis ball throw_1_2 icmat params
		//q <<    -0.0715487,     0.295047,     0.961764,      1.46727,     0.490885,     0.166672,  0.0, 0.0,   0.0,     -2.36325,      9.23724,      12.0231,    -0.963703,    -0.941048; // YuMi tennis ball throw_1_2 result: best_q = [   -0.0715487     0.295047     0.961764      1.46727     0.490885     0.166672  -0.00126503 -0.000470412   0.00162203     -2.36325      9.23724      12.0231    -0.963703    -0.941048]; 
		
		// YuMi tennis ball throw_1_1 ic initial conditions with throw_1_2 icmat material
		//q <<    -0.077379,     0.303773,     0.969505,      1.35727,     0.486499,     0.277709,  0.0, 0.0,   0.0,     -2.36325,      9.23724,      12.0231,    -0.963703,    -0.941048;

		// new throw (init cnd by hand) with 1_2 material
		// low and slow q <<   -0.1799 ,   0.1488 ,   0.9678 ,   0.9596 ,   0.4828 ,   0.1271 ,  0.0, 0.0,   0.0,     -2.36325,      9.23724,      12.0231,    -0.963703,    -0.941048;
		// low and fast q <<   -0.1799 ,   0.1488 ,   0.9678 ,   1.4198 ,   0.8341 ,   0.1958 ,  0.0, 0.0,   0.0,     -2.36325,      9.23724,      12.0231,    -0.963703,    -0.941048;
		// high and fast q <<    -0.1408 ,   0.4852 ,   0.7326  ,  1.9331  ,  0.7710   , 0.5735 ,  0.0, 0.0,   0.0,     -2.36325,      9.23724,      12.0231,    -0.963703,    -0.941048;

		// UR5 tennis ball throw2 final material params: -5.42788   9.43357   10.9539 -0.892405  -1.50169
		// UR5 tennis ball throw2 final initial conditions with YuMi throw_1_2 icmat material
		//q <<   0.240892,   0.0175126,    0.966319 ,   -2.42724 ,  -0.522853 ,   0.574877 ,  0.0, 0.0,   0.0,     -2.36325,      9.23724,      12.0231,    -0.963703,    -0.941048;

		// UR5 tennis ball throw2 initial conditions from transformed input with throw2 final material params
		//q << 0.5300 ,   0.0343,    0.9677,   -2.0615 ,   0.6225 ,   0.1511, 0.0, 0.0,   0.0, -5.42788 ,  9.43357 ,  10.9539, -0.892405,  -1.50169;
		//fileName << "_oriIC";
		// UR5 tennis ball throw1 initial conditions from transformed input with throw2 final material params
		//q << 0.5324 ,   0.1297 ,   0.9381  , -3.3884 ,   1.7319 ,   0.2434, 0.0, 0.0,   0.0, -5.42788 ,  9.43357 ,  10.9539, -0.892405,  -1.50169;
		//fileName << "_oriIC";


		// UR5 throw1 optimal material params:  -9.01878 ,  9.36237 ,  11.1109, 0.0589703 , -1.78977;
		// UR5 tennis ball throw3 optimal material params: -2.27939   9.26719   12.9943 -0.670595 -0.913773
		// UR5 tennis ball throw3 initial conditions from transformed input with throw3 optimal material params
		// UR5 tennis ball throw4 icmat material: -2.86982   ,   9.4475 ,    10.6594  ,  -1.49158 ,  -0.935922
		//q << 0.5276 ,  -0.0612 ,   0.9973,   -0.9464 ,   1.6924 ,  -0.3607, 0.0, 0.0,   0.0, -2.27939 ,  9.26719 ,  12.9943, -0.670595, -0.913773;
		// UR5 throw2 initial conditions from transformed input (new data repeat) with UR5 throw4 icmat==final material params
		//q << 0.5890,    0.0925 ,   1.0331 ,  -2.0709  ,  0.5482  , -0.2659 ,0.0, 0.0,   0.0, -2.86982   ,   9.4475 ,    10.6594  ,  -1.49158 ,  -0.935922;
		// UR5 throw2 initial conditions from transformed input (new data repeat) with UR5 throw1 final material params
		//q << 0.5890,    0.0925 ,   1.0331 ,  -2.0709  ,  0.5482  , -0.2659 ,0.0, 0.0,   0.0, -9.01878 ,  9.36237 ,  11.1109, 0.0589703 , -1.78977;
		// UR5 throw2 initial conditions from transformed input (new data repeat) with manual throw5070 final material params
		//q << 0.5890,    0.0925 ,   1.0331 ,  -2.0709  ,  0.5482  , -0.2659 ,0.0, 0.0,   0.0, -24.5911 , 6.90156,  12.1676,  8.47332, -1.15375;
		
		// for intro animation:
		q << 0.5890,    0.0925 ,   1.0331 ,  -2.0709  ,  0.5482  , -0.2659 ,0.0, 0.0,   0.0, -2   ,   9.4475 ,    10.3894  ,  -1.49158 ,  -0.935922;

		fileName << "_oriIC";

		DirectSensitivity analysis(phi, theQ, fem);
		analysis.setupDynamicSim( timestep, nSteps, false, fileName.str() );
		double phiVal = analysis(q, grad);
		cout << endl << "% validation run phi value = " << phiVal << endl;
	}	/**/


}


int main_contact_opt_CubeToPoint(int argc, char* argv[]){
	printf("\n%% === main_contact_opt_CubeToPoint === \n\n");
	std::stringstream fileName;
	std::string outDir="../_out/";

	ContactFEM fem;
	if( argc>2 ){
		if(       std::string(argv[2]).find("qp")!=std::string::npos ){ fem.method=fem.CONTACT_IGNORE;//fem.CONTACT_QP; outDir.append("qp_");
		}else if( std::string(argv[2]).find("tanh")!=std::string::npos ){ fem.method=fem.CONTACT_TANH_PENALTY; outDir.append("tanh_");
		}else if( std::string(argv[2]).find("clamp")!=std::string::npos ){ fem.method=fem.CONTACT_CLAMP_PENALTY; outDir.append("clamp_");
		}else if( std::string(argv[2]).find("hyb")!=std::string::npos ){ fem.method=fem.CONTACT_HYBRID; outDir.append("hybrid_");
		}else printf("\n%% unknown option %s - ignored \n", argv[2]);
	}

#ifdef _WINDOWS
	// argv[1] contains the input mesh file name - we'll write outputs to outDir/inputMeshDir/inputMeshName... make sure inputMeshDir exists ...
	std::cout << "% output dir \"" << outDir << argv[1] << " ... ";
	std::string sc("@for %f in (\""); sc.append(outDir).append(argv[1]).append("\") do @mkdir %~dpf"); std::replace( sc.begin(), sc.end(), '/', '\\');
	int sr = system(sc.c_str());
	printf("\n\n");
#endif // _WINDOWS

	fem.useBDF2=true;
	fem.MAX_SOLVER_ITERS = 99;
	fem.FORCE_BALANCE_EPS = 1e-8;
	fem.FORCE_BALANCE_DELTA = 1e-10;
	fem.setPenaltyFactors(1e6, 1e6);
	if( fem.method == ContactFEM::CONTACT_HYBRID ) fem.setSlipDirectionRegularizer(1e-8);
	fem.doPrintResiduals = true;

	bool enablePreview = false;
	double
		timestep = 0.001  ,//*0.5
		stoptime = 0.25,
		frictionCoeff = 0.2,//0.0,//
		viscosity = 0.0125, //0.1,
		lameLambda = 1e4,
		lameMu = 1e7,
		density = 3e4;
	unsigned int nSteps = (unsigned int)((stoptime+0.5*timestep)/timestep);

	fem.loadMeshFromElmerFiles(argv[1]);
	unsigned int bodyID = fem.bodyId.minCoeff();
	unsigned int bndryID = fem.bndId.minCoeff();
	MeshPreviewWindow* preview = NULL;
	if( enablePreview ) preview = new MeshPreviewWindow(fem.mesh.Get());
	fem.initializeMaterialModel(bodyID, LinearFEM::ISOTROPIC_NEOHOOKEAN, lameLambda, lameMu, density); printf("\n%% Neohookean material (la = %.4lg, mu = %.4lg, rho = %.4lg) ", lameLambda, lameMu, density);

	/** /
	vtkSmartPointer<RotationInvariantViscosityModel> viscMdl = vtkSmartPointer<RotationInvariantViscosityModel>::New();
	fem.initializeViscosityModel(bodyID, viscMdl,  &viscosity ); printf("\n%% Rotation invariant viscosity (nu = %.4lg) ", viscosity);
	/*/  //BDF1 instead of viscosity
	fem.useBDF2=false;
	/**/

	class GravityField : public VectorField{ public: virtual void eval(Eigen::Vector3d& g, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		g.setZero(); g[2]=-9.81;
		g = (Eigen::AngleAxisd(-0.0/180.0*M_PI, Eigen::Vector3d::UnitY())*g).eval();
	} } g;
	fem.setExternalAcceleration(bodyID,g);

	class FloorField : public DiffableScalarField{ public: virtual double eval(Eigen::Vector3d& n, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		double floorHeight = -0.05;
		n = (Eigen::AngleAxisd(0.0/180.0*M_PI, Eigen::Vector3d::UnitY())*Eigen::Vector3d::UnitZ()).eval().normalized(); // outward unit normal
		return n.dot(x)-floorHeight; // flat floor at (n'x)==floorHeight (positive outside, negative inside)
	} } floor;
	fem.addRigidObstacle(floor, frictionCoeff); printf("\n%% floor at -0.05, z-normal (cf = %.4lg) ", frictionCoeff);
	
	InitialConditionParameterHandler icQ( bodyID ); icQ.setVelocity=true; icQ.velocityGradients=true;
	Eigen::VectorXd initCnds( icQ.getNumberOfParams(fem) ); initCnds.setZero();
	printf("\n%% total mass %.6lg ", density*fem.computeBodyVolume() );
	cout << endl << "% initial conditions (" << initCnds.transpose() << ") ";
	icQ.setNewParams(initCnds,fem); icQ.applyInitialConditions(fem); fem.setResetPoseFromCurrent(); fem.setResetVelocitiesFromCurrent(); // slightly cumbersome way to set persistent initial conditions if the parameter handler is not used for optimization ...

	// optimize initial conditions
	ParameterHandler& theQ = icQ; Eigen::VectorXd& q = initCnds;

	AverageBoundaryValueObjectiveFunction phi;
	class TargetField : public VectorField{ public: virtual void eval(Eigen::Vector3d& g, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		g=x;
		if( t>=endtime ) g = Eigen::Vector3d( 0.1, 0.0, -0.04 );
	}	TargetField(double t_end){endtime=t_end;} double endtime;
	} finalTarget(stoptime-0.5*timestep);
	phi.addTargetField(bndryID, &finalTarget); printf("\n%% Target at (0.1, 0.0, -0.04) from t = %.4lf ", finalTarget.endtime );

	DirectSensitivity analysis(phi,theQ,fem); //AdjointSensitivity analysis(phi,theQ,fem, false );//expect non-symmetric matrices   //
	fileName.str(""); fileName.clear();
	fileName << outDir << argv[1];
	analysis.setupDynamicSim( timestep, nSteps, false, fileName.str(), preview );
	double objValue;

	LBFGSpp::LBFGSParam<double> optimOptions = defaultLBFGSoptions<LBFGSpp::LBFGSParam<double> >();
	optimOptions.init_step=0.1; optimOptions.m = 50;
	LBFGSpp::LBFGSSolver<double> solver(optimOptions); //minimize with LBFGS

	/** /
	int r = solver.minimize(analysis, q, objValue);
	q = analysis.bestRunParameters;
	cout << endl << "best_q = [ " << analysis.bestRunParameters.transpose() << "]; ";
	cout << endl << "best_grad_phi = [ " << analysis.bestRunGradient.transpose() << "]; best_phi = " << analysis.bestRunPhiValue << "; ";

	/*/{ // penalty continuation
		int r = 0;
		for(double pfCont=1e2; pfCont<1e6; pfCont*=2.0){
			printf("\n%% cont. pf = %.2lf (iters done %d) ... \n", pfCont,r);
			optimOptions.delta=1e-5; optimOptions.epsilon=1e-8;
			fem.setPenaltyFactors(pfCont,pfCont);
			r += solver.minimize(analysis, q, objValue);
			q = analysis.bestRunParameters;
		}
		printf("\n%% cont. pf final opt ... \n");
		optimOptions.delta=1e-8; optimOptions.epsilon=1e-16;
		//if( fem.method==fem.CONTACT_CLAMP_PENALTY ) fem.setContactMethod( fem.CONTACT_HYBRID );
		fem.setPenaltyFactors(1e6,1e6);
		r += solver.minimize(analysis, q, objValue);
		q = analysis.bestRunParameters;
	}/**/

	if( preview != NULL ) delete preview;
	return 0;
}

int main_contact(int argc, char* argv[]){
	//// for debugging: break on floating point math errors ...
	//_clearfp();
	//_controlfp(_controlfp(0, 0) & ~(_EM_INVALID | _EM_ZERODIVIDE | _EM_OVERFLOW | _EM_UNDERFLOW), _MCW_EM);


	if( std::string(argv[1]).find("SphereMarkers")!=std::string::npos )		return main_SphereMarkers_opt(argc,argv);
	if( std::string(argv[1]).find("CubeMarkers")!=std::string::npos )		return main_CubeMarkers_opt(argc,argv);
	if( std::string(argv[1]).find("Cube")!=std::string::npos )				return main_contact_opt_CubeToPoint(argc,argv);
	if( std::string(argv[1]).find("Sphere")!=std::string::npos )			return main_contact_opt_toPoint(argc,argv); //main_contact_opt_toPointWithWall(argc,argv); //main_contact_opt_toLine(argc,argv);// 
	if( std::string(argv[1]).find("Bunny")!=std::string::npos )				return main_contact_opt_Bunny(argc,argv);
	if( std::string(argv[1]).find("TennisBall")!=std::string::npos )		return main_contact_obstaclesFromVTK(argc,argv);
	if( std::string(argv[1]).find("BigBall")!=std::string::npos )			return main_contact_obstaclesFromVTK(argc,argv);

	printf("\n%% === main_contact === \n\n");
	
	if( argc<=1 ) return -1; // need at least the mesh file name as parameter
	Eigen::initParallel();
	std::stringstream fileName;
	std::string outDir="../_out/";
	ContactFEM fem;

	if( argc>2 ){
		if(       std::string(argv[2]).find("qp")!=std::string::npos ){ fem.method=fem.CONTACT_QP; outDir.append("qp_");
		}else if( std::string(argv[2]).find("tanh")!=std::string::npos ){ fem.method=fem.CONTACT_TANH_PENALTY; outDir.append("tanh_");
		}else if( std::string(argv[2]).find("clamp")!=std::string::npos ){ fem.method=fem.CONTACT_CLAMP_PENALTY; outDir.append("clamp_");
		}else if( std::string(argv[2]).find("class")!=std::string::npos ){ fem.method=92; outDir.append("class_");
		}else if( std::string(argv[2]).find("hyb")!=std::string::npos ){ fem.method=fem.CONTACT_HYBRID; outDir.append("hybrid_");
		}else printf("\n%% unknown option %s - ignored \n", argv[2]);
	}

#ifdef _WINDOWS
	// argv[1] contains the input mesh file name - we'll write outputs to outDir/inputMeshDir/inputMeshName... make sure inputMeshDir exists ...
	std::cout << "% output dir \"" << outDir << argv[1] << " ... ";
	std::string sc("@for %f in (\""); sc.append(outDir).append(argv[1]).append("\") do @mkdir %~dpf"); std::replace( sc.begin(), sc.end(), '/', '\\');
	int sr = system(sc.c_str());
	printf("\n\n");
#endif // _WINDOWS

	//fem.printConvergenceDebugInfo=true;
	//Paper results w/ convergence debug info:
	//- Cylinder, BDF1, invisc, ultrasoft (*), pen 1e2 -- SQP, HYB, TANH, CLAMP
	//- ToriBall, BDF2, visc(0.1), medium (**) -- HYB, TANH, CLAMP

	fem.useBDF2=true; //false; //
	fem.MAX_SOLVER_ITERS = 99;
	fem.FORCE_BALANCE_EPS = 1e-10;
	fem.FORCE_BALANCE_DELTA = 1e-12;
	if( fem.method == ContactFEM::CONTACT_HYBRID ) fem.setSlipDirectionRegularizer(1e-8);
	fem.loadMeshFromElmerFiles(argv[1]);
	unsigned int bodyID = fem.bodyId.minCoeff();

	double timestep = 0.005, frictionCoeff = 0.4 /*0.364 ~~ tan(20°)*/;
	double normalPenaltyFactor = 1e4, tangentialPenaltyFactor = 1e4;
	double lameLambda = 1e4, lameMu = 1e5, density = 150; // FlexFoam has roughly lambda = 1e4 Pa, mu = 1e5 Pa, rho = 150 kg/m^3

	normalPenaltyFactor = 1e3; tangentialPenaltyFactor = 1e3; lameLambda = 1e3; lameMu = 1e4; // (**) softer version ... (Dragon and ToriBall demo)
	//normalPenaltyFactor = 1e3; tangentialPenaltyFactor = 1e3; lameLambda = 1e2; lameMu = 1e3; // even softer version ... (Cylinder and ToriBall-medium demo)
	//normalPenaltyFactor = 1e3; tangentialPenaltyFactor = 1e3; lameLambda = 1e1; lameMu = 2e2; // (*) ultrasoft version ... (Cylinder-soft demo)
	//normalPenaltyFactor = 1e2; tangentialPenaltyFactor = 1e2;
	//normalPenaltyFactor = 1e5; tangentialPenaltyFactor = 1e5;
	
	fem.setPenaltyFactors(normalPenaltyFactor, tangentialPenaltyFactor);
	fem.initializeMaterialModel(bodyID, LinearFEM::ISOTROPIC_NEOHOOKEAN, lameLambda, lameMu, density);
	
	/**/
	double viscNu = 0.1;
	vtkSmartPointer<RotationInvariantViscosityModel> viscMdl = vtkSmartPointer<RotationInvariantViscosityModel>::New();
	viscMdl->setTimestep((fem.useBDF2?2.0/3.0:1.0)*timestep);
	fem.initializeViscosityModel(bodyID, viscMdl,  &viscNu ); /*/
	//double viscNuH[2] = {0.1, 2.0};
	//vtkSmartPointer<PowerLawViscosityModel> viscMdl = vtkSmartPointer<PowerLawViscosityModel>::New();
	//fem.initializeViscosityModel(bodyID, viscMdl, viscNuH ); /**/

	InitialConditionParameterHandler ic(bodyID); ic.setOrientation=true; //icQ.setPostion=true;
	Eigen::VectorXd icvals; icvals.resize( ic.getNumberOfParams(fem) );
	//icvals.segment<3>(icQ.getPositionIndex())    = Eigen::Vector3d( -0.1, 0.2, 0.3 );
	icvals.segment<3>(ic.getOrientationIndex()) = Eigen::Vector3d( 0.0, 0*M_PI_2, 0.0 );
	ic.setNewParams(icvals, fem); ic.applyInitialConditions(fem);

	class GravityField : public VectorField{ public: virtual void eval(Eigen::Vector3d& g, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		g.setZero(); g[2]=-9.81;
		g = (Eigen::AngleAxisd(-20.0/180.0*M_PI, Eigen::Vector3d::UnitY())*g).eval();
	} } g;
	fem.setExternalAcceleration(bodyID,g);

	class FloorField : public DiffableScalarField{ public: virtual double eval(Eigen::Vector3d& n, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		double floorHeight = -0.05;//0.0;//
		n = (Eigen::AngleAxisd(0.0/180.0*M_PI, Eigen::Vector3d::UnitY())*Eigen::Vector3d::UnitZ()).eval().normalized(); // outward unit normal
		return n.dot(x)-floorHeight; // flat floor at (n'x)==floorHeight (positive outside, negative inside)
	} } floor;
	fem.addRigidObstacle(floor, frictionCoeff);

	class WallField : public DiffableScalarField{ public: virtual double eval(Eigen::Vector3d& n, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		double wallX = 0.1;
		n = -Eigen::Vector3d::UnitX();
		return n.dot(x)+wallX;
	} } wall;
	if( std::string(argv[1]).find("Dragon")!=std::string::npos ){ fem.setSlipDirectionRegularizer(1e-6); } // more regularization for the Dragon (fine mesh, heavy object ...)
	else fem.addRigidObstacle(wall, 2.0*frictionCoeff); // add the wall for all other examples

	//class BoundaryField : public VectorField{ public: virtual void eval(Eigen::Vector3d& p, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
	//	p=x0; p.x() += 0.3*t; p.z()-=0.1*t; p.z() = std::max(-0.032,p.z());
	//} } bc;
	//fem.setBoundaryCondition( 1, bc );

	fem.assembleMassAndExternalForce(); printf("%% total mass is %.4lg ", fem.M.coeffs().sum()/fem.N_DOFS);

	vtkSmartPointer<vtkDoubleArray> vtkFc = vtkSmartPointer<vtkDoubleArray>::New();
	vtkFc->SetName(fieldNames[CONTACT_FORCE_NAME].c_str()); vtkFc->SetNumberOfComponents(3); vtkFc->SetNumberOfTuples(fem.getNumberOfNodes());
	fem.mesh->GetPointData()->AddArray(vtkFc);
	vtkSmartPointer<vtkIntArray> vtkFst = vtkSmartPointer<vtkIntArray>::New();
	vtkFst->SetName(fieldNames[CONTACT_STATE_NAME].c_str()); vtkFst->SetNumberOfComponents(1); vtkFst->SetNumberOfTuples(fem.getNumberOfNodes());
	for(unsigned int i=0; i<fem.getNumberOfNodes(); ++i){ vtkFc->SetTuple3(i,0.0,0.0,0.0); vtkFst->SetValue(i,0); }
	fem.mesh->GetPointData()->AddArray(vtkFst);

	fileName.str(""); fileName.clear();
	fileName << outDir << argv[1] /*<<"_visc"<<viscNu*/ << "_" << std::setfill ('0') << std::setw(5) << (0);
	fem.saveMeshToVTUFile(fileName.str());

	MeshPreviewWindow preview(fem.mesh.Get());

	QPSolver qpSolver;
	double qpEps = 1e-10 /*/ 1e-5 /**/; //MOSEK can't do much better than around 1e-5 ... not sure why

	/**
	fem.setContactMethod( ContactFEM::CONTACT_HYBRID );
	for(int i=0; i<5; ++i){ // run multiple times to check for memory leaks ...
		ObjectiveFunction empty; ParameterHandler nop(bodyID);
		AdjointSensitivity sa(empty, nop, fem, false);
		fileName.str(""); fileName.clear();
		if(i==0) fileName << outDir << argv[1];
		sa.setupDynamicSim(timestep,200,false, fileName.str());
		
		Eigen::VectorXd q(nop.getNumberOfParams(fem)), g(nop.getNumberOfParams(fem));
		nop.getCurrentParams(q,fem);
		sa(q,g);
	}

	/*/
	for(int step=0; step<500; ++step){
		printf("\n%% (%5d) ", step+1);
		fem.updateAllBoundaryData();

		switch( fem.method ){
		case fem.CONTACT_TANH_PENALTY:
			fem.dynamicImplicitTanhPenaltyContactTimestep(timestep);
			break;
		case fem.CONTACT_CLAMP_PENALTY:
			fem.dynamicImplicitClampedLinearPenaltyContactTimestep(timestep);
			break;
		case fem.CONTACT_HYBRID:
			fem.dynamicImplicitHybridPenaltyContactTimestep(timestep);
			break;
		case fem.CONTACT_QP:
			fem.dynamicImplicitQPContactTimestep(qpSolver, timestep, qpEps);
			break;
		case 92:
			fem.dynamicImplicitClassificationLinearPenaltyContactTimestep(timestep);
			break;
		}

		if( 0&& step>20 && step<25  ){ //FD-check stiffness
			printf("\n%% FD-check ...\n");
			VectXMap& x = fem.x; Eigen::VectorXd test, vi = fem.v, f_0; VectXMap& f = fem.f; SparseMatrixD& K = fem.K; SparseMatrixD& D = fem.D;
			Eigen::VectorXd x0 = x, fc(vi.size()); fc.setZero();
			if( fem.useBDF2 )
				x = x0 + 2.0/3.0*timestep*vi + 1.0/3.0*(x0-fem.x_old);
			else
				x = x0 + timestep*vi;
			fem.assembleForceAndStiffness(); // overwrites f vector (does setZero() first), also writes K = -df/dx
			fem.assembleViscousForceAndDamping( vi );
			//fem.assembleTanhPenaltyForceAndStiffness( vi, timestep ); //adds to fem.f and fem.K - call assembleForceAndStiffness first!
			fem.assembleClampedLinearPenaltyForceAndStiffness(fc, vi, fem.MAX_SOLVER_ITERS, timestep ); f+=fc;
			f_0 = f.eval();
			SparseMatrixD S = K*((fem.useBDF2?2.0/3.0:1.0)*timestep); if( D.size()>0 ) S+=D;

			double tmp, fdH=1e-8, maxErr=0.0; unsigned int outNode = 0/3;
			for(unsigned int i=0; i<fem.getNumberOfNodes(); ++i) for(unsigned int dof=0; dof<LinearFEM::N_DOFS; ++dof){
				unsigned int idof = fem.getNodalDof(i,dof);

				tmp = vi[idof];
				//vi[idof] -= 0.5*fdH;
				//if( fem.useBDF2 )
				//	x = x0 + 2.0/3.0*timestep*vi + 1.0/3.0*(x0-fem.x_old);
				//else
				//	x = x0 + timestep*vi;
				//fem.assembleForceAndStiffness( SKIP ); // overwrites f vector (does setZero() first), skip stiffness assembly
				//fem.assembleViscousForceAndDamping( vi, SKIP );
				////fem.assembleTanhPenaltyForceAndStiffness( vi, timestep, SKIP); //adds to fem.f and fem.K - keep order of assembly fcn. calls!
				//fem.assembleClampedLinearPenaltyForceAndStiffness(fc, vi, fem.MAX_SOLVER_ITERS, timestep, SKIP ); f+=fc;
				//f_0 = f.eval();

				vi[idof] = tmp + fdH; // *0.5
				if( fem.useBDF2 )
					x = x0 + 2.0/3.0*timestep*vi + 1.0/3.0*(x0-fem.x_old);
				else
					x = x0 + timestep*vi;
				fem.assembleForceAndStiffness( SKIP ); // overwrites f vector (does setZero() first), skip stiffness assembly
				fem.assembleViscousForceAndDamping( vi, SKIP );
				//fem.assembleTanhPenaltyForceAndStiffness( vi, timestep, SKIP); //adds to fem.f and fem.K - keep order of assembly fcn. calls!
				fem.assembleClampedLinearPenaltyForceAndStiffness(fc, vi, fem.MAX_SOLVER_ITERS, timestep, SKIP ); f+=fc;

				test=(S.col(idof) - (f_0-f)/fdH);
				maxErr = std::max( maxErr, test.cwiseAbs().maxCoeff() );

				vi[idof] = tmp;

				if( (idof/3)==outNode ) cout << (f_0-f).segment<3>(3*outNode)/fdH << endl;
			}
			x = x0;
			printf("\n%% FD-check S max coeff err %8.3lg (rel %8.3lg) (max S %8.3lg) \n", maxErr, maxErr/((S.coeffs().cwiseAbs().maxCoeff())),((S.coeffs().cwiseAbs().maxCoeff())));
			cout << S.block(3*outNode,3*outNode,3,3) << endl; // << "  .. v-norm " << vi.block(0,0,3,1).norm()
		}

		if( 0){ //debug output viscous forces
			Eigen::VectorXd f = fem.f, vi = fem.v; // store

			fem.f.setZero();

			fem.assembleViscousForceAndDamping( vi, MyFEM::SKIP );
			//printf("\n%% ||f_v|| = %.4lg, ||v|| = %.4lg", fem.f.norm() , vi.norm() );

			fileName.str(""); fileName.clear();
			fileName << outDir << argv[1] << "_" << std::setfill ('0') << std::setw(5) << (step+1);
			fem.saveMeshToVTUFile(fileName.str());

			fem.f = f; // restore
		}else{
			fileName.str(""); fileName.clear();
			fileName << outDir << argv[1] << "_" << std::setfill ('0') << std::setw(5) << (step+1);
			//fileName << outDir << argv[1] <<"_visc"<<viscNu << "_" << std::setfill ('0') << std::setw(5) << (step+1);
			fem.saveMeshToVTUFile(fileName.str());
		}

		// update preview
		//preview.render();

	}	printf("\n");
	/**/

	return 0;
}

