
#include "ElmerReader.h"
#include "LinearFEM.h"
#include "Materials.h"
#include "Remesher.h"
#include "AdjointSensitivity.h"
#include "DirectSensitivity.h"
#include "TemporalInterpolationField.h"
#include "BoundaryFieldObjectiveFunction.h"
#include "ElasticMaterialParameterHandler.h"
#include "ViscousMaterialParameterHandler.h"
#include "EigenmodeViscosityModel.h"
#include "DifferentiableSpline.h"
#include "../LBFGSpp/LBFGS.h"
#include "../LBFGSpp/Newton.h"
#include "../LBFGSpp/PLBFGS.h"
//Note: we could also try https://github.com/PatWie/CppNumericalSolvers -- that one also has an L-BFGS-B solver with box constraints l <= x <= u
//#include "../cppoptlib/solver/lbfgssolver.h" // does not seem to work properly - even with some extra tweaks
using namespace MyFEM;

#include <vtkUnstructuredGrid.h>
#include <sstream>
#include <fstream>
#include <omp.h> // for timing ...


#include "../Spectra/SymGEigsSolver.h" // just for testing ...

int main_Xworm(int argc, char* argv[]); // in main_Xworm.cpp
int main_CylinderMarkers(int argc, char* argv[]); // in main_CylinderMarkers.cpp
int main_TPUbar(int argc, char* argv[]); // in main_TPU_lattices.cpp
int main_SiliconeHand(int argc, char* argv[]); // in main_TPU_lattices.cpp
int main_PointCloudDemo(int argc, char* argv[]); // in main_PointCloudDemo.cpp
int main_YuMiFoamBar(int argc, char* argv[]); // in main_YuMiData.cpp
int main_contact(int argc, char* argv[]); // in main_contact.cpp

int main_Box_wMarkers(int argc, char* argv[]){
	bool enablePreview=false;
	bool startWithStaticSolve=true;
	std::string bndFileName, outDir="../_out/";
	std::stringstream fileName;

	printf("\n\n%% -- Box_wMarkers -- \n\n");
	
	// pinned boundary
	class BoundaryFieldFixed : public VectorField{ public: virtual void eval(Eigen::Vector3d& p, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		p=x0;
	} } fixedWallBC;

	// actuated boundary
	class BoundaryFieldMoving : public VectorField{ public: virtual void eval(Eigen::Vector3d& p, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		if( t<0.0   ) t = 0.0;
		if( t>0.125 ) t = 0.125;
		p = x0
			+ Eigen::Vector3d::UnitY()*(cos(13.0*M_PI*t)-1.0)*0.05
			+ Eigen::Vector3d::UnitZ()*(cos(21.0*M_PI*t)-1.0)*0.02
			+ Eigen::Vector3d::UnitX()*(cos(47.0*M_PI*t)-1.0)*0.03;
	} } movingWallBC;

	// gravity (-z)
	class GravityField : public VectorField{ public: virtual void eval(Eigen::Vector3d& g, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		g = -9.81 *Eigen::Vector3d::UnitZ();
	} } g;

	const unsigned int
		bodyID=19, // the body ID used in the mesh files for all elements
		fixedWallID1=17, // the boundary ID used for surfaces glued to the fixed wall
		fixedWallID2=17, // the boundary ID used for surfaces glued to the fixed wall
		movingWallID1=18, // the boundary ID used for surfaces glued to the moving wall
		movingWallID2=18; // the boundary ID used for surfaces glued to the moving wall

	const double dt=1.0/600.0, t_end=0.25; const unsigned int tsteps = (t_end+0.5*dt)/dt, substeps=1;

	LinearFEM theFEM;
	theFEM.loadMeshFromElmerFiles(argv[1]);
	
	//theFEM.doPrintResiduals=false;
	theFEM.useBDF2=true; printf("\n%% Using BDF2 time integration ");  // BDF2 works with both direct and adjoint sensitivity analysis - however accuracy of adjoint method is sometimes not great

	//Remesher reme; reme.targetEdgeLength = 0.02;
	//reme.remesh(theFEM);

	//theFEM.setBoundaryCondition(fixedWallID1, fixedWallBC);
	//theFEM.setBoundaryCondition(fixedWallID2, fixedWallBC);
	theFEM.setBoundaryCondition(movingWallID1, movingWallBC);
	theFEM.setBoundaryCondition(movingWallID2, movingWallBC);

	theFEM.setExternalAcceleration(bodyID, g);
	vtkSmartPointer<PowerLawViscosityModel> viscMdl = vtkSmartPointer<PowerLawViscosityModel>::New(); Eigen::VectorXd viscParam( viscMdl->getNumberOfParameters() ); viscParam.setZero();

	unsigned int materialChoice=3;
	switch( materialChoice ){
	case 2:
		theFEM.initializeMaterialModel(bodyID, LinearFEM::ISOTROPIC_NEOHOOKEAN,1e3,80e3,80);
		viscMdl->setViscosity( viscParam.data(), 120.0 ); viscMdl->setPowerLawH( viscParam.data(), 0.5 );
		printf("\n%% Material 2 Neohookean (1e3,80e3,80) viscosity (120,0.5) ");
		break;
	case 3:
		theFEM.initializeMaterialModel(bodyID, LinearFEM::ISOTROPIC_NEOHOOKEAN, 1e3,80e3,80);
		viscMdl->setViscosity( viscParam.data(), 60.0 ); viscMdl->setPowerLawH( viscParam.data(), 2.0 );
		printf("\n%% Material 3 Neohookean (1e3,80e3,80) viscosity (60,2.0) ");
		break;
	case 4:
		theFEM.initializeMaterialModel(bodyID, LinearFEM::ISOTROPIC_NEOHOOKEAN,20e3,40e3,80);
		viscMdl->setViscosity( viscParam.data(), 60.0 ); viscMdl->setPowerLawH( viscParam.data(), 2.0 );
		printf("\n%% Material 4 Neohookean (20e3,40e3,80) viscosity (60,2.0) ");
		break;
	case 5:
		theFEM.initializeMaterialModel(bodyID, LinearFEM::ISOTROPIC_NEOHOOKEAN,5e3,10e3,80);
		viscMdl->setViscosity( viscParam.data(), 10.0 ); viscMdl->setPowerLawH( viscParam.data(), 1.0 );
		printf("\n%% Material 5 Neohookean (5e3,10e3,80) viscosity (10,1.0) ");
		break;
	case 1:
	default:
		//theFEM.initializeMaterialModel(bodyID, LinearFEM::ISOTROPIC_LINEAR_ELASTIC,20e3,40e3,80);
		theFEM.initializeMaterialModel(bodyID, LinearFEM::ISOTROPIC_NEOHOOKEAN,20e3,40e3,80);
		viscMdl->setViscosity( viscParam.data(), 120.0 ); viscMdl->setPowerLawH( viscParam.data(), 0.5 );
		printf("\n%% Material 1 Neohookean (20e3,40e3,80) viscosity (120,0.5) ");
		break;
	}

	theFEM.initializeViscosityModel(bodyID, viscMdl, viscParam.data() );
	
	MeshPreviewWindow thePreview(theFEM.mesh.Get());

	// if we're given targets in a file, try to match them via parameter estimation ...
	if( argc>2 ){ // build objective function from given displacement data file ...
		printf("\n%% Target file \"%s\"", argv[2]);
		
		// for each tracked boundary (IDs in first line of given file)
		// ... create a TemporalInterpolationField object
		// ... store the points for each time (first value in line) of all boundaries (triplets of values in line)
		// ... add the interpolation fields to the objective function object

		bool velocityMode=false;
		AverageBoundaryValueObjectiveFunction thePhi;
		if(velocityMode) thePhi.targetMode = AverageBoundaryValueObjectiveFunction::TARGET_VELOCITY; printf("\n%% Objective function measures tracked velocities ");
		std::vector<vtkSmartPointer<TemporalInterpolationField> > trackedTargets; // storage for target field used by thePhi (does not take ownership)
		std::vector<unsigned int> trackedIDs;
		if( TemporalInterpolationField::buildFieldsFromTextFile( argv[2], trackedTargets, trackedIDs) < 0 ) return -1;
		for(int i=0; i<trackedIDs.size(); ++i){
			thePhi.addTargetField( trackedIDs[i] , trackedTargets[i] );
			/** / // here we assume that the input file contains velocity data already
		}	/*/
			// compute velocities from tracked locations ...
			if(velocityMode){
				trackedTargets[i]->evalMode = TemporalInterpolationField::EVAL_VELOCITY;
				trackedTargets[i]->setInitialVelocity(Eigen::Vector3d::Zero()); // seems not to make much of a difference ...
			}	printf("\n%% computing target velocities from input locations ");
		}
		/**/
		fileName.str(""); fileName.clear();
		fileName << outDir << argv[1] << "_targetBCs";
		TemporalInterpolationField::writeFieldsToVTU(fileName.str(), 2*tsteps, trackedTargets, trackedIDs);

		if( 0 /*testing: output upsampled input data ...*/){
			fileName << argv[2] << "_upsampled.log";
			ofstream tbOut(fileName.str().c_str()); tbOut << std::setprecision(16);
			fileName.str(""); fileName.clear();
			fileName << argv[2] << "_up_velocity.log"; // also write velocities
			ofstream tbOutV(fileName.str().c_str()); tbOutV << std::setprecision(16);
			for(int i=0; i<trackedIDs.size(); ++i){
				tbOut << trackedIDs[i] << " ";
			}	tbOut << endl;
			double r[2]; trackedTargets.begin()->Get()->getRange(r);
			for(double t=r[0]; t<r[1]; t+=(r[1]-r[0])/500.0){
				tbOut << t << " "; tbOutV << t << " ";
				for(int i=0; i<trackedIDs.size(); ++i){
					Eigen::Vector3d u,x,x0;
					trackedTargets[i]->eval(u,x,x0,t);
					tbOut << u[0] << " " << u[1] << " " << u[2] << " ";
					// also write velocities
					trackedTargets[i]->evalVelocity(u,x,x0,t);
					tbOutV << u[0] << " " << u[1] << " " << u[2] << " ";
				}	tbOut << endl; tbOutV << endl;
				Eigen::VectorXd dx0dy;
				trackedTargets.begin()->Get()->xSpline->EvaluateYDerivative(t,dx0dy);
				cout << endl << dx0dy.sum();
			}
			tbOut.close(); tbOutV.close();
			
			TemporalInterpolationField::writeFieldsToVTU( std::string(argv[2]), 4*tsteps+1, trackedTargets, trackedIDs );

			return 0;
		}

		double phi = 0.0; int r = -1, ev, rInit = -1, evInit;

		printf("\n%% Optimizing for combined global visco-elastic parameters. "); //printf("\n%% Optimizing for combined global viscous and local elastic parameters. "); //
		GlobalElasticMaterialParameterHandler     elasticParamHandler(bodyID); //PerElementElasticMaterialParameterHandler elasticParamHandler; //
		GlobalPLViscosityMaterialParameterHandler viscousParamHandler(bodyID);
		GlobalDensityParameterHandler			  densityParamHandler(bodyID);
		CombinedParameterHandler combinedParamHandler(elasticParamHandler,densityParamHandler); //(viscousParamHandler, elasticParamHandler);
		//combinedParamHandler.useLogOfParams = true; printf("\n%% Optimizing for log of parameters ");
		ParameterHandler& theQ ( combinedParamHandler ); // choose which one we use ...

		LBFGSpp::LBFGSParam<double> optimOptions;
		optimOptions.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_WOLFE; //_STRONG
		optimOptions.m = 10;
		optimOptions.epsilon = 1e-20;
		optimOptions.past = 1; // 0 == off, 1 == compare to previous iteration to detect if we got stuck ...
		optimOptions.delta = 1e-12; // relative change of objective function to previous iteration below which we give up
		optimOptions.max_iterations = 150;

		if( 0 /*elastostatic optimization first*/){
			GlobalElasticMaterialParameterHandler theQ(bodyID);
			Eigen::VectorXd q( theQ.getNumberOfParams(theFEM) );
			theQ.getCurrentParams(q, theFEM);
			/**/
			DirectSensitivity theSensitivity(thePhi, theQ, theFEM);
			LBFGSpp::NewtonSolver<double,Eigen::ColPivHouseholderQR<Eigen::MatrixXd> > solver(optimOptions);
			printf("\n%% Optimizing initial elastostatic configuration with Gauss-Newton (direct sensitivity analysis) ");
			/*/
			AdjointSensitivity theSensitivity(thePhi, theQ, theFEM);
			LBFGSpp::LBFGSSolver<double> solver(optimOptions);
			printf("\n%% Optimizing initial elastostatic configuration with LBFGS (adjoint sensitivity analysis) ");
			/**/
			 rInit = solver.minimize(theSensitivity,q,phi);
			evInit = theSensitivity.getEvalCounter();
			theQ.setNewParams( theSensitivity.bestRunParameters, theFEM );
			cout << endl << "% initial elastostatic solver iterations " << rInit << ", fcn.evals " << evInit << endl;
		}

		if( 0 /*viscosity-only optimization*/){
			ParameterHandler& theQ ( viscousParamHandler );
			Eigen::VectorXd q( theQ.getNumberOfParams(theFEM) );
			theQ.getCurrentParams(q, theFEM);
			DirectSensitivity theSensitivity(thePhi, theQ, theFEM);
			theSensitivity.setupDynamicSim(dt,tsteps, startWithStaticSolve ,""/*skip writing output files*/, (enablePreview?(&thePreview):NULL) );
			int tmp = optimOptions.linesearch;
			optimOptions.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_ARMIJO;
			//optimOptions.precond.resize(1); optimOptions.precond(0)=0.9;
			LBFGSpp::NewtonSolver<double,Eigen::ColPivHouseholderQR<Eigen::MatrixXd> > solver(optimOptions); //minimize with Gauss-Newton
			printf("\n%% Optimizing viscosity with Gauss-Newton (direct sensitivity analysis) ");
			rInit += solver.minimize(theSensitivity,q,phi);
			evInit += theSensitivity.getEvalCounter();
			cout << endl << "% initial elastostatic and viscosity solver iterations " << rInit << ", fcn.evals " << evInit << endl;
			optimOptions.linesearch = tmp;
			//optimOptions.precond.resize(0);
		}

		Eigen::VectorXd dphi, q( theQ.getNumberOfParams(theFEM) );
		theQ.getCurrentParams(q, theFEM);

		if( 1 /*finite-difference check dPhi/dq*/){ 
			//DirectSensitivity theSensitivity(thePhi, theQ, theFEM); printf("\n%% Direct sensitivity analysis finite-difference check ... ");
			AdjointSensitivity theSensitivity(thePhi, theQ, theFEM); printf("\n%% Adjoint sensitivity analysis finite-difference check ... ");
			theSensitivity.setupDynamicSim(dt,tsteps, startWithStaticSolve ,""/*skip writing output files*/, (enablePreview?(&thePreview):NULL) );
			double fdH=1e-5, tmp;
			Eigen::VectorXd phi_fd( q.size() ), dphi_dq;
			for(int k=0; k<q.size(); ++k){
				printf("\n%% FD-check sim-run %d/%d ... ", k+1, q.size()+1);
				tmp = q[k];
				q[k]+= fdH;
				phi_fd[k] = theSensitivity(q,dphi_dq);
				q[k] = tmp;
			}
			printf("\n%% FD-check final sim-run ... ");
			double phi_0 = theSensitivity(q,dphi_dq);
			phi_fd.array() -= phi_0; phi_fd /= fdH;
			cout << endl << "dphi_fd = [ " <<  phi_fd.transpose() << " ]";
			cout << endl << "dphi_dq = [ " << dphi_dq.transpose() << " ]";
			//return 0;
		}

		if( 0 /*minimize with LBFGS*/){
			printf("\n%% Optimizing with LBFGS (adjoint sensitivity analysis) ");
			AdjointSensitivity theSensitivity(thePhi, theQ, theFEM);
			theSensitivity.setupDynamicSim(dt,tsteps, startWithStaticSolve ,""/*skip writing output files*/, (enablePreview?(&thePreview):NULL) );
			//if( 0 && (&theQ == &combinedParamHandler) ){
			//	optimOptions.precond.resize( q.size() );
			//	for(int k=0; k<q.size(); ++k) optimOptions.precond(k) = (k==0)? 0.1 :((k==1)? 0.01 :((q.size()>5)? 1e6 : 1000.0 ));
			//	cout << endl << "% User-defined preconditioner = diag(" << optimOptions.precond.block(0,0,optimOptions.precond.size()>5?5:optimOptions.precond.size(),1).transpose() << ") "; if( optimOptions.precond.size()>5 ) printf("...");
			//}
			optimOptions.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_WOLFE; //_STRONG
			LBFGSpp::LBFGSSolver<double> solver(optimOptions);
			theSensitivity.resetEvalCounter();

			r = solver.minimize(theSensitivity, q, phi);

			ev   = theSensitivity.getEvalCounter();
			phi  = theSensitivity.bestRunPhiValue;  // sometimes the solver messes up at the end, so load the parameters from the best sim run we've had
			q    = theSensitivity.bestRunParameters;
			dphi = theSensitivity.bestRunGradient;
		}else
		{	
			DirectSensitivity theSensitivity(thePhi, theQ, theFEM);
			theSensitivity.setupDynamicSim(dt,tsteps, startWithStaticSolve ,""/*skip writing output files*/, (enablePreview?(&thePreview):NULL) );

			/**/
			optimOptions.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_ARMIJO;
			LBFGSpp::NewtonSolver<double,Eigen::ColPivHouseholderQR<Eigen::MatrixXd> > solver(optimOptions); //minimize with Gauss-Newton
			printf("\n%% Optimizing with Gauss-Newton (direct sensitivity analysis) ");
			/*/ //PLBFGSSolver seems to not quite work as intended -- preconditioning wrong?
			LBFGSpp::PLBFGSSolver<double,Eigen::ColPivHouseholderQR<Eigen::MatrixXd> > solver(optimOptions); //minimize with H-preconditioned LBFGS
			printf("\n%% Optimizing with H-preconditioned LBFGS (direct sensitivity analysis) ");
			/**/
			r = solver.minimize(theSensitivity, q, phi);

			ev   = theSensitivity.getEvalCounter();
			phi  = theSensitivity.bestRunPhiValue;
			q    = theSensitivity.bestRunParameters;
			dphi = theSensitivity.bestRunGradient;
		}
		cout << endl << "% solver iterations " << r << ", fcn.evals " << ev;
		fileName.str(""); fileName.clear();
		fileName << outDir << argv[1] << "_optimResult.txt";
		ofstream optOut(fileName.str());
		if( rInit>0 ) optOut << "% initial elastostatic solver iterations " << rInit << ", fcn.evals " << evInit << endl;
		optOut << "% solver iterations " << r << ", fcn.evals " << ev << ", objective function value " << phi << endl;
		optOut << endl << "% params:" << endl << q << endl;
		optOut << endl << "% gradient:" << endl << dphi << endl;
		optOut.close();



		//return 0;
		theQ.setNewParams(q,theFEM); theFEM.doPrintResiduals=false;
		theFEM.reset();
	}
	
	if( 1 /*forward sim and output ...*/){
		printf("\n%% Forward sim - writing output ");
		id_set trackedBoundaries; // record average displacements for these boundary surface IDs
		for(unsigned int i=1; i<=11; ++i) trackedBoundaries.insert(i);
		trackedBoundaries.insert(14); trackedBoundaries.insert(15); trackedBoundaries.insert(16);
		fileName.str(""); fileName.clear();
		fileName << outDir << argv[1] << "_trackedBCs.log";
		ofstream tbOut(fileName.str().c_str()); tbOut << std::setprecision(18);
		for(id_set::iterator tb=trackedBoundaries.begin(); tb!=trackedBoundaries.end(); ++tb){
			tbOut << *tb << " ";
		}	tbOut << endl;


		if( startWithStaticSolve ){
			printf("\n%% init ");
			theFEM.assembleMassAndExternalForce();
			theFEM.updateAllBoundaryData();
			theFEM.staticSolve(); // in the end we want to do this (because in experiments we'll never get rid of gravity, but we can start from zero velocity)
		}

		// output tracked boundaries (avg. displacement)
		Eigen::Vector3d uB;
		tbOut << theFEM.simTime << " "; // should be 0.0 here
		for(id_set::iterator tb=trackedBoundaries.begin(); tb!=trackedBoundaries.end(); ++tb){
			theFEM.computeAverageDeformedCoordinateOfBoundary(*tb,uB);
			//theFEM.computeAverageVelocityOfBoundary(*tb,uB);
			tbOut << uB[0] << " " << uB[1] << " " << uB[2] << " ";
		}	tbOut << endl;

		for(int step=0; step<tsteps; ++step){
			// write output file before step
			fileName.str(""); fileName.clear();
			fileName << outDir << argv[1] << "_" << std::setfill ('0') << std::setw(5) << step;
			theFEM.saveMeshToVTUFile(fileName.str());

			printf("\n%% step %5d/%d ", step+1,tsteps);
			for(int substep=0; substep<substeps; ++substep){
				if( theFEM.doPrintResiduals && substeps>1 ) printf("\n%%");
				theFEM.updateAllBoundaryData();
				theFEM.dynamicImplicitTimestep(dt/(double)substeps);
			}

			// update preview
			if(enablePreview) thePreview.render();
		
			// output tracked boundaries (avg. displacement)
			tbOut << theFEM.simTime << " ";
			for(id_set::iterator tb=trackedBoundaries.begin(); tb!=trackedBoundaries.end(); ++tb){
				theFEM.computeAverageDeformedCoordinateOfBoundary(*tb,uB);
				//theFEM.computeAverageVelocityOfBoundary(*tb,uB);
				tbOut << uB[0] << " " << uB[1] << " " << uB[2] << " ";
			}	tbOut << endl;

		}

		tbOut.close();

		// write last output file
		fileName.str(""); fileName.clear();
		fileName << outDir << argv[1] << "_" << std::setfill ('0') << std::setw(5) << tsteps;
		theFEM.saveMeshToVTUFile(fileName.str(), true);

		return 0;
	}
}

// version for regular contraction with flat lower surface -- works quite well, don't change too much

int main_Cylinder_5x2cm(int argc, char* argv[]){
	bool enablePreview=false;

	double dt=0.004, t_end=0.2; unsigned int tsteps = (t_end+0.5*dt)/dt;


	std::string bndFileName, outDir="../_out/";
	std::stringstream fileName;

	LinearFEM theFEM;
	theFEM.useBDF2=true; // we now have full support for BDF2 in both direct and adjoint sensitivity analysis
	theFEM.doPrintResiduals=true;
	theFEM.FORCE_BALANCE_EPS = 1e-6;
	theFEM.loadMeshFromElmerFiles(argv[1]);

	Remesher remesh; remesh.setTargetEdgeLengthFromMesh(theFEM.mesh); remesh.targetEdgeLength*=0.5;
	remesh.remesh(theFEM); printf("\n%% remeshed to %d elems", theFEM.getNumberOfElems());

	MeshPreviewWindow thePreview(theFEM.mesh.Get());
	const unsigned int bodyId = 3;


	// boundary conditions
	class BoundaryField : public VectorField{ public: virtual void eval(Eigen::Vector3d& p, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		p=x0; // simple clamp
	} } bc;
	theFEM.setBoundaryCondition(1, bc);

	// gravity
	class GravityField : public VectorField{ public: virtual void eval(Eigen::Vector3d& g, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		g.setZero(); g[2]=-9.81;
	} } g;
	theFEM.setExternalAcceleration(bodyId, g);

	// elastic material model
	theFEM.initializeMaterialModel(bodyId, LinearFEM::ISOTROPIC_NEOHOOKEAN,20,40,100);

	// damping model
	vtkSmartPointer<PowerSeriesViscosityModel> viscMdl = vtkSmartPointer<PowerSeriesViscosityModel>::New();
	viscMdl->powerIndices.resize(1); viscMdl->powerIndices(0)=1.0;
	Eigen::VectorXd coeffs( viscMdl->getNumberOfParameters() ); coeffs(0)=2.0;
	theFEM.initializeViscosityModel(bodyId, viscMdl,  coeffs.data() );
	cout << endl << "% power series viscous damping: flow indices are " << viscMdl->powerIndices.transpose() << ", coeffs are " << coeffs.transpose();

	// objective function
	class TargetDisplacementField : public VectorField{ public:
		virtual void eval(Eigen::Vector3d& p, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
			p = x-x0; // don't care for early times ... set target to current displacement
			if( t > (t_end - 0.5*dt) ) {
				p = -0.2*x0; // x* = x0 + p
				p[2]=-0.02;
			}
		}	
		double dt, t_end;
		TargetDisplacementField(double dt_, double t_end_) : dt(dt_), t_end(t_end_){}
	} uStar(dt,t_end);
	BoundaryFieldObjectiveFunction thePhi;
	thePhi.addTargetField(2,uStar);

	//// for debug output
	//thePhi.debugFileName=outDir; thePhi.debugFileName.append(argv[1]); thePhi.debugFileName.append("_objFcnDebug_");

	// solver configuration
	LBFGSpp::LBFGSParam<double> optimOptions;
	optimOptions.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_WOLFE;
	optimOptions.m = 5;
	optimOptions.epsilon = 1e-20;
	optimOptions.past = 1; // compare to previous iteration to detect if we got stuck ...
	optimOptions.delta = 5e-5; //1e-12;// relative change of objective function to previous iteration below which we give up
	optimOptions.max_iterations = 150;
	optimOptions.wolfe = 1.0-1e-3;
	LBFGSpp::LBFGSSolver<double> lbfgs(optimOptions); // don't change optimOptions later, it's stored by ref!
	
	LBFGSpp::LBFGSParam<double> optimOptions2(optimOptions);											
	optimOptions2.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_ARMIJO;
	LBFGSpp::NewtonSolver<double,Eigen::ColPivHouseholderQR<Eigen::MatrixXd> > newton(optimOptions);

	// parameter vector ...
	Eigen::VectorXd q;

	if( 1 /*global parameter estimation*/){
		// parameter handler
		GlobalElasticMaterialParameterHandler elQ(bodyId); GlobalPSViscosityMaterialParameterHandler visQ(bodyId); CombinedParameterHandler theQ(visQ,elQ);
		theQ.useLogOfParams=true;

		// sensitivity analysis
		DirectSensitivity theSensitivity(thePhi, theQ, theFEM); // AdjointSensitivity

		// initial parameters from current FEM setup
		q.resize( theSensitivity.getParameterHandler().getNumberOfParams(theFEM) );
		theQ.getCurrentParams(q,theFEM);

		// optimize ...
		printf("\n\n%% ***");
		theSensitivity.setupDynamicSim(dt,tsteps ,false,""/*skip writing output files*/, enablePreview?(&thePreview):NULL );
		theSensitivity.getParameterHandler().getCurrentParams(q, theFEM);
		theSensitivity.resetEvalCounter();
		double phi=0.0;
		int r = lbfgs.minimize(theSensitivity, q, phi);
		//int r = newton.minimize(theSensitivity, q, phi); // does not work so great here
		phi = theSensitivity.bestRunPhiValue;
		q   = theSensitivity.bestRunParameters;
		theQ.setNewParams(q,theFEM); // also reset the FEM parameters just in case

		printf("\n%% *** solver iterations %d, fcn.evals %d, objective function value %.4lg, obj. grad. norm %.4lg", r, theSensitivity.getEvalCounter(), phi, theSensitivity.bestRunGradient.norm());
		cout << endl << "params  = [ " << q.block(0,0,q.size()>10?10:q.size(),1).transpose() << " ]; " << (q.size()>10?"% ... ":"");
	}
	else{ // previously computed result: % obj.fcn. value  2.652e-09 ( 7.657e-12), reg.fcn.      0, grad.norm  3.326e-11, runtime 3368.29 // q = [ -0.394185    2.9528  -13.5266 ] 
		GlobalElasticMaterialParameterHandler elQ(bodyId); GlobalPSViscosityMaterialParameterHandler visQ(bodyId); CombinedParameterHandler theQ(visQ,elQ); theQ.useLogOfParams=true;
		q.resize( theQ.getNumberOfParams(theFEM) );
		q[0] = -0.394185; q[1] = 2.9528 ; q[2] =  -13.5266; 
		theQ.setNewParams(q, theFEM);

		// run with output
		fileName.str(""); fileName.clear(); fileName << outDir << argv[1] << "_global";
		DirectSensitivity theSensitivity(thePhi, theQ, theFEM);
		theSensitivity.bestRunParameters = q;
		theSensitivity.setupDynamicSim(dt,tsteps,false, fileName.str(), NULL );
	
		theSensitivity(theSensitivity.bestRunParameters,q); // writes gradient into q

	}
	if( 1 /*local param est*/){
		/** / // refine using a principal-stretch material
		printf("\n\n%% optimizing principal stretch material ... \n");
		vtkSmartPointer<HomogeneousMaterial> psMat = vtkSmartPointer<PrincipalStretchMaterial>::New();
		Eigen::VectorXd psMatParams( psMat->getNumberOfParameters() );
		double la,mu; theFEM.getMaterialModels()[bodyId]->getElasticParams(theFEM.getMaterialParameters(0).data(),la,mu);
		psMat->setElasticParams(psMatParams.data(), la, mu); psMat->setDensity(psMatParams.data(), 100);
		theFEM.initializeMaterialModel(bodyId,psMat, psMatParams.data());
		GlobalPrincipalStretchMaterialParameterHandler theQ(bodyId); //ParameterHandler doNothing; CombinedParameterHandler theQ(elQ,doNothing); theQ.useLogOfParams=true;
		theQ.alpha=5e-11; // for psm-xp 20 low-res mesh

		/*/ //refine with a per-element elastic optimization
		printf("\n\n%% optimizing per-element parameters ... \n");
		PerElementElasticMaterialParameterHandler elQ(bodyId); ParameterHandler doNothing(bodyId); CombinedParameterHandler theQ(elQ,doNothing); theQ.useLogOfParams=true; // results look better if we don't allow changes to viscosity anymore
		//PerElementDensityMaterialParameterHandler rhoQ(bodyId); ParameterHandler doNothing(bodyId); CombinedParameterHandler theQ(rhoQ,doNothing); theQ.useLogOfParams=true;
		//PerElementDensityMaterialParameterHandler rhoQ(bodyId,true); PerElementElasticMaterialParameterHandler elQ(bodyId); CombinedParameterHandler theQ(rhoQ,elQ); theQ.useLogOfParams=true;
		/**/

		q.resize(theQ.getNumberOfParams(theFEM));
		AdjointSensitivity theSensitivity(thePhi, theQ, theFEM); 
		theQ.getCurrentParams(q, theFEM); // read current params from FEM (mapping in param handler may have changed)

		fileName.str(""); fileName.clear(); fileName << outDir << argv[1];
		theSensitivity.setupDynamicSim(dt,tsteps ,false, fileName.str(), enablePreview?(&thePreview):NULL );

		double phi=0.0;
		int r = lbfgs.minimize(theSensitivity, q, phi);
		phi = theSensitivity.bestRunPhiValue;
		printf("\n%% *** solver iterations %d, fcn.evals %d, objective function value %.4lg, obj. grad. norm %.4lg", r, theSensitivity.getEvalCounter(), phi, theSensitivity.bestRunGradient.norm());
		q = theSensitivity.bestRunParameters;
		cout << endl << endl << "params  = [ " << q.transpose() << " ]; ";
		q = theSensitivity.bestRunGradient;
		cout << endl << endl << "dphi_dq = [ " << q.transpose() << " ]; ";

		// final run with output
		fileName.str(""); fileName.clear(); fileName << outDir << argv[1];
		theSensitivity.setupDynamicSim(dt,tsteps,false, fileName.str(), NULL );
	
		theSensitivity(theSensitivity.bestRunParameters,q); // writes gradient into q
	}

	return 0;
}


// version for negative-ish Poisson effect -- not working so well ...
//int main_Cylinder_5x2cm(int argc, char* argv[]){
//	bool enablePreview=false;
//
//	double dt=0.005, t_end=0.2; unsigned int tsteps = (t_end+0.5*dt)/dt;
//
//
//	std::string bndFileName, outDir="../_out/";
//	std::stringstream fileName;
//
//	LinearFEM theFEM;
//	theFEM.useBDF2=true; // BDF2 works with both direct and adjoint sensitivity analysis - however accuracy of adjoint method is sometimes not great
//	theFEM.doPrintResiduals=false;
//	theFEM.loadMeshFromElmerFiles(argv[1]);
//
//	//Remesher remesh; remesh.setTargetEdgeLengthFromMesh(theFEM.mesh); remesh.targetEdgeLength*=0.8;
//	//remesh.remesh(theFEM); printf("\n%% remeshed to %d elems", theFEM.getNumberOfElems());
//
//	MeshPreviewWindow thePreview(theFEM.mesh.Get());
//
//	// boundary conditions
//	class BoundaryField : public VectorField{ public: virtual void eval(Eigen::Vector3d& p, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
//		p=x0; // simple clamp
//	} } bc;
//	theFEM.setBoundaryCondition(1, bc);
//
//	// gravity
//	class GravityField : public VectorField{ public: virtual void eval(Eigen::Vector3d& g, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
//		g.setZero(); g[2]=-9.81;
//	} } g;
//	theFEM.setExternalAcceleration(3, g);
//
//	// elastic material model
//	theFEM.initializeMaterialModel(3, LinearFEM::ISOTROPIC_NEOHOOKEAN,20,40,100);
//
//	// damping model
//	vtkSmartPointer<PowerSeriesViscosityModel> viscMdl = vtkSmartPointer<PowerSeriesViscosityModel>::New();
//	viscMdl->powerIndices.resize(1); viscMdl->powerIndices(0)=1.0;
//	Eigen::VectorXd coeffs( viscMdl->getNumberOfParameters() ); coeffs(0)=2.0;
//	theFEM.initializeViscosityModel(3, viscMdl,  coeffs.data() );
//	cout << endl << "% power series viscous damping: flow indices are " << viscMdl->powerIndices.transpose() << ", coeffs are " << coeffs.transpose();
//
//	// objective function
//	class TargetDisplacementField : public VectorField{ public:
//		virtual void eval(Eigen::Vector3d& p, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
//			p = x-x0; // don't care for early times ... set target to current displacement
//			if( t > (t_end - 0.5*dt) ) {
//				p = +0.2*x0; // x* = x0 + u*
//				p[2]=-0.02;
//			}
//		}	
//		double dt, t_end;
//		TargetDisplacementField(double dt_, double t_end_) : dt(dt_), t_end(t_end_){}
//	} uStar(dt,t_end);
//	BoundaryFieldObjectiveFunction thePhi;
//	thePhi.addTargetField(2,uStar);
//
//	// for debug output
//	thePhi.debugFileName=outDir; thePhi.debugFileName.append(argv[1]); thePhi.debugFileName.append("_objFcnDebug_");
//
//	// solver configuration
//	LBFGSpp::LBFGSParam<double> optimOptions;
//	optimOptions.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_WOLFE; // _STRONG
//	optimOptions.m = 5;
//	optimOptions.epsilon = 1e-20;
//	optimOptions.past = 1; // compare to previous iteration to detect if we got stuck ...
//	optimOptions.delta = 1e-6; // relative change of objective function to previous iteration below which we give up
//	optimOptions.max_iterations = 150;
//	LBFGSpp::LBFGSSolver<double> lbfgs(optimOptions); // don't change optimOptions later, it's stored by ref!
//	
//	//LBFGSpp::LBFGSParam<double> optimOptions2(optimOptions);											
//	//optimOptions2.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_ARMIJO;
//	//LBFGSpp::NewtonSolver<double,Eigen::ColPivHouseholderQR<Eigen::MatrixXd> > newton(optimOptions);
//
//	// parameter vector ...
//	Eigen::VectorXd q;
//
//	if( 1 /*global parameter estimation*/){
//		// parameter handler
//		GlobalElasticMaterialParameterHandler elQ; GlobalPSViscosityMaterialParameterHandler visQ; CombinedParameterHandler theQ(visQ,elQ);
//
//		// sensitivity analysis
//		DirectSensitivity theSensitivity(thePhi, theQ, theFEM); // AdjointSensitivity
//
//		// initial parameters from current FEM setup
//		q.resize( theSensitivity.getParameterHandler().getNumberOfParams(theFEM) );
//		theQ.getCurrentParams(q,theFEM);
//
//		// optimize ...
//		printf("\n\n%% ***");
//		theSensitivity.setupDynamicSim(dt,tsteps ,false,""/*skip writing output files*/, enablePreview?(&thePreview):NULL );
//		theSensitivity.getParameterHandler().getCurrentParams(q, theFEM);
//		theSensitivity.resetEvalCounter();
//		double phi=0.0;
//		int r = lbfgs.minimize(theSensitivity, q, phi); // why won't Gauss-Newton work here -- maybe check H approx.
//		phi = theSensitivity.bestRunPhiValue;
//		q   = theSensitivity.bestRunParameters;
//		theQ.setNewParams(q,theFEM); // also reset the FEM parameters just in case
//
//		printf("\n%% *** solver iterations %d, fcn.evals %d, objective function value %.4lg, obj. grad. norm %.4lg", r, theSensitivity.getEvalCounter(), phi, theSensitivity.bestRunGradient.norm());
//		cout << endl << "params  = [ " << q.block(0,0,q.size()>10?10:q.size(),1).transpose() << " ]; " << (q.size()>10?"% ... ":"");
//	}
//
//	// now take the current parameters and run per-element elastic optimization 
//	PerElementElasticMaterialParameterHandler elQ; ParameterHandler doNothing; CombinedParameterHandler theQ(elQ,doNothing);
//	q.resize(theQ.getNumberOfParams(theFEM));
//	theQ.getCurrentParams(q, theFEM); // read current params from FEM (mapping in param handler may have changed)
//	AdjointSensitivity theSensitivity(thePhi, theQ, theFEM); 
//	theSensitivity.setupDynamicSim(dt,tsteps ,false,""/*skip writing output files*/, enablePreview?(&thePreview):NULL );
//	double phi=0.0;
//	int r = lbfgs.minimize(theSensitivity, q, phi);
//	phi = theSensitivity.bestRunPhiValue;
//	q   = theSensitivity.bestRunParameters;
//	printf("\n%% *** solver iterations %d, fcn.evals %d, objective function value %.4lg, obj. grad. norm %.4lg", r, theSensitivity.getEvalCounter(), phi, theSensitivity.bestRunGradient.norm());
//	cout << endl << "params  = [ " << q.block(0,0,q.size()>10?10:q.size(),1).transpose() << " ]; " << (q.size()>10?"% ... ":"");
//
//
//
//	// final run with output
//	fileName.str(""); fileName.clear(); fileName << outDir << argv[1];
//	theSensitivity.setupDynamicSim(dt,tsteps,false, fileName.str(), NULL );
//	
//	theSensitivity(theSensitivity.bestRunParameters,q); // writes gradient into q
//}


int main_Bunny(int argc, char* argv[]){
	std::string outDir="../_out/";
	std::stringstream fileName;
	fileName << outDir << argv[1];

	const double dt=1.0/240.0, t_end=0.75; const unsigned int tsteps = (t_end+0.5*dt)/dt;
	const unsigned int baseID=5, bodyID=8;

	/**/ // this setup is from the original example in the submitted paper
	class GravityField : public VectorField{ public: virtual void eval(Eigen::Vector3d& g, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		g.setZero(); g[2]=-9.81;
	} } g;
	class BoundaryField : public VectorField{ public: virtual void eval(Eigen::Vector3d& p, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		//p=x0;
		if( t<0.0  ) t = 0.0;
		if( t>0.25 ) t = 0.25;
		p = x0
			+ Eigen::Vector3d::UnitX()*(cos(17.0*M_PI*t)-1.0)*0.001
			+ Eigen::Vector3d::UnitY()*(cos(13.0*M_PI*t)-1.0)*0.005
			+ Eigen::Vector3d::UnitZ()*(cos( 7.0*M_PI*t)-1.0)*0.0;
	} } bc;
	/*/ // this setup is for the additional example in the revised version
	class GravityField : public VectorField{ public: virtual void eval(Eigen::Vector3d& g, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		g.setZero(); g[1]=-9.81;
	} } g;
	class BoundaryField : public VectorField{ public: virtual void eval(Eigen::Vector3d& p, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		//p=x0;
		if( t<0.0  ) t = 0.0;
		if( t>0.25 ) t = 0.25;
		p = x0
			+ Eigen::Vector3d::UnitX()*(cos(13.0*M_PI*t)-1.0)*0.0
			+ Eigen::Vector3d::UnitY()*(cos(15.0*M_PI*t)-1.0)*0.001
			+ Eigen::Vector3d::UnitZ()*(cos( 9.0*M_PI*t)-1.0)*0.005;
			//+ Eigen::Vector3d::UnitX()*(cos(13.0*M_PI*t)-1.0)*0.004
			//+ Eigen::Vector3d::UnitY()*(cos(15.0*M_PI*t)-1.0)*0.002
			//+ Eigen::Vector3d::UnitZ()*(cos( 9.0*M_PI*t)-1.0)*0.001;
	} } bc;
	/**/
	std::vector<vtkSmartPointer<TemporalInterpolationField> > trackedFields; std::vector<unsigned int> trackedIDs; // storage for target fields

	// create FEM, load the mesh
	LinearFEM theFEM; theFEM.useBDF2=true;
	theFEM.loadMeshFromElmerFiles(argv[1]);

	//Remesher remesh; remesh.setTargetEdgeLengthFromMesh(theFEM.mesh); // this one creates 5.1k surface elements and ~50k volume elements
	//remesh.targetEdgeLength *= 0.66; remesh.remesh(theFEM); printf("\n%% remeshed to %d elems \n", theFEM.getNumberOfElems());

	Remesher remesh; remesh.setTargetEdgeLengthFromMesh(theFEM.mesh); remesh.dOpt[22]=1e5; // 22 is surface Hausdorff distance threshold -- set it high so it won't affect anything
	/**/ remesh.targetEdgeLength *= 2.0;
	/* /  remesh.targetEdgeLength *= 5.0; /**/
	//remesh.remesh(theFEM); printf("\n%% remeshed to %d elems \n", theFEM.getNumberOfElems()); 

	// apply boundary conditions and gravity load
	theFEM.setBoundaryCondition(baseID, bc);
	theFEM.setExternalAcceleration(bodyID, g);

	// material model
	vtkSmartPointer<PowerLawViscosityModel> viscMdl = vtkSmartPointer<PowerLawViscosityModel>::New(); Eigen::VectorXd viscParam( viscMdl->getNumberOfParameters() ); viscParam.setZero();
	switch(0){
	case 0: // these are the original ground truth parameters
		theFEM.initializeMaterialModel(bodyID, LinearFEM::ISOTROPIC_NEOHOOKEAN,12e3,110e2,103);
		viscMdl->setViscosity( viscParam.data(), 8.0 ); viscMdl->setPowerLawH( viscParam.data(), 0.8 );
		break;
	case 8: // these are the optimized parameters from the original setup for 8.5k elems (8.123978522	0.982183328	5413.223308	9967.549048)
		theFEM.initializeMaterialModel(bodyID, LinearFEM::ISOTROPIC_NEOHOOKEAN, 5413.223308, 9967.549048, 103);
		viscMdl->setViscosity( viscParam.data(), 8.123978522 ); viscMdl->setPowerLawH( viscParam.data(), 0.982183328 );
		break;
	case 2: // these are the optimized parameters from the original setup for 2.7k elems (6.785310127	1.142591582	5916.557172	8738.291771)
		theFEM.initializeMaterialModel(bodyID, LinearFEM::ISOTROPIC_NEOHOOKEAN, 5916.557172, 8738.291771, 103);
		viscMdl->setViscosity( viscParam.data(), 6.785310127 ); viscMdl->setPowerLawH( viscParam.data(), 1.142591582 );
		break;
	case 1: // these are the optimized parameters from the original setup for 1.2k elems (2.330013357	1.756194057	6302.804676	6109.987461)
		theFEM.initializeMaterialModel(bodyID, LinearFEM::ISOTROPIC_NEOHOOKEAN, 6302.804676, 6109.987461, 103);
		viscMdl->setViscosity( viscParam.data(), 2.330013357 ); viscMdl->setPowerLawH( viscParam.data(), 1.756194057 );
		break;
	}
	theFEM.initializeViscosityModel(bodyID, viscMdl,  viscParam.data() );


	if( argc>2 ){ // build objective function from given displacement data file ...
		printf("\n%% Target file \"%s\"", argv[2]);
		AverageBoundaryValueObjectiveFunction thePhi;
		if( TemporalInterpolationField::buildFieldsFromTextFile( argv[2], trackedFields, trackedIDs) < 0 ) return -1;
		for(int i=0; i<trackedIDs.size(); ++i){ thePhi.addTargetField( trackedIDs[i] , trackedFields[i] ); }
		fileName.str(""); fileName.clear();
		fileName << outDir << argv[1] << "_targetBCs";
		TemporalInterpolationField::writeFieldsToVTU(fileName.str(), 3*tsteps, trackedFields, trackedIDs);

		printf("\n%% Optimizing for combined global visco-elastic parameters. "); //printf("\n%% Optimizing for combined global viscous and local elastic parameters. "); //
		GlobalElasticMaterialParameterHandler     elasticParamHandler(bodyID); //PerElementElasticMaterialParameterHandler elasticParamHandler; //
		GlobalPLViscosityMaterialParameterHandler viscousParamHandler(bodyID);
		CombinedParameterHandler combinedParamHandler(viscousParamHandler, elasticParamHandler);
		combinedParamHandler.useLogOfParams = true; printf("\n%% Optimizing for log of parameters ");
		ParameterHandler& theQ ( combinedParamHandler ); // choose which one we use ...
		
		double phi=0.0;
		Eigen::VectorXd dphi, q( theQ.getNumberOfParams(theFEM) );
		theQ.getCurrentParams(q, theFEM);

		AdjointSensitivity theSensitivity(thePhi, theQ, theFEM);
		theSensitivity.setupDynamicSim(dt,tsteps, true ,""/*skip writing output files*/, NULL );
		
		LBFGSpp::LBFGSParam<double> optimOptions = defaultLBFGSoptions<LBFGSpp::LBFGSParam<double> >();
		LBFGSpp::LBFGSSolver<double> solver(optimOptions);
		int r = solver.minimize(theSensitivity, q, phi);

		int ev = theSensitivity.getEvalCounter();
		phi  = theSensitivity.bestRunPhiValue;
		q    = theSensitivity.bestRunParameters;
		dphi = theSensitivity.bestRunGradient;
		cout << endl << "% solver iterations " << r << ", fcn.evals " << ev;
		fileName.str(""); fileName.clear();
		fileName << outDir << argv[1] << "_optimResult.txt";
		ofstream optOut(fileName.str());
		optOut << "% solver iterations " << r << ", fcn.evals " << ev << ", objective function value " << phi << endl;
		optOut << endl << "% params:" << endl << q << endl;
		optOut << endl << "% gradient:" << endl << dphi << endl;
		optOut.close();

		theQ.setNewParams( theSensitivity.bestRunParameters, theFEM );
		theFEM.reset();
	}



	// *******
	// fwd sim
	// *******

	trackedIDs.clear(); trackedFields.clear();
	for(int i=1; i<=4; ++i) trackedIDs.push_back(i); trackedIDs.push_back(7); // choose which boundary regions to track in forward sim.
	for(int i=0; i<trackedIDs.size(); ++i) trackedFields.push_back(vtkSmartPointer<TemporalInterpolationField>::New() );

	// write tracked fields to plain text log
	fileName.str(""); fileName.clear();
	fileName << outDir << argv[1] << "_trackedBCs.log";
	ofstream tbOut(fileName.str().c_str()); tbOut << std::setprecision(18);
	for(int i=0; i<trackedIDs.size(); ++i){
		tbOut << trackedIDs[i] << " ";
	}	tbOut << endl;

	// static initial solve
	theFEM.updateAllBoundaryData();
	theFEM.assembleMassAndExternalForce();
	theFEM.staticSolve();

	fileName.str(""); fileName.clear(); fileName << outDir << argv[1] << "_" << std::setfill ('0') << std::setw(5) << (int)0;
	theFEM.saveMeshToVTUFile(fileName.str(),true);

	Eigen::Vector3d xB;
	tbOut << theFEM.simTime << " "; // should be 0.0 here
	for(int i=0; i<trackedIDs.size(); ++i){
		theFEM.computeAverageDeformedCoordinateOfBoundary(trackedIDs[i],xB);
		trackedFields[i]->addPoint(0.0, xB);
		tbOut << xB[0] << " " << xB[1] << " " << xB[2] << " ";
	}	tbOut << endl;

	// dynamic sim
	for(int step=0; step<tsteps; ++step){	printf("\n%% step %5d/%d ", step+1,tsteps);

		theFEM.updateAllBoundaryData();
		theFEM.dynamicImplicitTimestep(dt);

		fileName.str(""); fileName.clear(); fileName << outDir << argv[1] << "_" << std::setfill ('0') << std::setw(5) << (step+1);
		theFEM.saveMeshToVTUFile(fileName.str());

		tbOut << theFEM.simTime << " ";
		for(int i=0; i<trackedIDs.size(); ++i){
			theFEM.computeAverageDeformedCoordinateOfBoundary(trackedIDs[i],xB);
			trackedFields[i]->addPoint(theFEM.simTime, xB);
			tbOut << xB[0] << " " << xB[1] << " " << xB[2] << " ";
		}	tbOut << endl;
	}

	tbOut.close();

	fileName.str(""); fileName.clear(); fileName << outDir << argv[1] << "_trackedFields";
	TemporalInterpolationField::writeFieldsToVTU( fileName.str() , 3*tsteps, trackedFields, trackedIDs );

	return 0;
}

int main_Dragon(int argc, char* argv[]){
	std::string outDir="../_out/";
	std::stringstream fileName;
	fileName << outDir << argv[1];

	const double dt=1.0/240.0, t_end=0.75; const unsigned int tsteps = (t_end+0.5*dt)/dt;
	const unsigned int footFL=1, footFR=2, footRL=3, footRR=4, bodyID=13;

	class GravityField : public VectorField{ public: virtual void eval(Eigen::Vector3d& g, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		g.setZero(); g[1]=-9.81; // Dragon is a Y-up model
	} } g;
	class BoundaryField : public VectorField{ public: virtual void eval(Eigen::Vector3d& p, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		p=x0;
	} } bc;
	class BoundaryFieldMoving : public VectorField{ public: virtual void eval(Eigen::Vector3d& p, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		//p=x0;
		double alpha=1.0;
		if( t<0.0 ) t = 0.0;
		if( t>0.5 ) t = 0.5;
		if( t>0.3 ) alpha = 1.0-(t-0.3)/(0.5-0.3);
		p = x0
			+ Eigen::Vector3d::UnitX()*(cos(6.0*M_PI*t)-1.0)*0.02*alpha
			- Eigen::Vector3d::UnitY()*(cos(8.0*M_PI*t)-1.0)*0.03*alpha
			+ Eigen::Vector3d::UnitZ()*(cos(13.0*M_PI*t)-1.0)*0.05*alpha;
	} } bcMoving;
	std::vector<vtkSmartPointer<TemporalInterpolationField> > trackedFields; std::vector<unsigned int> trackedIDs; // storage for target fields


	// create FEM, load the mesh and apply boundary conditions and gravity load
	LinearFEM theFEM; theFEM.useBDF2=true; theFEM.FORCE_BALANCE_EPS = 1e-6;
	theFEM.loadMeshFromElmerFiles(argv[1]);
	theFEM.setBoundaryCondition(footRL, bc); theFEM.setBoundaryCondition(footRR, bc);
	theFEM.setBoundaryCondition(footFL, bcMoving); theFEM.setBoundaryCondition(footFR, bcMoving);
	theFEM.setExternalAcceleration(bodyID, g);

	// material model
	theFEM.initializeMaterialModel(bodyID, LinearFEM::ISOTROPIC_NEOHOOKEAN,12e3,110e3,103); // roughly the FlexFoam-III data
	vtkSmartPointer<PowerLawViscosityModel> viscMdl = vtkSmartPointer<PowerLawViscosityModel>::New(); Eigen::VectorXd viscParam( viscMdl->getNumberOfParameters() ); viscParam.setZero();
	viscMdl->setViscosity( viscParam.data(), 8.0 ); viscMdl->setPowerLawH( viscParam.data(), 0.8 );
	theFEM.initializeViscosityModel(bodyID, viscMdl,  viscParam.data() );

	trackedIDs.clear(); trackedFields.clear();
	for(int i=5; i<=12; ++i) trackedIDs.push_back(i); // choose which boundary regions to track in forward sim.
	for(int i=0; i<trackedIDs.size(); ++i) trackedFields.push_back(vtkSmartPointer<TemporalInterpolationField>::New() );

	// static initial solve
	theFEM.updateAllBoundaryData();
	theFEM.assembleMassAndExternalForce();
	theFEM.staticSolve();

	fileName.str(""); fileName.clear(); fileName << outDir << argv[1] << "_" << std::setfill ('0') << std::setw(5) << (int)0;
	theFEM.saveMeshToVTUFile(fileName.str());

	Eigen::Vector3d xB;
	for(int i=0; i<trackedIDs.size(); ++i){ theFEM.computeAverageDeformedCoordinateOfBoundary(trackedIDs[i],xB); trackedFields[i]->addPoint(0.0, xB); }

	// dynamic sim
	for(int step=0; step<tsteps; ++step){	printf("\n%% step %5d/%d ", step+1,tsteps);

		theFEM.updateAllBoundaryData();
		theFEM.dynamicImplicitTimestep(dt);

		fileName.str(""); fileName.clear(); fileName << outDir << argv[1] << "_" << std::setfill ('0') << std::setw(5) << (step+1);
		theFEM.saveMeshToVTUFile(fileName.str());

		for(int i=0; i<trackedIDs.size(); ++i){ theFEM.computeAverageDeformedCoordinateOfBoundary(trackedIDs[i],xB); trackedFields[i]->addPoint(theFEM.simTime, xB); }
	}

	fileName.str(""); fileName.clear(); fileName << outDir << argv[1] << "_trackedFields";
	TemporalInterpolationField::writeFieldsToVTU( fileName.str() , 3*tsteps, trackedFields, trackedIDs );


	return 0;
}

int main(int argc, char* argv[]){
	//// for debugging: break on floating point math errors ...
	//_clearfp();
	//_controlfp(_controlfp(0, 0) & ~(_EM_INVALID | _EM_ZERODIVIDE | _EM_OVERFLOW), _MCW_EM);

	if( argc<=1 ) return -1; // need at least the mesh file name as parameter
	Eigen::initParallel();
	std::string outDir="../_out/";


	if(0 /*QnD convert tracked boundary condition from plain text to VTU format for visualization*/){	
		std::stringstream fileName;
		std::vector<vtkSmartPointer<TemporalInterpolationField> > trackedFields; std::vector<unsigned int> trackedIDs; // storage for target fields
		TemporalInterpolationField::buildFieldsFromTextFile( argv[1], trackedFields, trackedIDs);
		fileName.str(""); fileName.clear();
		fileName << argv[1] << "_";
		TemporalInterpolationField::writeFieldsToVTU(fileName.str(), 3*trackedFields[0]->xSpline->GetNumberOfPoints(), trackedFields, trackedIDs);
		return 0;
	}

	// these examples take care of their own output directory ...
	if( std::string(argv[1]).find("contact")!=std::string::npos )
		return main_contact(argc,argv);

	if( std::string(argv[1]).find("PointCloudDemo")!=std::string::npos )
		return main_PointCloudDemo(argc,argv);

	if( std::string(argv[1]).find("YuMiFoamBar")!=std::string::npos )
		return main_YuMiFoamBar(argc,argv);

	if( std::string(argv[1]).find("Xworm")!=std::string::npos )
		return main_Xworm(argc,argv);

	if( std::string(argv[1]).find("CylinderMarkers")!=std::string::npos )
		return main_CylinderMarkers(argc,argv);

	if( std::string(argv[1]).find("TPUbar")!=std::string::npos )
		return main_TPUbar(argc,argv);

	if( std::string(argv[1]).find("SiliconeHand")!=std::string::npos )
		return main_SiliconeHand(argc,argv);


#ifdef _WINDOWS
	// argv[1] contains the input mesh file name - we'll write outputs to outDir/inputMeshDir/inputMeshName... make sure inputMeshDir exists ...
	std::string sc("for %f in (\""); sc.append(outDir).append(argv[1]).append("\") do mkdir %~dpf"); std::replace( sc.begin(), sc.end(), '/', '\\');
	int sr = system(sc.c_str());
	//printf("\n%% \"%s\" returned %d\n",sc.c_str(), sr);
	//ToDo: implement for Unix systems ('touch' cmd?)
#endif // _WINDOWS

	if( std::string(argv[1]).find("Box_wMarkers")!=std::string::npos )
		return main_Box_wMarkers(argc,argv);

	if( std::string(argv[1]).find("Bunny")!=std::string::npos )
		return main_Bunny(argc,argv);

	if( std::string(argv[1]).find("Dragon")!=std::string::npos )
		return main_Dragon(argc,argv);

	if( std::string(argv[1]).find("Cylinder_5x2cm")!=std::string::npos )
		return main_Cylinder_5x2cm(argc,argv);
	if( std::string(argv[1]).find("Cylinder5x2cm")!=std::string::npos )
		return main_Cylinder_5x2cm(argc,argv);


	// basic tests and experiments -- should not be reached ... unless we're testing ...
	std::string bndFileName;
	std::stringstream fileName;
	fileName << outDir << argv[1];

	// some generic tests .... mostly intended for the Cylinder_5x2cm mesh

	class BoundaryField : public VectorField{ public: virtual void eval(Eigen::Vector3d& p, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		p=x0;
		//Eigen::Quaterniond r(Eigen::AngleAxisd(0.2,Eigen::Vector3d::UnitY())); Eigen::Vector3d s(0.005,0.0,0.0), c(0.0,0.0,0.02);
		//p = (r*(x0-c))+c+s; // rotate by r with centre of rotation c, then shift by s
	} } bc;
	class BoundaryField2 : public VectorField{ public: virtual void eval(Eigen::Vector3d& p, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		p=x; p[2]-=0.0001; p[0]+=10.0*t*0.03*p[1]; p[1]-=10.0*t*0.03*p[0]; //x0+Eigen::Vector3d(0.0,0.0,-0.02);//
	} } bc2;
	class TractionField : public VectorField{ public: virtual void eval(Eigen::Vector3d& q, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		q = Eigen::Vector3d( 0.0, 0.5, -1.5 );
	} } q;
	class GravityField : public VectorField{ public: virtual void eval(Eigen::Vector3d& g, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		g.setZero(); g[2]=-9.81;
	} } g;
	class TargetDisplacementField : public VectorField{ public: virtual void eval(Eigen::Vector3d& p, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		p = Eigen::Vector3d(0.0,0.0,-0.005); //
		if( t < 0.18 ) p = x-x0; // don't care for early times ... set target to current displacement
	} } uTarget;

	if( 1 /*test a sensitivity analysis case ...*/){
		LinearFEM theFEM; theFEM.useBDF2=true; // we now have full support for BDF2 in both direct and adjoint sensitivity analysis
		theFEM.loadMeshFromElmerFiles(argv[1]);
		theFEM.setBoundaryCondition(1, bc);
		theFEM.setExternalAcceleration(3, g);

		theFEM.initializeMaterialModel(3, LinearFEM::ISOTROPIC_NEOHOOKEAN,20e3,40e2,1e5);
		//vtkSmartPointer<HomogeneousMaterial> psMat = vtkSmartPointer<PrincipalStretchMaterial>::New();
		//Eigen::VectorXd psMatParams( psMat->getNumberOfParameters() );
		//psMat->setElasticParams(psMatParams.data(), 20e3, 40e2); psMat->setDensity(psMatParams.data(), 1e5);
		//theFEM.initializeMaterialModel(3,psMat, psMatParams.data());
		//theFEM.doPrintResiduals=false;

		/**/
		vtkSmartPointer<PowerLawViscosityModel> viscMdl = vtkSmartPointer<PowerLawViscosityModel>::New(); Eigen::VectorXd viscParam( viscMdl->getNumberOfParameters() ); viscParam.setZero();
		viscMdl->setViscosity( viscParam.data(), 200.0 ); viscMdl->setPowerLawH( viscParam.data(), 1.0 );
		theFEM.initializeViscosityModel(3, viscMdl,  viscParam.data() );
		/** /
		vtkSmartPointer<PowerSeriesViscosityModel> viscMdl = vtkSmartPointer<PowerSeriesViscosityModel>::New();
		viscMdl->powerIndices.resize(7); viscMdl->powerIndices.setLinSpaced(0.5,2.0);
		Eigen::VectorXd coeffs( viscMdl->getNumberOfParameters() ); coeffs.setLinSpaced(1.01,10.001);
		theFEM.initializeViscosityModel(3, viscMdl,  coeffs.data() );
		cout << endl << "% power series viscous damping: flow indices are " << viscMdl->powerIndices.transpose() << ", coeffs are " << coeffs.transpose();
		/** / // try not to combine this one with the PrincipalStretchMaterial elasticity -- strange things may happen
		vtkSmartPointer<EigenmodeViscosityModel> viscMdl = vtkSmartPointer<EigenmodeViscosityModel>::New();
		viscMdl->setNumberOfAdjustmentPoints(4); viscMdl->setNumberOfModes( 150 ); // must initialize after changing these!
		viscMdl->initializeToConstantViscosity( 100.0, theFEM, 3 );
		//// test adjustment ...
		//Eigen::VectorXd frq, coeff;
		//viscMdl->getAdjustmentFrequencies(frq); viscMdl->getViscosityCoefficients(coeff);
		//for(int i=0; i<frq.size(); ++i){
		//	double fRatio = (frq.maxCoeff() - frq(i)) / (frq.maxCoeff()-frq.minCoeff());
		//	coeff(i) *= (1.0 + 2.0* fRatio*fRatio ) ; // example: triple the damping on the lowest frequency, keep highest unchanged, quadratic in between
		//	//coeff(i) *= (1.0-fRatio)*(1.0-fRatio); // example: remove damping on the lowest frequency, keep highest unchanged, quadratic in between
		//}
		//viscMdl->setViscosityCoefficients(coeff, theFEM);
		 //Eigen::Vector2d c; c << 100.0, 300.0;
		 //viscMdl->setViscosityCoefficients(c, theFEM);
		/**/

		MeshPreviewWindow thePreview(theFEM.mesh.Get());
		
		
		//GlobalElasticMaterialParameterHandler theQ;
		//PerElementElasticMaterialParameterHandler theQ;
		//GlobalPLViscosityMaterialParameterHandler theQ;
		//GlobalPSViscosityMaterialParameterHandler theQ;

		//PerElementElasticMaterialParameterHandler elQ; GlobalPSViscosityMaterialParameterHandler visQ; CombinedParameterHandler theQ(visQ,elQ);
		GlobalElasticMaterialParameterHandler elQ(3); GlobalPSViscosityMaterialParameterHandler visQ(3); CombinedParameterHandler theQ(visQ,elQ); //theQ.useLogOfParams=true;
		//EigenmodeViscosityMaterialParameterHandler theQ;
		//GlobalPrincipalStretchMaterialParameterHandler theQ(3);
		//PerElementDensityMaterialParameterHandler densityQ(3); //GlobalDensityParameterHandler densityQ(3);
		//GlobalElasticMaterialParameterHandler elQ(3); CombinedParameterHandler theQ(elQ,densityQ); theQ.useLogOfParams=true;

		BoundaryFieldObjectiveFunction thePhi(1e8);
		thePhi.addTargetField(2,uTarget);
		AdjointSensitivity theSensitivity(thePhi, theQ, theFEM); //DirectSensitivity theSensitivity(thePhi, theQ, theFEM); //
		double dt=0.005, t_end=0.2; unsigned int tsteps = (t_end+0.5*dt)/dt;
		double phi=0.0;
		Eigen::VectorXd dphi_dq;

		if( 0 /*do FD-check ... better use the one for grad and H below*/){
			thePreview.render();
			//phi = theSensitivity.solveStaticFEM(dphi_dq);

			theSensitivity.setupDynamicSim(dt,tsteps ,false,fileName.str(), &thePreview);
			double t0=omp_get_wtime();
			phi = theSensitivity.solveImplicitDynamicFEM(dphi_dq);
			Eigen::VectorXd dphi_dq0 = dphi_dq;
			printf("\n%% %9.4lg sec fist run", omp_get_wtime()-t0);

			//printf("\n%% objective function value = %.4lg, objective function gradient norm = %.4lg ", phi, dphi_dq.norm() );
			//cout << endl << "% objective function gradient: [ " << dphi_dq.transpose() << " ] ";

			theFEM.doPrintResiduals=false;
			theSensitivity.setupDynamicSim(dt,tsteps); // disable output files and preview
			double fdH=1e-7, tmp;
			Eigen::VectorXd params( theSensitivity.getParameterHandler().getNumberOfParams(theFEM) );
			theSensitivity.getParameterHandler().getCurrentParams(params, theFEM);
			tmp = params[0];
			params[0]+= fdH;
			t0=omp_get_wtime();
			double phi_d0 = theSensitivity(params,dphi_dq);
			params[0] = tmp;
			params[1]+= fdH;
			double phi_d1 = theSensitivity(params,dphi_dq);
			printf("\n%% %9.4lg sec avg FD runs", 0.5*(omp_get_wtime()-t0));
			printf("\n%% FD test: fd-approx obj.grad: [ %12.6lg %12.6lg ] ", (phi_d0-phi)/fdH, (phi_d1-phi)/fdH );
			printf("\n%% reported objective gradient: [ %12.6lg %12.6lg ] ", dphi_dq0(0), dphi_dq0(1));
			printf("\n%% FD test: fd-approx abs.err.: [ %12.6lg %12.6lg ] ", dphi_dq0(0) - (phi_d0-phi)/fdH, dphi_dq0(1) - (phi_d1-phi)/fdH );
		}

		theFEM.doPrintResiduals=false;
		
		Eigen::VectorXd q( theSensitivity.getParameterHandler().getNumberOfParams(theFEM) );

		//if( 0 && argc>3 /*single evaluation if we return to MatLab optimization function*/){
		//	sscanf(argv[2], "%lg", &q[0]);
		//	sscanf(argv[3], "%lg", &q[1]);
		//	theSensitivity.setupDynamicSim(dt,tsteps ,false,fileName.str());
		//	phi = theSensitivity(q, dphi_dq);
		//	std::cout << endl << phi << " " << dphi_dq.transpose();
		//}

		LBFGSpp::LBFGSParam<double> optimOptions;
		optimOptions.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_WOLFE;
		optimOptions.m = 5;
		optimOptions.epsilon = 1e-20; //1e-16; // use less strict eps for global parameters ...
		optimOptions.past = 1; // compare to previous iteration to detect if we got stuck ...
		optimOptions.delta = 1e-6; // relative change of objective function to previous iteration below which we give up
		optimOptions.max_iterations = 150;

		//if( 0 /*test LBFGS from cppoptlib*/){
		//	class SensitivityProblem : public cppoptlib::Problem<double>{ public:
		//		SensitivityAnalysis& sa; TVector lastGradient, lastX; double lastValue;
		//		SensitivityProblem(SensitivityAnalysis& sa_) : sa(sa_) {}
		//		virtual double value(const  TVector &x){
		//			if( lastX.size()==0 || !(x.isApprox(lastX,0.0)) ){
		//				lastX = x;
		//				lastValue = sa(x,lastGradient);
		//			}
		//			return lastValue;
		//		}
		//		virtual void gradient(const  TVector &x,  TVector &grad){
		//			if( lastX.size()==0 || !(x.isApprox(lastX,0.0)) ) value(x);
		//			else printf("\n%% *** used last grad *** ");
		//			grad = lastGradient;
		//		}
		//	} theProblem(theSensitivity);

		//	cppoptlib::LbfgsSolver<SensitivityProblem> solver;
		//	SensitivityProblem::TCriteria opt = solver.criteria(); // get solver defaults
		//	opt.fDelta = 1e-6;
		//	opt.gradNorm = 1e-20;
		//	opt.iterations = 150;
		//	solver.setStopCriteria(opt);

		//	printf("\n\n%% ***");
		//	theSensitivity.setupDynamicSim(dt,tsteps ,false,""/*skip writing output files*/, &thePreview);
		//	theSensitivity.getParameterHandler().getCurrentParams(q, theFEM);
		//	theSensitivity.resetEvalCounter();

		//	solver.minimize(theProblem, q);
		//	phi = theSensitivity.bestRunPhiValue;  // sometimes the solver messes up at the end, so load the parameters from the best sim run we've had
		//	q   = theSensitivity.bestRunParameters;

		//	printf("\n%% *** solver iterations %d, fcn.evals %d, objective function value %.4lg", solver.criteria().iterations, theSensitivity.getEvalCounter(), phi);
		//	cout << endl << "params  = [ " << q.block(0,0,q.size()>5?5:q.size(),1).transpose() << " ]; " << (q.size()>5?"% ... ":"");
		//	cout << " status " << solver.status();

		//}else
		if( 0 /*minimize with LBFGS++*/){
			LBFGSpp::LBFGSSolver<double> solver(optimOptions);

			printf("\n\n%% ***");
			theSensitivity.setupDynamicSim(dt,tsteps ,false,""/*skip writing output files*/, &thePreview);
			theSensitivity.getParameterHandler().getCurrentParams(q, theFEM);
			theSensitivity.resetEvalCounter();
			int r = solver.minimize(theSensitivity, q, phi);
			phi = theSensitivity.bestRunPhiValue;  // sometimes the solver messes up at the end, so load the parameters from the best sim run we've had
			printf("\n%% *** solver iterations %d, fcn.evals %d, objective function value %.4lg", r, theSensitivity.getEvalCounter(), phi);

			q = theSensitivity.bestRunGradient; // make sure we set q to parameter values after this, we'll need them later on ...
			cout << endl << endl << "dphi_dq = [ " << q.transpose() << " ]; ";

			q   = theSensitivity.bestRunParameters;
			cout << endl << endl << "params  = [ " << q.transpose() << " ]; ";
		}else
		if( 1 /*minimize with Gauss-Newton*/){
			DirectSensitivity theSensitivity(thePhi, theQ, theFEM); // override to direct sensitivity analysis
			theSensitivity.setupDynamicSim(dt,tsteps ,false,""/*skip writing output files*/, &thePreview);
			theSensitivity.getParameterHandler().getCurrentParams(q, theFEM);

			if( 1 /*FD check Hessian approximation*/ ){
				double fdH=1e-3, tmp;
				Eigen::VectorXd tmpV, grad_fd(  q.size() ); grad_fd.setZero();
				Eigen::MatrixXd H_fd( q.size(), q.size() ); H_fd.setZero();
				for(int k=0; k<q.size(); ++k){
					printf("\n%% FD-check sim-run %d/%d ... ", k+1, q.size()+1);
					tmp = q[k];
					q[k]+= fdH;
					grad_fd(k) = theSensitivity(q,tmpV);
					H_fd.col(k)= tmpV;
					//H_fd(k,k) =- theSensitivity(q,tmpV);
					q[k] = tmp;
					//q[k]-= fdH;
					//H_fd(k,k) -= theSensitivity(q,tmpV);
				}
				printf("\n%% FD-check final sim-run ... ");
				double val = theSensitivity(q,tmpV);
				for(int k=0; k<q.size(); ++k){ H_fd.col(k)-=tmpV; grad_fd(k)-=val; } H_fd /= fdH; grad_fd /= fdH;
				//double phi_0 = theSensitivity(q,tmpV);
				//H_fd.diagonal().array() += 2.0*phi_0; H_fd /= (fdH*fdH);
				cout << endl << "g_fd = [ " <<  grad_fd.transpose() << " ]";
				cout << endl << "g_sa = [ " <<  tmpV.transpose() << " ]";
				cout << endl << "g_er = [ " << (grad_fd-tmpV).transpose() << "]" << endl;
				cout << endl << "H_fd = [ " <<  H_fd << " ]";
				cout << endl << "H_sa = [ " << theSensitivity.getHessian() << " ]";
				cout << endl << "H_er = [ " << (H_fd-theSensitivity.getHessian()) << "]" << endl;

			}

			theSensitivity.resetEvalCounter();
			printf("\n\n%% ***");
			optimOptions.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_ARMIJO;
			LBFGSpp::NewtonSolver<double,Eigen::ColPivHouseholderQR<Eigen::MatrixXd> > solver(optimOptions);
			int r = solver.minimize(theSensitivity, q, phi);

			printf("\n%% *** solver iterations %d, fcn.evals %d, objective function value %.4lg", r, theSensitivity.getEvalCounter(), phi);
			q = theSensitivity.bestRunParameters;
			cout << endl << endl << "params  = [ " << q.transpose() << " ]; ";
			q = theSensitivity.bestRunGradient;
			cout << endl << endl << "dphi_dq = [ " << q.transpose() << " ]; ";
		}
		// run the sim one more time and write output files
		theFEM.reset();
		theSensitivity.setupDynamicSim(dt,tsteps ,false,fileName.str());
		theSensitivity.getParameterHandler().setNewParams(q, theFEM);
		phi = theSensitivity.solveImplicitDynamicFEM(q);
		//Eigen::Vector3d uFinal;
		//theSensitivity.getSimObject().computeAverageDisplacementOnBoundary(2,uFinal);
		//cout << endl << "% avg. surface displacement " << uFinal.transpose();
		//cout << endl << "dphi_dq = [ " << q.block(0,0,q.size()>5?5:q.size(),1).transpose() << " ]; "; if(q.size()>5) cout << "% ... ||.|| = " << q.norm();

	}

	Remesher r;

	if( 0 /*run a basic FEM sim ...*/){
		LinearFEM fem; fem.useBDF2=true;
		fem.loadMeshFromElmerFiles(argv[1]);
		r.setTargetEdgeLengthFromMesh(fem.mesh);

		MeshPreviewWindow preview(fem.mesh.Get());

		fem.setBoundaryCondition(1,bc );//, LinearFEM::BC_DIRICHLET, LinearFEM::Z_MASK);
		//fem.setBoundaryCondition(2,bc2);
		//fem.setExternalAcceleration(3,g);
		fem.setBoundaryCondition(2,q,BC_NEUMANN);

		double lameLambda=0.5, lameMu=1.0, density=2.5;
		//fem.initializeMaterialModel(3, LinearFEM::ISOTROPIC_LINEAR_ELASTIC, lameLambda, lameMu, density);
		fem.initializeMaterialModel(3, LinearFEM::ISOTROPIC_NEOHOOKEAN, lameLambda, lameMu, density);
		//vtkSmartPointer<HomogeneousMaterial> psMat = vtkSmartPointer<PrincipalStretchMaterial>::New();
		//Eigen::VectorXd psMatParams( psMat->getNumberOfParameters() );
		//psMat->setElasticParams(psMatParams.data(), lameLambda, lameMu); psMat->setDensity(psMatParams.data(), density);
		//fem.initializeMaterialModel(3,psMat, psMatParams.data());


		if( 0 /*test eigenmode computations*/){
			//r.targetEdgeLength *=0.75; r.remesh(fem);
			fem.assembleMassAndExternalForce();
			fem.assembleForceAndStiffness();
			//Eigen::VectorXd g= fem.f; fem.applyStaticBoundaryConditions(fem.K,g,fem.x,0.0); // also works

			double t1,t0=omp_get_wtime();
			/** /
			Spectra::SparseSymMatProd<double,Eigen::SelfAdjoint,SparseMatrixD::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor,SparseMatrixD::StorageIndex> K(fem.getStiffnessMatrix());
			Spectra::DenseCholesky<double> M(fem.getMassMatrix().toDense());
			Spectra::SymGEigsSolver<double, Spectra::SMALLEST_MAGN,
				Spectra::SparseSymMatProd<double,Eigen::SelfAdjoint,SparseMatrixD::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor,SparseMatrixD::StorageIndex>,
				Spectra::DenseCholesky<double>, Spectra::GEIGS_CHOLESKY> 
				eigs( &K,&M, 24, K.cols() );
			/*/ // there seems to be little difference in the runtime between these two methods (we could speed things up by lumping the masses and computing M^-1 K first ...)
			Spectra::SparseSymMatProd<double,Eigen::SelfAdjoint,SparseMatrixD::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor,SparseMatrixD::StorageIndex> K(fem.getStiffnessMatrix());
			Spectra::SparseRegularInverse<double,Eigen::SelfAdjoint,SparseMatrixD::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor,SparseMatrixD::StorageIndex> M(fem.getMassMatrix());
			Spectra::SymGEigsSolver<double, Spectra::SMALLEST_MAGN,
				Spectra::SparseSymMatProd<double,Eigen::SelfAdjoint,SparseMatrixD::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor,SparseMatrixD::StorageIndex>,
				Spectra::SparseRegularInverse<double,Eigen::SelfAdjoint,SparseMatrixD::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor,SparseMatrixD::StorageIndex>, Spectra::GEIGS_REGULAR_INVERSE> 
				eigs( &K, &M, 12, K.cols() );
			/**/
			eigs.init();
			int n = eigs.compute();
			t1=omp_get_wtime()-t0;

			cout << endl << " ** " << t1 << "s for evs " << n << " " << eigs.eigenvalues().size();

			for(int i = 0; i<n && i<12; ++i){ // looks like the eigenvalues are sorted largest to smallest - so the last 6 are the rigid modes ... unless we have Dirichlet BCs applied
				cout << endl << "** ev " << i << ": " << eigs.eigenvalues()(i) << " -- f = sqrt(ev)/2pi = " << sqrt(abs(eigs.eigenvalues()(i)))/(2.0*M_PI); 
				Eigen::VectorXd u = eigs.eigenvectors().col(i); u = 0.08*u.normalized();
				fem.x += u;
				preview.render(); getchar();
				fem.x -= u;
			}
			return 0;
		}

		/** /
		vtkSmartPointer<PowerLawViscosityModel> viscMdl = vtkSmartPointer<PowerLawViscosityModel>::New();
		Eigen::VectorXd viscParam( viscMdl->getNumberOfParameters() ); viscParam.setZero();
		viscMdl->setViscosity( viscParam.data(), 0.005 ); viscMdl->setPowerLawH( viscParam.data(), 1.0 );
		//viscMdl->setViscosity( viscParam.data(), 0.001 ); viscMdl->setPowerLawH( viscParam.data(), 2.0 );
		//viscMdl->setViscosity( viscParam.data(), 0.02 ); viscMdl->setPowerLawH( viscParam.data(), 0.5 ); // low power law flow index < 1/3 (shear thinning) can be numerically problematic!
		fem.initializeViscosityModel(3, viscMdl,  viscParam.data() ); //cout << fem.elemViscParams << endl;
		/**/ // try not to combine this one with the PrincipalStretchMaterial elasticity -- strange things may happen
		vtkSmartPointer<EigenmodeViscosityModel> viscMdl = vtkSmartPointer<EigenmodeViscosityModel>::New();
		viscMdl->setNumberOfModes( 150 ); //ToDo: be very careful to initialize afterwards or stuff will break! Same for setting number of adjustment coefficients!
		viscMdl->initializeToConstantViscosity( 0.02, fem, 3 );
		// test adjustment ...
		Eigen::VectorXd frq, coeff;
		viscMdl->getAdjustmentFrequencies(frq); viscMdl->getViscosityCoefficients(coeff);
		for(int i=0; i<frq.size(); ++i){
			double fRatio = (frq.maxCoeff() - frq(i)) / (frq.maxCoeff()-frq.minCoeff());
			coeff(i) *= (1.0 + 2.0* fRatio*fRatio ) ; // example: triple the damping on the lowest frequency, keep highest unchanged, quadratic in between
			//coeff(i) *= (1.0-fRatio)*(1.0-fRatio); // example: remove damping on the lowest frequency, keep highest unchanged, quadratic in between
		}
		viscMdl->setViscosityCoefficients(coeff, fem);
		/** / 
		vtkSmartPointer<PowerSeriesViscosityModel> viscMdl = vtkSmartPointer<PowerSeriesViscosityModel>::New();
		viscMdl->powerIndices.resize(7); viscMdl->powerIndices.setLinSpaced(0.5,2.0);
		Eigen::VectorXd coeffs( viscMdl->getNumberOfParameters() ); coeffs.setLinSpaced(0.01,0.001);
		fem.initializeViscosityModel(3, viscMdl,  coeffs.data() );
		cout << endl << "% power series viscous damping: flow indices are " << viscMdl->powerIndices.transpose() << ", coeffs are " << coeffs.transpose();
		/**/

		fem.assembleMassAndExternalForce();
		fem.assembleForceAndStiffness();
		//printf("\n%% total mass = sum of element volumes * density   = %.4lg", fem.vol.sum()*fem.getMaterialParameters(0)(2) );
		//printf("\n%% total mass = sum of mass matrix / dofs per node = %.4lg", fem.M.sum()/3.0);

		fileName.str(""); fileName.clear();
		fileName << outDir << argv[1] << "_" << std::setfill ('0') << std::setw(5) << 0;
		bndFileName = fem.saveMeshToVTUFile(fileName.str(), true /*write boundary file - needed for restarting as it contains the rest coordinates*/);

		//cout << endl << "cpp_f = [" << endl << (fem.f + fem.f_ext) << endl << "];";
		//cout << endl << "cpp_K = [" << endl << (fem.K) << endl << "];";

		for(int step=0; step<100; ++step){
			printf("\n%% step = %5d ", step+1);
			fem.updateAllBoundaryData(); // but it's probably better to do it this way (fields will be evaluated with current sim time for dynamics)

			/** /
			fem.dynamicExplicitEulerSolve(0.0003);
			fem.assembleForceAndStiffness();
			/*/
			fem.dynamicImplicitTimestep(0.003); // takes care of force and stiffness assembly internally
			//Eigen::Vector3d fB; double areaB=0.0;
			//fem.computeTotalInternalForceOnBoundary(2,areaB, fB);
			//cout << " fB/area = [ " << fB.transpose()/areaB << " ]; "; //ToDo: check against analytical solution ...
			/**/

			fileName.str(""); fileName.clear();
			fileName << outDir << argv[1] << "_" << std::setfill ('0') << std::setw(5) << (step+1);
			fem.saveMeshToVTUFile(fileName.str());

			//if( step%10==0 && step>0 ){
			//	r.remesh(fem);
			//	printf("\n%% remeshed to %d elems", fem.getNumberOfElems());
			//	bndFileName = fem.saveMeshToVTUFile(fileName.str(), true /*write boundary file so we can re-start from the new mesh*/);
			//}

			// update preview
			preview.render();

		}	printf("\n");
	}

	if( 0 /*load the output files back and continue simulating*/ ){
		LinearFEM fem2;
		fem2.loadMeshFromVTUFile(fileName.str().append(".vtu"), bndFileName);

		// material and material parameters, mass, stiffness, and boundary conditions are NOT stored, need to re-build these ...
		fem2.setBoundaryCondition(1,bc);
		//fem2.setBoundaryCondition(2,bc2);
		fem2.initializeMaterialModel(3, LinearFEM::ISOTROPIC_NEOHOOKEAN,0.5,1.0,2.5);
		//fem2.setExternalAcceleration(3,g);
		fem2.assembleForceAndStiffness(); 
		fem2.assembleMassAndExternalForce();

		for(int step=100; step<500; ++step){
			bool saveBndFile=false;
			printf("\n%% step = %5d ", step+1);

			//fem2.setBoundaryCondition(2,bc2);
			/** /
			Eigen::VectorXd g=fem2.f+fem2.f_ext;
			for(int iter=0; iter<5 && fem2.freeDOFnorm(g)>1e-10 || iter==0; ++iter){
				fem2.staticSolveStep(10.0/(step));
				fem2.assembleForceAndStiffness(); // first assembly before the loop so we get post-solve internal forces in the output files
				g=fem2.f+fem2.f_ext;
				printf(" (%7.2lg)",fem2.freeDOFnorm(g));
			}
			/*/
			fem2.dynamicImplicitTimestep(0.00025); // takes care of force and stiffness assembly internally
			/**/

			fileName.str(""); fileName.clear();
			fileName << outDir << argv[1] << "_" << std::setfill ('0') << std::setw(5) << (step+1);
			fem2.saveMeshToVTUFile(fileName.str(), saveBndFile);
			
			if( step%10==0 && step>100 ){
				r.remesh(fem2); saveBndFile=true;
				printf("\n%% remeshed to %d elems", fem2.getNumberOfElems());
			}
		}	printf("\n");
	}
	
	return 0;
}
