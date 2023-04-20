
#include "ElmerReader.h"
#include "LinearFEM.h"
#include "Materials.h"

#include "AdjointSensitivity.h"
#include "DirectSensitivity.h"
#include "TemporalInterpolationField.h"
#include "BoundaryFieldObjectiveFunction.h"
#include "ElasticMaterialParameterHandler.h"
#include "ViscousMaterialParameterHandler.h"

#include "../LBFGSpp/LBFGS.h"
#include "../LBFGSpp/Newton.h"

using namespace MyFEM;

int main_TPUbar(int argc, char* argv[]){
	// args:
	//  [0]: bin file
	//  [1]: mesh file
	//  [2]: tracked target file -- skip param est. if ""
	//  [3]: sample weight -- default 9.3 g
	//  [4]: lameLambda -- default 10 kPa
	//  [5]: lameMu -- default 20 kPa
	//  [6]: viscNu -- default 12 Pas
	//  [7]: viscH  -- default 1
	//  [8]: framerate -- default 180 Hz

	bool doParamEst = true;
	bool useGaussNewton = false; // otherwise use LBFGS
	bool logParamMode = true; // only helps for LBFGS (very useful there though)
	std::string bndFileName, outFile="../_out/";
	std::stringstream fileName;

	double dt=1.0/180.0, t_end=0.5; unsigned int tsteps = (t_end+0.5*dt)/dt;
	double weight = 3e-3, // 3 gamms - just a guess for now
		lameLamda = 10e3, lameMu = 20e3, viscNu = 12.0, viscH= 1.0; // initial guess for most optimization runs

	{
		double tmp;
		if( argc>3 ) if( sscanf(argv[3], "%lg", &tmp)==1) weight=tmp;
		if( argc>4 ) if( sscanf(argv[4], "%lg", &tmp)==1) lameLamda=tmp;
		if( argc>5 ) if( sscanf(argv[5], "%lg", &tmp)==1) lameMu=tmp;
		if( argc>6 ) if( sscanf(argv[6], "%lg", &tmp)==1) viscNu=tmp;
		if( argc>7 ) if( sscanf(argv[7], "%lg", &tmp)==1) viscH=tmp;
		if( argc>8 ) if( sscanf(argv[8], "%lg", &tmp)==1) dt=1/tmp; 
		printf("\n%% initial weight %.2lg, lambda %.2lg, mu %.2lg, nu %.2lg, h %.2lg, fps %.2lg", weight, lameLamda, lameMu, viscNu, viscH, 1.0/dt);
	}

	if( argc>2 ){
		std::string s(argv[2]);
		size_t lastPathSep = s.find_last_of("/\\")+1;
		size_t lastExtSep = s.find_last_of(".");
		//printf("\n%% lastPathSep %d, lastExtSep %d, substr \"%s\"", lastPathSep,lastExtSep, s.substr(lastPathSep, lastExtSep-lastPathSep).c_str());
		outFile.append(s.substr(lastPathSep, lastExtSep-lastPathSep).c_str()).append("/");
	}else{
		outFile.append("TPUbar/");
	}
	{
		std::string s(argv[1]);
		size_t lastPathSep = s.find_last_of("/\\")+1;
		outFile.append(s.substr(lastPathSep));

#ifdef _WINDOWS
		std::string sc("for %f in (\""); sc.append(outFile).append("\") do mkdir %~dpf"); std::replace( sc.begin(), sc.end(), '/', '\\');
		int sr = system(sc.c_str());
#endif // _WINDOWS

	}

	printf("\n\n%% -- TPUbar -- \n\n");

	// the main body of the mesh to simulate
	unsigned int bodyID=4;


	// gravity (-y)
	class GravityField : public VectorField{ public: virtual void eval(Eigen::Vector3d& g, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		g = -9.81 *Eigen::Vector3d::UnitY();
	} } g;

	LinearFEM::FORCE_BALANCE_EPS=1e-8; // reduce default accuracy a bit ...
	LinearFEM theFEM;
	theFEM.useBDF2 = true; printf("\n%% Using BDF2 time integration "); // BDF2 works with both direct and adjoint sensitivity analysis - however accuracy of adjoint method is sometimes not great
	theFEM.loadMeshFromElmerFiles(argv[1]);

	std::vector<vtkSmartPointer<TemporalInterpolationField> > trackedTargets; std::vector<unsigned int> trackedIDs; // storage for target fields
	std::vector<vtkSmartPointer<TemporalInterpolationField> > markerData; std::vector<std::string> markerNames;
	std::vector<vtkSmartPointer<PositionRotationInterpolationField> > rbData; std::vector<std::string> rbNames; 
	if( argc>2 ){
		printf("\n%% Reading mocap data from \"%s\"", argv[2]);
		printf("\n%% ... expecting one rigid body and markers labelled \"Front:Left\" and \"Front:Right\"");

		loadMocapDataFromCSV(std::string(argv[2]), rbData,rbNames,markerData,markerNames, 1e-3 /*file units assumed in mm*/);

		// assign labelled data to mesh boundary IDs
		for(unsigned int i=0; i<markerNames.size(); ++i){
			if( markerNames[i].find("Front:Left")!=std::string::npos ){
				trackedIDs.push_back(2); printf("\n%% found marker data for Front:Left "); // check mesh.names for ID of boundary regions
				trackedTargets.push_back(markerData[i]);
			}else
			if( markerNames[i].find("Front:Right")!=std::string::npos ){
				trackedIDs.push_back(3); printf("\n%% found marker data for Front:Right "); // check mesh.names for ID of boundary regions
				trackedTargets.push_back(markerData[i]);
			}
		}

		{	// adjust the time-range of the simulation to cover the data
			double trange[2];
			rbData[0]->getRange(trange);
			for(int i=0; i<rbData.size(); ++i) rbData[i]->t_shift = trange[0];
			for(int i=0; i<trackedTargets.size(); ++i) trackedTargets[i]->t_shift = trange[0];
			if( 1 ){
				t_end=trange[1]-trange[0]; tsteps = (t_end+0.5*dt)/dt; // also adjust the simulated time accordingly
			}
			printf("\n%% Data time range (%.2lg %.2lg), simulated range (0 %.2lg), time step %.2lg (%u steps) ",trange[0],trange[1], t_end, dt, tsteps );
		}

		if( rbData.size()>=1 ){
			theFEM.setBoundaryCondition( 1, *(rbData[0]) ); // boundary ID 1 is the base face
			// in order to get a more stable initial guess for the sim, we set the deformed coordinates of the entire mesh to the rigid-body transform described by rbBndryField at t==0
			for(unsigned int i=0; i<theFEM.getNumberOfNodes(); ++i){
				Eigen::Vector3d u;
				rbData[0]->eval(u , theFEM.getRestCoord(i), theFEM.getDeformedCoord(i), 0.0);
				theFEM.getDeformedCoord(i) = u;
			}
			if( rbData.size()>1 ) printf("\n%% Warning: expected one rigid body as boundary condition in mocap data. Found %d. Using %s\n", rbData.size(), rbNames[0].c_str() );
		}else{
			printf("\n%% Warning: expected one rigid body as boundary condition in mocap data. Found %d.\n", rbData.size());
		}
	}else{ doParamEst = false; }

	// material model and body force
	double density = weight / theFEM.computeBodyVolume(); printf("\n%% density %.4lg ", density);
	theFEM.setExternalAcceleration(bodyID, g);

	/** /
	theFEM.initializeMaterialModel(bodyID, LinearFEM::ISOTROPIC_NEOHOOKEAN,lameLamda,lameMu,density);
	typedef GlobalElasticMaterialParameterHandler ElasticityMaterialParameterHandler;
	/*/
	{
		theFEM.initializeMaterialModel(bodyID, LinearFEM::ISOTROPIC_NEOHOOKEAN,lameLamda,lameMu,density);
		printf("\n%% pre-init ");
		theFEM.assembleMassAndExternalForce();
		theFEM.updateAllBoundaryData();
		theFEM.staticSolve();
	}

	vtkSmartPointer<PrincipalStretchMaterial> psMat = vtkSmartPointer<PrincipalStretchMaterial>::New();
	typedef GlobalPrincipalStretchMaterialParameterHandler ElasticityMaterialParameterHandler;
	logParamMode = false; // never try to use this with ps-material!
	Eigen::VectorXd psMatParams( psMat->getNumberOfParameters() );
	psMat->setElasticParams(psMatParams.data(), lameLamda, lameMu); psMat->setDensity(psMatParams.data(), density);
	theFEM.initializeMaterialModel(bodyID,psMat, psMatParams.data());
	/**/


	/** /
	vtkSmartPointer<PowerLawViscosityModel> viscMdl = vtkSmartPointer<PowerLawViscosityModel>::New(); Eigen::VectorXd viscParam( viscMdl->getNumberOfParameters() ); viscParam.setZero();
	typedef GlobalPLViscosityMaterialParameterHandler ViscosityMaterialParameterHandler;
	viscMdl->setViscosity( viscParam.data(), viscNu ); viscMdl->setPowerLawH( viscParam.data(), viscH );	
	theFEM.initializeViscosityModel(bodyID, viscMdl, viscParam.data() );
	/*/
	vtkSmartPointer<PowerSeriesViscosityModel> viscMdl = vtkSmartPointer<PowerSeriesViscosityModel>::New();
	typedef GlobalPSViscosityMaterialParameterHandler ViscosityMaterialParameterHandler;
	viscMdl->powerIndices.resize(1); viscMdl->powerIndices(0) = viscH;
	Eigen::VectorXd coeffs( viscMdl->getNumberOfParameters() ); coeffs.setOnes(); coeffs*=viscNu/(viscMdl->powerIndices.size());
	theFEM.initializeViscosityModel(bodyID, viscMdl,  coeffs.data() );
	cout << endl << "% power series viscous damping: flow indices are " << viscMdl->powerIndices.transpose() << ", coeffs are " << coeffs.transpose();
	/**/

	if( doParamEst /*run parameter optimization*/){
		theFEM.setResetPoseFromCurrent(); // always use the current (rigidly transformed pose) during resets
		AverageBoundaryValueObjectiveFunction thePhi;
		for(int i=0; i<trackedIDs.size(); ++i){ thePhi.addTargetField( trackedIDs[i] , trackedTargets[i] ); }
		// for debug: output the target displacements to VTU
		fileName.str(""); fileName.clear();
		fileName << outFile << "_targetBClocations";
		TemporalInterpolationField::writeFieldsToVTU(fileName.str(), 3*tsteps, trackedTargets, trackedIDs);
		fileName.str(""); fileName.clear();
		fileName << outFile << "_transformedRestPose";
		theFEM.saveMeshToVTUFile(fileName.str(),false);


		Eigen::VectorXd q, dphi;
		double phi = 0.0; int r = 0, ev = 0;
		LBFGSpp::LBFGSParam<double> optimOptions = defaultLBFGSoptions<LBFGSpp::LBFGSParam<double> >();
		optimOptions.epsilon = 1e-10;
		optimOptions.delta = 1e-8; // stop a bit earlier in term of objective function decrease (relative) per iteration
		ParameterHandler noOp(bodyID);

		ElasticityMaterialParameterHandler		elasticParamHandler(bodyID);
		ViscosityMaterialParameterHandler		viscousParamHandler(bodyID);
		GlobalDensityParameterHandler           densityParamHandler(bodyID);
		CombinedParameterHandler tmpQ(elasticParamHandler, densityParamHandler);
		CombinedParameterHandler theQ(tmpQ, viscousParamHandler );
		//CombinedParameterHandler theQ(elasticParamHandler, viscousParamHandler );

		if( logParamMode /*&& !useGaussNewton*/ ){ theQ.useLogOfParams=true; printf("\n%% Using log-params for optimization ...\n"); }

		if( useGaussNewton ){
			DirectSensitivity theSensitivity(thePhi, theQ, theFEM);
			LBFGSpp::NewtonSolver<double,Eigen::ColPivHouseholderQR<Eigen::MatrixXd> > solver(optimOptions); //minimize with Gauss-Newton
			printf("\n%% Optimizing with Gauss-Newton (direct sensitivity analysis) ");

			theSensitivity.setupDynamicSim(dt,tsteps, true , outFile );
			q.resize( theQ.getNumberOfParams(theFEM) );
			theQ.getCurrentParams(q, theFEM);
			r = solver.minimize(theSensitivity, q, phi);
		
			ev  += theSensitivity.getEvalCounter();
			phi  = theSensitivity.bestRunPhiValue;
			q    = theSensitivity.bestRunParameters;
			dphi = theSensitivity.bestRunGradient;

			theQ.setNewParams( theSensitivity.bestRunParameters, theFEM );
		}else{
			AdjointSensitivity theSensitivity(thePhi, theQ, theFEM);
			LBFGSpp::LBFGSSolver<double> solver(optimOptions); //minimize with LBFGS
			printf("\n%% Optimizing with LBFGS (adjoint sensitivity analysis) ");

			theSensitivity.setupDynamicSim(dt,tsteps, true , outFile );
			q.resize( theQ.getNumberOfParams(theFEM) );
			theQ.getCurrentParams(q, theFEM);
			r = solver.minimize(theSensitivity, q, phi);
		
			ev  += theSensitivity.getEvalCounter();
			phi  = theSensitivity.bestRunPhiValue;
			q    = theSensitivity.bestRunParameters;
			dphi = theSensitivity.bestRunGradient;

			theQ.setNewParams( theSensitivity.bestRunParameters, theFEM );
		}
		
		cout << endl << "% solver iterations " << r << ", fcn.evals " << ev;
		fileName.str(""); fileName.clear();
		fileName << outFile << "_optimResult.txt";
		ofstream optOut(fileName.str());
		optOut << "% solver iterations " << r << ", fcn.evals " << ev << ", objective function value " << phi << endl;
		optOut << endl << "% params:" << endl << q << endl;
		optOut << endl << "% gradient:" << endl << dphi << endl;
		optOut.close();

		theFEM.reset();
	}

	if( 1 /* Forward sim and output result */ ){

		printf("\n%% Forward sim - writing motion data ");
		id_set trackedBoundaries; // record average displacements for these boundary surface IDs
		trackedBoundaries.insert(2); trackedBoundaries.insert(3);
		fileName.str(""); fileName.clear();
		fileName << outFile << "_trackedBClocations.log";
		ofstream tbOut(fileName.str().c_str()); tbOut << std::setprecision(18);
		for(id_set::iterator tb=trackedBoundaries.begin(); tb!=trackedBoundaries.end(); ++tb){
			tbOut << *tb << " ";
		}	tbOut << endl;


		printf("\n%% init ");
		theFEM.assembleMassAndExternalForce();
		theFEM.updateAllBoundaryData();
		theFEM.staticSolve();
		// output tracked boundaries (avg. displacement)
		Eigen::Vector3d uB;
		tbOut << theFEM.simTime << " "; // should be 0.0 here
		for(id_set::iterator tb=trackedBoundaries.begin(); tb!=trackedBoundaries.end(); ++tb){
			theFEM.computeAverageDeformedCoordinateOfBoundary(*tb,uB);
			tbOut << uB[0] << " " << uB[1] << " " << uB[2] << " ";
		}	tbOut << endl;


		for(int step=0; step<tsteps; ++step){
			// write output file before step
			fileName.str(""); fileName.clear();
			fileName << outFile << "_" << std::setfill ('0') << std::setw(5) << step;
			theFEM.saveMeshToVTUFile(fileName.str());

			printf("\n%% step %5d/%d ", step+1,tsteps);
			theFEM.updateAllBoundaryData();
			theFEM.dynamicImplicitTimestep(dt);

			// output tracked boundaries (avg. displacement)
			tbOut << theFEM.simTime << " ";
			for(id_set::iterator tb=trackedBoundaries.begin(); tb!=trackedBoundaries.end(); ++tb){
				theFEM.computeAverageDeformedCoordinateOfBoundary(*tb,uB);
				tbOut << uB[0] << " " << uB[1] << " " << uB[2] << " ";
			}	tbOut << endl;
		}
		tbOut.close();

		// write last output file
		fileName.str(""); fileName.clear();
		fileName << outFile << "_" << std::setfill ('0') << std::setw(5) << tsteps;
		theFEM.saveMeshToVTUFile(fileName.str(), true);

		// readback tracked motion and save to VTU for visualization (ToDo: write first to TemporalInterpolationField objects and output directly from there - either text or VTU)
		fileName.str(""); fileName.clear();
		fileName << outFile << "_trackedBClocations.log";
		if( TemporalInterpolationField::buildFieldsFromTextFile( fileName.str(), trackedTargets, trackedIDs) < 0 ) return -1;
		fileName.str(""); fileName.clear();
		fileName << outFile << "_trackedBClocations";
		TemporalInterpolationField::writeFieldsToVTU(fileName.str(), 3*tsteps, trackedTargets, trackedIDs);
	}

	printf("\n\n%% DONE.\n\n");

	return 0;
}


int main_SiliconeHand(int argc, char* argv[]){
	// args:
	//  [0]: bin file
	//  [1]: mesh file
	//  [2]: tracked target file -- skip param est. if ""
	//  [3]: sample weight -- default 33.5 g
	//  [4]: lameLambda -- default 10 kPa
	//  [5]: lameMu -- default 20 kPa
	//  [6]: viscNu -- default 12 Pas
	//  [7]: viscH  -- default 1
	//  [8]: framerate -- default 120 Hz

	bool enablePreview=false;
	bool doParamEst = true;
	bool logParamMode = true; // only for LBFGS at the moment (very useful there though)
	bool velocityMode = false; // not super useful ...
	std::string bndFileName, outFile="../_out/";
	std::stringstream fileName;

	double dt=1.0/180.0, t_end=0.5; unsigned int tsteps = (t_end+0.5*dt)/dt;
	double weight = 110.1e-3, // sample weight
		lameLamda = 10e3, lameMu = 20e3, viscNu = 12.0, viscH= 1.0; // initial guess for most optimization runs
		//lameLamda = 34485.70, lameMu = 139107.18, viscNu = 6.49, viscH= 1.0; // result of take2_part2 at 120Hz BDF2 LBFGS(log) Newtonian visc.

	{
		double tmp;
		if( argc>3 ) if( sscanf(argv[3], "%lg", &tmp)==1) weight=tmp;
		if( argc>4 ) if( sscanf(argv[4], "%lg", &tmp)==1) lameLamda=tmp;
		if( argc>5 ) if( sscanf(argv[5], "%lg", &tmp)==1) lameMu=tmp;
		if( argc>6 ) if( sscanf(argv[6], "%lg", &tmp)==1) viscNu=tmp;
		if( argc>7 ) if( sscanf(argv[7], "%lg", &tmp)==1) viscH=tmp;
		if( argc>8 ) if( sscanf(argv[8], "%lg", &tmp)==1) dt=1.0/tmp;
		printf("\n%% initial weight %.2lg, lambda %.2lg, mu %.2lg, nu %.2lg, h %.2lg, fps %.2lg", weight, lameLamda, lameMu, viscNu, viscH, 1.0/dt);
	}

	if( argc>2 ){
		std::string s(argv[2]);
		size_t lastPathSep = s.find_last_of("/\\")+1;
		size_t lastExtSep = s.find_last_of(".");
		//printf("\n%% lastPathSep %d, lastExtSep %d, substr \"%s\"", lastPathSep,lastExtSep, s.substr(lastPathSep, lastExtSep-lastPathSep).c_str());
		outFile.append(s.substr(lastPathSep, lastExtSep-lastPathSep).c_str()).append("/");
	}
	{
		std::string s(argv[1]);
		size_t lastPathSep = s.find_last_of("/\\")+1;
		outFile.append(s.substr(lastPathSep));

#ifdef _WINDOWS
		std::string sc("for %f in (\""); sc.append(outFile).append("\") do mkdir %~dpf"); std::replace( sc.begin(), sc.end(), '/', '\\');
		int sr = system(sc.c_str());
#endif // _WINDOWS

	}
	std::vector<vtkSmartPointer<TemporalInterpolationField> > trackedTargets; std::vector<unsigned int> trackedIDs; // storage for target fields

	printf("\n\n%% -- SiliconeHand -- \n\n");

	printf("\n%% Reading mocap data from \"%s\"", argv[2]);
	printf("\n%% ... expecting one rigid body and markers labelled \"Finger:Index\", \"Finger:Middle\", \"Finger:Pinky\", \"Finger:Ring\", \"Finger:Thumb\"");
	std::vector<vtkSmartPointer<TemporalInterpolationField> > markerData; std::vector<std::string> markerNames;
	std::vector<vtkSmartPointer<PositionRotationInterpolationField> > rbData; std::vector<std::string> rbNames; 
	loadMocapDataFromCSV(std::string(argv[2]), rbData,rbNames,markerData,markerNames, 1e-3 /*file units assumed in mm*/);
	for(unsigned int i=0; i<markerNames.size(); ++i){
		if( markerNames[i].find("Finger:Index")!=std::string::npos ){
			trackedIDs.push_back(6);
			trackedTargets.push_back(markerData[i]);
		}else
		if( markerNames[i].find("Finger:Middle")!=std::string::npos ){
			trackedIDs.push_back(5);
			trackedTargets.push_back(markerData[i]);
		}
		if( markerNames[i].find("Finger:Pinky")!=std::string::npos ){
			trackedIDs.push_back(3);
			trackedTargets.push_back(markerData[i]);
		}else
		if( markerNames[i].find("Finger:Ring")!=std::string::npos ){
			trackedIDs.push_back(2);
			trackedTargets.push_back(markerData[i]);
		}
		if( markerNames[i].find("Finger:Thumb")!=std::string::npos ){
			trackedIDs.push_back(1);
			trackedTargets.push_back(markerData[i]);
		}
	}

	// gravity (-y)
	class GravityField : public VectorField{ public: virtual void eval(Eigen::Vector3d& g, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		g = -9.81 *Eigen::Vector3d::UnitY();
	} } g;

	LinearFEM::FORCE_BALANCE_EPS=1e-6; // reduce default accuracy a bit ...
	LinearFEM theFEM;
	theFEM.useBDF2 = true; printf("\n%% Using BDF2 time integration "); // BDF2 works with both direct and adjoint sensitivity analysis - however accuracy of adjoint method is sometimes not great
	theFEM.loadMeshFromElmerFiles(argv[1]);

	//// boundary condition from rigid body tracking data
	//PositionRotationInterpolationField rbBndryField;
	//// set centre of rotation due to offset between origin in mesh file and centre of rotation set in the tracking system (average of used markers?)
	//rbBndryField.rc = Eigen::Vector3d( -0.088 , 0.012917 , 0.0 ); // this offset is the centre of mass of all markers wrt. mesh coordinates (see XwormClamp.hdf)
	//rbBndryField.LoadDataFromCSV(  std::string(argv[1]).append("_TrackingData.csv") ); // can use second parameter for scaling of lenght units - make sure the file is in meters otherwise

	PositionRotationInterpolationField& rbBndryField = *(rbData[0]); // using the first RB-capture data as boundary condition
	
	//// set centre of rotation due to offset between origin in mesh file and centre of rotation set in the tracking system (average of used markers)
	//rbBndryField.rc = Eigen::Vector3d( -0.088 , 0.012917 , 0.0 ); // this offset is the centre of mass of all markers wrt. mesh coordinates (see XwormClamp.hdf)
	// updated geometry (Mar. 2019) is moved such that the marker COM is at the origin

	theFEM.setBoundaryCondition( 4, rbBndryField );

	{	// adjust the time-range of the simulation to cover the data
		double trange[2]; 
		rbBndryField.getRange(trange);
		rbBndryField.t_shift = trange[0];
		for(int i=0; i<trackedTargets.size(); ++i) trackedTargets[i]->t_shift = trange[0];
		if( 1 ){
			t_end=trange[1]-trange[0]; tsteps = (t_end+0.5*dt)/dt; // also adjust the simulated time accordingly
		}
		printf("\n%% Data time range (%.2lg %.2lg), simulated range (0 %.2lg), time step %.2lg (%u steps) ",trange[0],trange[1], t_end, dt, tsteps );
	}

	// in order to get a more stable initial guess for the sim, we set the deformed coordinates of the entire mesh to the rigid-body transform described by rbBndryField at t==0
	for(unsigned int i=0; i<theFEM.getNumberOfNodes(); ++i){
		Eigen::Vector3d u;
		rbBndryField.eval(u , theFEM.getRestCoord(i), theFEM.getDeformedCoord(i), 0.0);
		theFEM.getDeformedCoord(i) = u;
	}

	// material model and body force
	unsigned int bodyID=7;
	double density = weight / theFEM.computeBodyVolume(); printf("\n%% density %.4lg ", density); // sample of FlexFoam V weighs 33.5g
	theFEM.initializeMaterialModel(bodyID, LinearFEM::ISOTROPIC_NEOHOOKEAN,lameLamda,lameMu,density);
	theFEM.setExternalAcceleration(bodyID, g);

	vtkSmartPointer<PowerLawViscosityModel> viscMdl = vtkSmartPointer<PowerLawViscosityModel>::New(); Eigen::VectorXd viscParam( viscMdl->getNumberOfParameters() ); viscParam.setZero();
	typedef GlobalPLViscosityMaterialParameterHandler ViscosityMaterialParameterHandler;
	viscMdl->setViscosity( viscParam.data(), viscNu ); viscMdl->setPowerLawH( viscParam.data(), 1.0 );	
	theFEM.initializeViscosityModel(bodyID, viscMdl, viscParam.data() );

	if( doParamEst /*run parameter optimization*/){
		theFEM.setResetPoseFromCurrent(); // always use the current (rigidly transformed pose) during resets
		AverageBoundaryValueObjectiveFunction thePhi;
		if( velocityMode){
			thePhi.targetMode = AverageBoundaryValueObjectiveFunction::TARGET_VELOCITY;
			printf("\n%% Objective function measures tracked velocities ");
		}
		for(int i=0; i<trackedIDs.size(); ++i){
			if( velocityMode){
				trackedTargets[i]->evalMode = TemporalInterpolationField::EVAL_VELOCITY;
				trackedTargets[i]->setInitialVelocity( Eigen::Vector3d::Zero() );
			}

			thePhi.addTargetField( trackedIDs[i] , trackedTargets[i] );
		}
		// for debug: output the target displacements to VTU
		fileName.str(""); fileName.clear();
		fileName << outFile << "_targetBCs";
		TemporalInterpolationField::writeFieldsToVTU(fileName.str(), 3*tsteps, trackedTargets, trackedIDs);
		fileName.str(""); fileName.clear();
		fileName << outFile << "_transformedRestPose";
		theFEM.saveMeshToVTUFile(fileName.str(),true);


		double phi = 0.0; int r = 0, ev = 0, rInit = 0, evInit = 0;
		LBFGSpp::LBFGSParam<double> optimOptions = defaultLBFGSoptions<LBFGSpp::LBFGSParam<double> >();

		bool startWithElstatOpt=false;
		if( startWithElstatOpt /*elastostatic optimization first*/){
			GlobalElasticMaterialParameterHandler theQ(bodyID);
			Eigen::VectorXd q( theQ.getNumberOfParams(theFEM) );
			theQ.getCurrentParams(q, theFEM);
			DirectSensitivity theSensitivity(thePhi, theQ, theFEM);
			//LBFGSpp::NewtonSolver<double,Eigen::ColPivHouseholderQR<Eigen::MatrixXd> > solver(optimOptions);
			LBFGSpp::LBFGSSolver<double> solver(optimOptions);
			printf("\n%% Optimizing initial elastostatic configuration with LBFGS (direct sensitivity analysis) ");
			rInit = solver.minimize(theSensitivity,q,phi);
			evInit = theSensitivity.getEvalCounter();
			theQ.setNewParams( theSensitivity.bestRunParameters, theFEM );
			cout << endl << "% initial elastostatic solver iterations " << rInit << ", fcn.evals " << evInit << endl;
		}else
		if( 0 ){ // different approach: instead of optimizing elastostatics to match marker positions (which may be unreliable), change constant offset of tracked targets at the start of each sim to match the elastostatic pose - then optimize for the motion -> this approach should auto-correct for badly positioned markers to some extent
			thePhi.resetMode = AverageBoundaryValueObjectiveFunction::RESET_FIELD_OFFSETS;
			// seems to cause more problems than it solves -- maybe we should just measure the marker offsets better in the experimental setup ...
		}
		
#ifdef _USE_PER_ELEM_LA_MU_RHO
		GlobalElasticMaterialParameterHandler elasticParamHandler(bodyID); //PerElementElasticMaterialParameterHandler elasticParamHandler(bodyID); //
		ParameterHandler doNothing(bodyID);
		PerElementDensityMaterialParameterHandler densityParamHandler(bodyID, true /*regularize to maintain total mass*/,1e3);
		//CombinedParameterHandler theQ(densityParamHandler,elasticParamHandler);
		//printf("\n%% Optimizing per-element density and homogeneous elasticity. ");
		//printf("\n%% Optimizing per-element parameters: elasticity (lambda,mu) and density. ");
		//CombinedParameterHandler theQ(densityParamHandler,doNothing);
		//printf("\n%% Optimizing per-element density. ");

		ViscosityMaterialParameterHandler viscousParamHandler(bodyID);
		CombinedParameterHandler globalParamHandler(viscousParamHandler,elasticParamHandler);
		CombinedParameterHandler theQ(globalParamHandler,densityParamHandler);
		printf("\n%% Optimizing per-element density and homogeneous viscoelasticity. ");
		optimOptions.wolfe = 1.0-1e-3;
#undef _USE_GAUSS_NEWTON // this will never work with per-element parameters .. not enough memory and time
#else
		GlobalElasticMaterialParameterHandler elasticParamHandler(bodyID);
		ViscosityMaterialParameterHandler     viscousParamHandler(bodyID);
		CombinedParameterHandler theQ(elasticParamHandler, viscousParamHandler);
		printf("\n%% Optimizing globally homogeneous viscoelastic parameters. ");
#endif // _USE_PER_ELEM_LA_MU_RHO

		Eigen::VectorXd q( theQ.getNumberOfParams(theFEM) ), dphi;

		if( logParamMode ){ theQ.useLogOfParams=true; printf("\n%% Using log-params for optimization ...\n"); }
		theQ.getCurrentParams(q, theFEM);
		AdjointSensitivity theSensitivity(thePhi, theQ, theFEM);
		LBFGSpp::LBFGSSolver<double> solver(optimOptions); //minimize with LBFGS
		printf("\n%% Optimizing with LBFGS (adjoint sensitivity analysis) ");
		theSensitivity.setupDynamicSim(dt,tsteps, true , outFile );		
		r = solver.minimize(theSensitivity, q, phi);

		ev  += theSensitivity.getEvalCounter();
		phi  = theSensitivity.bestRunPhiValue;
		q    = theSensitivity.bestRunParameters;
		dphi = theSensitivity.bestRunGradient;
		cout << endl << "% solver iterations " << r << ", fcn.evals " << ev;
		fileName.str(""); fileName.clear();
		fileName << outFile << "_optimResult.txt";
		ofstream optOut(fileName.str());
		if( rInit>0 ) optOut << "% initial elastostatic solver iterations " << rInit << ", fcn.evals " << evInit << endl;
		optOut << "% solver iterations " << r << ", fcn.evals " << ev << ", objective function value " << phi << endl;
		optOut << endl << "% params:" << endl << q << endl;
		optOut << endl << "% gradient:" << endl << dphi << endl;
		optOut.close();
		theQ.setNewParams( theSensitivity.bestRunParameters, theFEM );
		
		theFEM.reset();
	}

	if( 1 /* Forward sim and output result */ ){

		printf("\n%% Forward sim - writing motion data ");
		id_set trackedBoundaries; // record average displacements for these boundary surface IDs
		trackedBoundaries.insert(1); trackedBoundaries.insert(2);
		trackedBoundaries.insert(3); trackedBoundaries.insert(4);
		fileName.str(""); fileName.clear();
		fileName << outFile << "_trackedBClocations.log";
		ofstream tbOut(fileName.str().c_str()); tbOut << std::setprecision(18);
		for(id_set::iterator tb=trackedBoundaries.begin(); tb!=trackedBoundaries.end(); ++tb){
			tbOut << *tb << " ";
		}	tbOut << endl;


		printf("\n%% init ");
		theFEM.assembleMassAndExternalForce();
		theFEM.updateAllBoundaryData();
		theFEM.staticSolve();
		// output tracked boundaries (avg. displacement)
		Eigen::Vector3d uB;
		tbOut << theFEM.simTime << " "; // should be 0.0 here
		for(id_set::iterator tb=trackedBoundaries.begin(); tb!=trackedBoundaries.end(); ++tb){
			theFEM.computeAverageDeformedCoordinateOfBoundary(*tb,uB);
			tbOut << uB[0] << " " << uB[1] << " " << uB[2] << " ";
		}	tbOut << endl;


		//theFEM.useBDF2=true; // not supported for adjoint method yet - works in forward sim
		for(int step=0; step<tsteps; ++step){
			// write output file before step
			fileName.str(""); fileName.clear();
			fileName << outFile << "_" << std::setfill ('0') << std::setw(5) << step;
			theFEM.saveMeshToVTUFile(fileName.str());

			printf("\n%% step %5d/%d ", step+1,tsteps);
			theFEM.updateAllBoundaryData();
			theFEM.dynamicImplicitTimestep(dt);

			// output tracked boundaries (avg. displacement)
			tbOut << theFEM.simTime << " ";
			for(id_set::iterator tb=trackedBoundaries.begin(); tb!=trackedBoundaries.end(); ++tb){
				theFEM.computeAverageDeformedCoordinateOfBoundary(*tb,uB);
				tbOut << uB[0] << " " << uB[1] << " " << uB[2] << " ";
			}	tbOut << endl;
		}
		tbOut.close();

		// write last output file
		fileName.str(""); fileName.clear();
		fileName << outFile << "_" << std::setfill ('0') << std::setw(5) << tsteps;
		theFEM.saveMeshToVTUFile(fileName.str(), true);

		// readback tracked motion and save to VTU for visualization (ToDo: write first to TemporalInterpolationField objects and output directly from there - either text or VTU)
		fileName.str(""); fileName.clear();
		fileName << outFile << "_trackedBClocations.log";
		if( TemporalInterpolationField::buildFieldsFromTextFile( fileName.str(), trackedTargets, trackedIDs) < 0 ) return -1;
		fileName.str(""); fileName.clear();
		fileName << outFile << "_trackedBClocations";
		TemporalInterpolationField::writeFieldsToVTU(fileName.str(), 3*tsteps, trackedTargets, trackedIDs);
	}

	printf("\n\n%% DONE.\n\n");

	return 0;
}
