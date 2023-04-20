
#include <sstream>
#include <fstream>

#include "LinearFEM.h"
#include "ElmerReader.h"
#include "DirectSensitivity.h"
#include "AdjointSensitivity.h"
#include "ElasticMaterialParameterHandler.h"
#include "TemporalInterpolationField.h"
#include "PointCloudObjectiveFunction.h"

#include "../LBFGSpp/LBFGS.h"

#include "vtkSmartPointer.h"
#include "vtkUnstructuredGrid.h"
#include "vtkXMLUnstructuredGridWriter.h"
#include "vtkXMLUnstructuredGridReader.h"


using namespace MyFEM;

int main_YuMiFoamBar(int argc, char* argv[]) {
	if (argc <= 1) return -1; // input parameter is a text file that describes the optimization problem
	Eigen::initParallel();

	std::stringstream fileName;
	std::string outFile = "../_out/";
	printf("\n\n%% -- YuMiFoamBar -- \n\n");


	// --- parse input file ---
	std::cout << "argv[1]" << argv[1] << std::endl;
	std::ifstream fileInput(argv[1]);
	// which boundary was gripper attached to?
	unsigned int gripBcID;
	fileInput >> gripBcID;
	// gripper's offsets
	double offsetGripperX, offsetGripperY, offsetGripperZ;
	fileInput >> offsetGripperX >> offsetGripperY >> offsetGripperZ;
	// extra orientation is needed in case gripper's boundary ID is not the default one
	double idOrientationAngleX, idOrientationAngleY, idOrientationAngleZ;
	fileInput >> idOrientationAngleX >> idOrientationAngleY >> idOrientationAngleZ;
	Eigen::AngleAxisd idOrientationX = Eigen::AngleAxisd(idOrientationAngleX, Eigen::Vector3d(1, 0, 0));
	Eigen::AngleAxisd idOrientationY = Eigen::AngleAxisd(idOrientationAngleY, Eigen::Vector3d(0, 1, 0));
	Eigen::AngleAxisd idOrientationZ = Eigen::AngleAxisd(idOrientationAngleZ, Eigen::Vector3d(0, 0, 1));
	// weight of the object
	double weight;
	fileInput >> weight;
	// first material parameters guess
	double lameLamda, lameMu;
	fileInput >> lameLamda >> lameMu;
	// get path where mesh files are stored
	std::string pathMeshFiles;
	fileInput >> pathMeshFiles;
	// get path of file with gripper's information
	std::string filenameGripperInfo;
	fileInput >> filenameGripperInfo;
	// get output folder
	std::string outputFolderName;
	fileInput >> outputFolderName;
	cout << outputFolderName;

	// read position and orientation of gripper
	std::ifstream fileGripperInfo(filenameGripperInfo);
	double x, y, z, w;
	fileGripperInfo >> x >> y >> z;
	Eigen::Vector3d gripperPosition = Eigen::Vector3d(x, y, z);
	fileGripperInfo >> w >> x >> y >> z;
	Eigen::Quaterniond gripperOrientation = Eigen::Quaterniond(w, x, y, z);
	// get name of file containing point cloud data from Kinect's scan
	std::string filenamePointCloud;
	fileGripperInfo >> filenamePointCloud;
	// --- parse input file done ---

#ifdef _WINDOWS
	// create output data directory (tree)
	std::string sc("for %f in (\""); sc.append(outFile).append(outputFolderName).append("/").append("\") do mkdir %~dpf"); std::replace(sc.begin(), sc.end(), '/', '\\');
	int sr = system(sc.c_str());
	outFile.append(outputFolderName).append("/").append("mesh");
#endif // _WINDOWS

	// gravity
	class GravityField : public VectorField {
	public: virtual void eval(Eigen::Vector3d& g, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		g = -9.81 * Eigen::Vector3d::UnitY();
	} } g; printf("\n%% Gravity set to -y axis ");

	// boundary condition for YuMi gripper
	PositionRotationInterpolationField bc; bc.GlobalWarningDisplayOff(); //PositionRotationInterpolationField is a subclass of vtkObject and can be used with vtkSmartPointer - we don't use that here, so silence all warnings
	bc.addPoint(-1., Eigen::Vector3d::Zero(), Eigen::Quaterniond(Eigen::AngleAxisd(0.0,Eigen::Vector3d::UnitX()))); // add a second point that is never used so the internal splines can be built correctly
	bc.addPoint(0.0, // for a static solution, only time t=0 will be used
		gripperPosition,
		gripperOrientation * idOrientationX * idOrientationY * idOrientationZ
	);
	bc.rc.x() = offsetGripperX;
	bc.rc.y() = offsetGripperY;
	bc.rc.z() = offsetGripperZ;
	bc.p_shift = -bc.rc;

	// define the FEM mesh, loads, and boundary conditions
	LinearFEM theFEM;
	theFEM.loadMeshFromElmerFiles(pathMeshFiles);
	unsigned int bodyID = theFEM.bodyId.minCoeff();
	if (theFEM.bodyId.maxCoeff() != bodyID) printf("\n%% WARNING: there are multiple bodies in the mesh, this will likely cause problems. ");
	theFEM.setExternalAcceleration(bodyID, g);
	theFEM.setBoundaryCondition(gripBcID, bc); printf("\n%% Gripper boundary ID is %u ", gripBcID);

	// apply material parameters
	double density = weight / theFEM.computeBodyVolume(bodyID);
	theFEM.initializeMaterialModel(bodyID, LinearFEM::ISOTROPIC_NEOHOOKEAN, lameLamda, lameMu, density);

	// in order to get a more stable initial guess for the sim, we set the deformed coordinates of the entire mesh to the rigid-body transform described by bc
	Eigen::Vector3d u;
	for (unsigned int i = 0; i < theFEM.getNumberOfNodes(); ++i) {
		bc.eval(u, theFEM.getRestCoord(i), theFEM.getDeformedCoord(i), 0.0);
		theFEM.getDeformedCoord(i) = u;
	}
	theFEM.setResetPoseFromCurrent();
	fileName.str(""); fileName.clear();
	fileName << outFile << "_restPose";
	theFEM.saveMeshToVTUFile(fileName.str(), true);


	// solve
	theFEM.assembleMassAndExternalForce();
	theFEM.updateAllBoundaryData();
	printf("\n%% Initial solve ... ");
	theFEM.staticSolvePreregularized(1e-2, 1e-2, 50 * LinearFEM::MAX_SOLVER_ITERS);
	theFEM.setResetPoseFromCurrent(); // for static optimization, it can help to reset to a decent static solution ...

	// write output
	fileName.str(""); fileName.clear();
	fileName << outFile << "_staticSolution";
	theFEM.saveMeshToVTUFile(fileName.str());


	// load point cloud from real-world data and optimize ...
	PointCloudObjectiveFunction thePhi;
	GlobalElasticMaterialParameterHandler elQ(bodyID);
	ParameterHandler nop(bodyID);
	CombinedParameterHandler theQ(elQ, nop); theQ.useLogOfParams = true;
	AdjointSensitivity theSensitivity(thePhi, theQ, theFEM);
	LBFGSpp::LBFGSParam<double> optimOptions = defaultLBFGSoptions<LBFGSpp::LBFGSParam<double> >();
	LBFGSpp::LBFGSSolver<double> solver(optimOptions);
	double phiVal;
	Eigen::VectorXd q(theQ.getNumberOfParams(theFEM));//, dPhi_dq(theQ.getNumberOfParams(theFEM));

	{	//read point cloud from file
		fileName.str(""); fileName.clear();
		fileName << filenamePointCloud;
		vtkSmartPointer<vtkXMLUnstructuredGridReader> reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
		reader->GlobalWarningDisplayOff(); // there seem to be some deprecation warnings in VTK 8.1 that should not affect the way we use the reader here, so silence them
		reader->SetFileName(fileName.str().c_str());
		reader->Update();
		thePhi.setPointCloud(reader->GetOutput()); // might need to deep-copy here - actually maybe not (ref counting?)

		printf("\n%% Read target point cloud from \"%s\", have %u points ", fileName.str().c_str(), thePhi.pointCloud->GetNumberOfPoints());
		// for convenience, write a copy of the point cloud data to the output dir
		fileName.str(""); fileName.clear();
		fileName << outFile << "_pointCloudTarget.vtu";
		vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
		writer->SetInputData(thePhi.pointCloud);
		writer->SetFileName(fileName.str().c_str());
		writer->Write();
	}
	theQ.getCurrentParams(q, theFEM);
	theSensitivity.setupStaticPreregularizer(1e-2, LinearFEM::MAX_SOLVER_ITERS, 1e-1);
	//theSensitivity.finiteDifferenceTest(q);
	solver.minimize(theSensitivity, q, phiVal);

	// write output
	printf("\n\n Closest point distance RMS: \t%8.3lg", sqrt(2.0 * phiVal / ((double)thePhi.pointCloud->GetNumberOfPoints())));
	printf("\n\n Params: "); cout << q.transpose();
	printf("\n\n Sim runs %u", theSensitivity.getEvalCounter());
	fileName.str(""); fileName.clear();
	fileName << outFile << "_optimResult";
	theFEM.saveMeshToVTUFile(fileName.str(), true);

	fileName.str(""); fileName.clear();
	fileName << outFile << "_params.txt";
	std::ofstream fileOutputParams(fileName.str().c_str());
	fileOutputParams << q.transpose();

	return 0;
}
