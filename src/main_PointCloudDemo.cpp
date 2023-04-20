
#include <sstream>
#include <fstream>

#include "vtkXMLUnstructuredGridWriter.h"
#include "vtkXMLUnstructuredGridReader.h"
#include "vtkUnstructuredGrid.h"
#include "vtkSmartPointer.h"
#include "vtkCellLocator.h"
#include "vtkGenericCell.h"
#include "vtkDoubleArray.h"
#include "vtkIdTypeArray.h"
#include "vtkPointData.h"
#include "vtkCellArray.h"

#include "../LBFGSpp/LBFGS.h"

#include "LinearFEM.h"
#include "ElmerReader.h"
#include "DirectSensitivity.h"
#include "AdjointSensitivity.h"
#include "ElasticMaterialParameterHandler.h"
#include "TemporalInterpolationField.h"
#include "PointCloudObjectiveFunction.h"

using namespace MyFEM;

int main_PointCloudDemo(int argc, char* argv[]) {
	// run an elastostatic FEM sim with the given mesh and default material parameters
	// sample a random point cloud of the deformed surface
	// use this point cloud to build an objective function (closest point distance)
	// start from different parameters and minimize the objective function ...
	// =============================

	if (argc <= 1) return -1; // need at least the mesh file name as parameter
	Eigen::initParallel();
	std::stringstream fileName;
	std::string outFile = "../_out/";
	printf("\n\n%% -- PointCloudDemo -- \n\n");

#ifdef _WINDOWS
	std::string sc("for %f in (\""); sc.append(outFile).append(argv[1]).append("\") do mkdir %~dpf"); std::replace(sc.begin(), sc.end(), '/', '\\');
	int sr = system(sc.c_str());
	outFile.append(argv[1]);
#endif // _WINDOWS

	unsigned int bodyID = 5, diriID = 1; // for X-worm mesh ... should be configurable eventually ...

	class GravityField : public VectorField {
	public: virtual void eval(Eigen::Vector3d& g, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		g = -9.81 * Eigen::Vector3d::UnitY();
	}
	} g;
	class DirichletField : public VectorField {
	public: virtual void eval(Eigen::Vector3d& p, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
		p = x0;
	}
	} bc;

	vtkSmartPointer<vtkUnstructuredGrid> pointCloud = vtkSmartPointer<vtkUnstructuredGrid>::New();
	if (1 /*generate point cloud from base parameters*/) {
		double weight = 30e-3, lameLamda = 5e3, lameMu = 20e3;

		// define the FEM mesh, loads, and boundary conditions
		LinearFEM theFEM;
		theFEM.loadMeshFromElmerFiles(argv[1]);
		theFEM.setExternalAcceleration(bodyID, g);
		theFEM.setBoundaryCondition(diriID, bc);

		// set material parameters
		double density = weight / theFEM.computeBodyVolume(bodyID);
		theFEM.initializeMaterialModel(bodyID, LinearFEM::ISOTROPIC_NEOHOOKEAN, lameLamda, lameMu, density);

		// solve
		theFEM.assembleMassAndExternalForce();
		theFEM.updateAllBoundaryData();
		theFEM.staticSolve();

		// write output
		fileName.str(""); fileName.clear();
		fileName << outFile << "_staticGroundTruth";
		theFEM.saveMeshToVTUFile(fileName.str(), true);

		// create point samples
		pointCloud->SetPoints(vtkSmartPointer<vtkPoints>::New());
		vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
		srand(time(NULL));
		vtkIdType cellCount = 0;
		for (unsigned int k = 0; k < theFEM.getNumberOfBndryElems(); ++k) { //use this to sample all elements, version below takes only a random subset of the first quarter (which is the leg opposite the boundary condition on Jim's X-walker mesh)
		//for(unsigned int k=0; k<theFEM.getNumberOfBndryElems()/4; ++k) if( (((double)rand())/((double)RAND_MAX))<0.3 ){
			Eigen::Vector3d b; b.setConstant(1.0 / 3.0);
			b(0) = (((double)rand()) / ((double)RAND_MAX)); b(1) = (((double)rand()) / ((double)RAND_MAX));
			if ((b(0) + b(1)) > 1.0) { b(0) = 1.0 - b(0); b(1) = 1.0 - b(1); } b(2) = 1.0 - b(0) - b(1);
			Eigen::Vector3d p =
				b(0) * theFEM.getDeformedCoord(theFEM.getBoundaryElement(k)(0)) +
				b(1) * theFEM.getDeformedCoord(theFEM.getBoundaryElement(k)(1)) +
				b(2) * theFEM.getDeformedCoord(theFEM.getBoundaryElement(k)(2));
			pointCloud->GetPoints()->InsertNextPoint(p.data());
			cells->InsertNextCell(1); cells->InsertCellPoint(cellCount++);
		}
		pointCloud->SetCells(VTK_VERTEX, cells);

		// write point cloud output
		fileName.str(""); fileName.clear();
		fileName << outFile << "_pointCloud.vtu";
		vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
		writer->SetInputData(pointCloud);
		writer->SetFileName(fileName.str().c_str());
		writer->Write();
	}	// end: generate point cloud from base parameters
	else { //read from file
		fileName.str(""); fileName.clear();
		fileName << outFile << "_pointCloud.vtu";
		vtkSmartPointer<vtkXMLUnstructuredGridReader> reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
		reader->GlobalWarningDisplayOff(); // there seem to be some deprecation warnings in VTK 8.1 that should not affect the way we use the reader here, so silence them
		reader->SetFileName(fileName.str().c_str());
		reader->Update();
		pointCloud = reader->GetOutput();
	}

	if (1 /*measure distance from point cloud to surface*/) {
		double weight = 30e-3, lameLamda = 10e3, lameMu = 40e3;

		// define the FEM mesh, loads, and boundary conditions
		LinearFEM theFEM;
		theFEM.loadMeshFromElmerFiles(argv[1]);
		theFEM.setExternalAcceleration(bodyID, g);
		theFEM.setBoundaryCondition(diriID, bc);

		// set material parameters
		double density = weight / theFEM.computeBodyVolume(bodyID);
		theFEM.initializeMaterialModel(bodyID, LinearFEM::ISOTROPIC_NEOHOOKEAN, lameLamda, lameMu, density);

		// define the objective function
		PointCloudObjectiveFunction thePhi;
		thePhi.setPointCloud(pointCloud);

		// solve
		theFEM.assembleMassAndExternalForce();
		theFEM.updateAllBoundaryData();
		theFEM.staticSolve();

		// evaluate objective function
		Eigen::VectorXd phi_x, unused;
		double phiVal = thePhi.evaluate(theFEM, phi_x, unused, unused, unused);

		// write output
		fileName.str(""); fileName.clear();
		fileName << outFile << "_staticInitGuess";
		vtkSmartPointer<vtkDoubleArray> vtkPhiX = vtkSmartPointer<vtkDoubleArray>::New();
		vtkPhiX->SetName("dPhi_dx");
		vtkPhiX->SetNumberOfComponents(3);
		vtkPhiX->SetArray(phi_x.data(), phi_x.size(), 1 /*save==1 -> no delete on cleanup*/);
		int vtkPhiXidx = theFEM.mesh->GetPointData()->AddArray(vtkPhiX);
		theFEM.saveMeshToVTUFile(fileName.str(), true);
		theFEM.mesh->GetPointData()->RemoveArray(vtkPhiXidx);

		// write closest point output
		fileName.str(""); fileName.clear();
		fileName << outFile << "_closestPoints.vtu";
		vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
		writer->SetInputData(thePhi.closestPoints);
		writer->SetFileName(fileName.str().c_str());
		writer->Write();
		if (thePhi.storeCPlines) {
			fileName.str(""); fileName.clear();
			fileName << outFile << "_closestPointLines.vtu";
			writer->SetInputData(thePhi.closestPointLines);
			writer->SetFileName(fileName.str().c_str());
			writer->Write();
		}
		printf("\n\n Sum of squared distances to closest points: \t%8.3lg", 2.0 * phiVal);
	}

	if (1 /*optimize parameters from different initial guess*/) {
		double weight = 30e-3, lameLamda = 10e3, lameMu = 40e3;

		// define the FEM mesh, loads, and boundary conditions
		LinearFEM theFEM;
		theFEM.loadMeshFromElmerFiles(argv[1]);
		theFEM.setExternalAcceleration(bodyID, g);
		theFEM.setBoundaryCondition(diriID, bc);

		// set material parameters
		double density = weight / theFEM.computeBodyVolume(bodyID);
		theFEM.initializeMaterialModel(bodyID, LinearFEM::ISOTROPIC_NEOHOOKEAN, lameLamda, lameMu, density);

		// define the objective function
		PointCloudObjectiveFunction thePhi;
		thePhi.setPointCloud(pointCloud);

		// optimize stuff
		GlobalElasticMaterialParameterHandler theQ(bodyID);
		DirectSensitivity theSensitivity(thePhi, theQ, theFEM);
		double phiVal;
		Eigen::VectorXd q(theQ.getNumberOfParams(theFEM));//, dPhi_dq(theQ.getNumberOfParams(theFEM));
		theQ.getCurrentParams(q, theFEM);
		theSensitivity.finiteDifferenceTest(q);
		//// just testing...
		//phiVal = theSensitivity(q,dPhi_dq);
		//printf("\n\n Sum of squared distances to closest points: \t%8.3lg", 2.0*phiVal);
		//printf("\n\n Objective function gradient: \t%8.3lg \t%8.3lg", dPhi_dq(0),dPhi_dq(1));
		LBFGSpp::LBFGSParam<double> optimOptions = defaultLBFGSoptions<LBFGSpp::LBFGSParam<double> >();
		LBFGSpp::LBFGSSolver<double> solver(optimOptions);
		solver.minimize(theSensitivity, q, phiVal);
		printf("\n\n Closest point distance RMS: \t%8.3lg", sqrt(2.0 * phiVal / ((double)pointCloud->GetNumberOfPoints())));
		printf("\n\n Params: \t%.16lg \t%.16lg", q(0), q(1));
		printf("\n\n Sim runs %u", theSensitivity.getEvalCounter());

		// write output
		fileName.str(""); fileName.clear();
		fileName << outFile << "_optimResult";
		theFEM.saveMeshToVTUFile(fileName.str(), true);

		// write closest point output
		fileName.str(""); fileName.clear();
		fileName << outFile << "_optimClosestPoints.vtu";
		vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
		writer->SetInputData(thePhi.closestPoints);
		writer->SetFileName(fileName.str().c_str());
		writer->Write();
		if (thePhi.storeCPlines) {
			fileName.str(""); fileName.clear();
			fileName << outFile << "_optimClosestPointLines.vtu";
			writer->SetInputData(thePhi.closestPointLines);
			writer->SetFileName(fileName.str().c_str());
			writer->Write();
		}

	}

	return 0;
}
