
#include <filesystem>
#include <fstream>
#include <string>

#include "PointCloudObjectiveFunction.h"
#include "LinearFEM.h"

#include "vtkCellLocator.h"
#include "vtkGenericCell.h"
#include "vtkDoubleArray.h"
#include "vtkIdTypeArray.h"
#include "vtkPointData.h"
#include "vtkCellArray.h"
#include "vtkXMLUnstructuredGridReader.h"


using namespace MyFEM;

double PointCloudObjectiveFunction::evaluate(LinearFEM& theFEM,
	Eigen::VectorXd& phi_x, Eigen::VectorXd& phi_v, Eigen::VectorXd& phi_f, Eigen::VectorXd& phi_q
) {
	double totalSquaredDist = 0.0;
	// initialize storage
	closestPoints = vtkSmartPointer<vtkUnstructuredGrid>::New();
	closestPoints->SetPoints(vtkSmartPointer<vtkPoints>::New());
	vtkSmartPointer<vtkDoubleArray> closestPointBarycentricCoords = vtkSmartPointer<vtkDoubleArray>::New();
	vtkSmartPointer<vtkIdTypeArray> closestPointCells = vtkSmartPointer<vtkIdTypeArray>::New();
	vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
	closestPointBarycentricCoords->SetNumberOfComponents(3); closestPointBarycentricCoords->SetNumberOfTuples(pointCloud->GetNumberOfPoints());
	closestPointBarycentricCoords->SetName(barycentricName);
	closestPointCells->SetNumberOfComponents(1); closestPointCells->SetNumberOfTuples(pointCloud->GetNumberOfPoints());
	closestPointCells->SetName(closestCellName);

	// compute point cloud to surface distance ...
	vtkSmartPointer<vtkUnstructuredGrid> deformedSurface = vtkSmartPointer<vtkUnstructuredGrid>::New();
	deformedSurface->SetPoints(theFEM.mesh->GetPoints());
	deformedSurface->GetPoints()->Modified(); // super important to set this - otherwise stuff gets cached and we don't find the correct closest points!
	deformedSurface->SetCells(VTK_TRIANGLE, theFEM.meshBoundary->GetCells());		//deformedSurface->SetCells(VTK_TETRA, theFEM.mesh->GetCells());
	vtkSmartPointer<vtkCellLocator> locator = vtkSmartPointer<vtkCellLocator>::New();
	locator->SetDataSet(deformedSurface);
	locator->SetNumberOfCellsPerBucket(8); // not sure if this helps much, default should be 32, VTK's ICP implementation uses 1
	locator->ForceBuildLocator();
	vtkSmartPointer<vtkGenericCell> cell = vtkSmartPointer<vtkGenericCell>::New();
	vtkIdType cellId; int subId; // subId always 0 for non-composite cells
	double squaredDist;

	// for output
	vtkSmartPointer<vtkCellArray> lineCells;
	if (storeCPlines) {
		closestPointLines = vtkSmartPointer<vtkUnstructuredGrid>::New();
		closestPointLines->SetPoints(vtkSmartPointer<vtkPoints>::New());
		lineCells = vtkSmartPointer<vtkCellArray>::New();
	}

	//printf("\n pts = [");
	for (unsigned int i = 0; i < pointCloud->GetNumberOfPoints(); ++i) {
		Eigen::Map<Eigen::Vector3d> p(pointCloud->GetPoint(i));
		Eigen::Vector3d q;
		Eigen::Vector3d unusedV1, unusedV2, baryCoords; double unusedD;

		// find closest point to p on surface of theFEM ...
		locator->FindClosestPoint(p.data(), q.data(), cell, cellId, subId, squaredDist); //seems to work for triangle mesh (not good for tet mesh)
		cell->EvaluatePosition(q.data(), unusedV1.data(), subId, unusedV2.data(), unusedD, baryCoords.data());
		totalSquaredDist += squaredDist;

		// store stuff
		closestPoints->GetPoints()->InsertNextPoint(q.data());
		closestPointCells->SetValue(i, cellId);
		cells->InsertNextCell(1); cells->InsertCellPoint(i);
		closestPointBarycentricCoords->SetTuple(i, baryCoords.data());
		//printf("\n %7.3lg \t%7.3lg \t%7.3lg  -->  \t%7.3lg \t%7.3lg \t%7.3lg == d2( %.3lg ) %% sub = %d", p(0),p(1),p(2), q(0),q(1),q(2), squaredDist, subId);
		//printf("\n% bary( %8.3lg \t%8.3lg \t%8.3lg ) sum \t%.3lg, min \t%.3lg, max \t%.3lg", baryCoords(0), baryCoords(1), baryCoords(2), baryCoords.sum(), baryCoords.minCoeff(), baryCoords.maxCoeff() );
		if (storeCPlines) {
			closestPointLines->GetPoints()->InsertNextPoint(p.data());
			closestPointLines->GetPoints()->InsertNextPoint(q.data());
			lineCells->InsertNextCell(2); lineCells->InsertCellPoint(2 * i); lineCells->InsertCellPoint(2 * i + 1);
		}
		//printf("\n %7.3lg \t%7.3lg \t%7.3lg  \t%% %5u", p(0),p(1),p(2),i);
	}	//printf("\n];\n\n");
	closestPoints->GetPointData()->AddArray(closestPointBarycentricCoords);
	closestPoints->GetPointData()->AddArray(closestPointCells);
	closestPoints->SetCells(VTK_VERTEX, cells);
	if (storeCPlines) closestPointLines->SetCells(VTK_LINE, lineCells);

	if (0 /*check if cells + barycentric coords give correct closest point coords*/) {
		for (unsigned int i = 0; i < closestPoints->GetNumberOfPoints(); ++i) {
			Eigen::Map<Eigen::Vector3d> p(closestPoints->GetPoint(i));
			Eigen::Map<Eigen::Vector3d> b(closestPointBarycentricCoords->GetTuple(i));
			vtkIdType k = closestPointCells->GetValue(i);
			Eigen::Vector3d cp =
				b(0) * theFEM.getDeformedCoord(theFEM.getBoundaryElement(k)(0)) +
				b(1) * theFEM.getDeformedCoord(theFEM.getBoundaryElement(k)(1)) +
				b(2) * theFEM.getDeformedCoord(theFEM.getBoundaryElement(k)(2));

			if ((p - cp).squaredNorm() > (10 * DBL_EPSILON * p.squaredNorm())) printf("\n %4u: sq.cp.err %.4le", i, (p - cp).squaredNorm());
		}
	}

	// compute gradient d(phi)/dx ...
	phi_x.resize(theFEM.getNumberOfNodes() * LinearFEM::N_DOFS);
	phi_x.setZero();
	for (unsigned int i = 0; i < closestPoints->GetNumberOfPoints(); ++i) {
		Eigen::Map<Eigen::Vector3d> p(pointCloud->GetPoint(i));
		Eigen::Map<Eigen::Vector3d> q(closestPoints->GetPoint(i));
		Eigen::Map<Eigen::Vector3d> b(closestPointBarycentricCoords->GetTuple(i));
		vtkIdType k = closestPointCells->GetValue(i);
		if (k < theFEM.getNumberOfBndryElems()) for (unsigned int lnd = 0; lnd < 3; ++lnd) {
			for (unsigned int dof = 0; dof < LinearFEM::N_DOFS; ++dof) {
				phi_x(theFEM.getNodalDof(theFEM.getBoundaryElement(k)(lnd), dof)) +=
					(q(dof) - p(dof)) * b(lnd);
			}
		}
	}


	return 0.5* totalSquaredDist;
}



double TimeDependentPointCloudObjectiveFunction::evaluate(LinearFEM& theFEM,
	Eigen::VectorXd& phi_x, Eigen::VectorXd& phi_v, Eigen::VectorXd& phi_f, Eigen::VectorXd& phi_q
){
	double t = theFEM.simTime, t0,t1;
	unsigned int i0,i1;
	double phi0,phi1, w0,w1;
	Eigen::VectorXd phi_x1,phi_v1,phi_f1,phi_q1;

	i0 = getFrameBefore(t);
	i1 = getFrameAfter(t);

	if( i1==0 ) return 0.0; // no frame after this time found -> skip

	t0 = timeOfFrame[i0];
	t1 = timeOfFrame[i1];

	w1 = (t-t0)/(t1-t0); // 1 if t==t1, 0 if t==t0
	w0 = 1.0-w1;
	//printf("%% pcobj t0=%.3lg t1=%.3lg w0=%.3lg w1=%.3lg ", t0,t1, w0,w1);

	
	//printf("\n%% EVAL 0 \n");
	pointCloud = pointCloudFrames[i0];
	phi0 = PointCloudObjectiveFunction::evaluate(theFEM, phi_x, phi_v, phi_f, phi_q);

	//printf("\n%% EVAL 1 \n");
	pointCloud = pointCloudFrames[i1];
	phi1 = PointCloudObjectiveFunction::evaluate(theFEM,phi_x1,phi_v1,phi_f1,phi_q1);

	if( phi_x.size()>0 && phi_x.size()==phi_x1.size() ) phi_x += phi_x1;
	else if( phi_x.size()==0 && phi_x1.size()>0 ) phi_x = phi_x1;

	if( phi_v.size()>0 && phi_v.size()==phi_v1.size() ) phi_v += phi_v1;
	else if( phi_v.size()==0 && phi_v1.size()>0 ) phi_v = phi_v1;

	if( phi_f.size()>0 && phi_f.size()==phi_f1.size() ) phi_f += phi_f1;
	else if( phi_f.size()==0 && phi_f1.size()>0 ) phi_f = phi_f1;

	if( phi_q.size()>0 && phi_q.size()==phi_q1.size() ) phi_q += phi_q1;
	else if( phi_q.size()==0 && phi_q1.size()>0 ) phi_q = phi_q1;

	//printf(" phi=%.3lg, |phi_x|=%d , ||phi_x||=%.3lg", (w0*phi0 + w1*phi1), phi_x.size(), phi_x.norm() );
	return (w0*phi0 + w1*phi1);
}

void TimeDependentPointCloudObjectiveFunction::loadFromFileSeries(std::string fileNamePrefix, double timestep){
	//load files here -- should be numbered .vtu files, i.e. fileNamePrefix + "%d.vtu"
	//if fileNamePrefix is just a directory we'll look for files "points_%d.vtu" in there
	//there should also be a file giving time values for each frame called fileNamePrefix + "frametimes.txt"
	//otherwise we'll use time = file number * time step
	pointCloudFrames.clear();
	timeOfFrame.clear();

	unsigned int fnr;
	std::filesystem::path fspath(fileNamePrefix);
	std::string namePrefix("points_"), nameFormat;
	if(!std::filesystem::is_directory(fspath)){
		namePrefix = fspath.filename().string();
		fspath = fspath.parent_path().make_preferred(); // now fspath must be a directory (right?)
	}
	std::cout << std::endl << "% Time dependent point cloud objective function --- file path: " << fspath.string() << " prefix: " << namePrefix << std::endl;

	vtkSmartPointer<vtkXMLUnstructuredGridReader> reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();

	nameFormat = namePrefix; nameFormat.append("%u.vtu");
	if( std::filesystem::exists(fspath)) // check if directory exists
		for(std::filesystem::directory_iterator it(fspath); it!=std::filesystem::end(it); ++it) // iterate all files
			if( sscanf( it->path().filename().string().c_str(), nameFormat.c_str(), &fnr )==1 ){ // parse and check file name format
				reader->SetFileName( it->path().string().c_str() );
				reader->Update();
				pointCloudFrames[fnr] = vtkSmartPointer<vtkUnstructuredGrid>::New();
				pointCloudFrames[fnr]->DeepCopy( reader->GetOutput() );
				timeOfFrame[fnr] = (double)fnr * timestep;
				printf("\n%% read file %s for t=%.3lg, nPts=%d ", it->path().filename().string().c_str(), timeOfFrame[fnr], pointCloudFrames[fnr]->GetNumberOfPoints() );
			}

	std::filesystem::path frameTimeFile(fspath); frameTimeFile.append(namePrefix).concat("frametimes.txt");
	printf("\n%% frametimes: %s \n", (std::filesystem::exists( frameTimeFile ) ? frameTimeFile.string().c_str() : "MISSING") );
	if( std::filesystem::exists( frameTimeFile ) ){
		// read frametimes file
		ifstream ftfile(frameTimeFile);
		std::string line; int r; double t;
		while( ftfile.good() ){
			std::getline(ftfile, line);
			r = sscanf(line.c_str(), "%u %lf", &fnr, &t);
			if( r==2 && timeOfFrame.count(fnr)>0 ){
				timeOfFrame[fnr] = t;
				printf("\n%% frametime for %d is %.3lg", fnr, timeOfFrame[fnr]);
			}else{
				printf("\n%% frametime failed to parse line \"%s\" ", line.c_str() );
			}
		}
		ftfile.close();
	}
}

unsigned int TimeDependentPointCloudObjectiveFunction::getFrameBefore(double time){
	unsigned int i=0; double ti=-1.0;
	for(std::map<unsigned int, double>::iterator it=timeOfFrame.begin(); it!=timeOfFrame.end(); ++it){
		if( it->second <= time && ti <= it->second ){
			i = it->first; ti = it->second;
		}
	}
	return i;
}

unsigned int TimeDependentPointCloudObjectiveFunction::getFrameAfter (double time){
	unsigned int i=0; double ti=-1.0;
	for(std::map<unsigned int, double>::iterator it=timeOfFrame.begin(); it!=timeOfFrame.end(); ++it){
		if( it->second > time && (ti > it->second || ti < 0.0) ){
			i = it->first; ti = it->second;
		}
	}
	return i;
}
