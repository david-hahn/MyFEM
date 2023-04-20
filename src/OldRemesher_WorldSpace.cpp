
#include "Remesher.h"
#include "fieldNames.h"
using namespace MyFEM;

// MMG include
#include <mmg/libmmg.h>

// VTK includes
#include <vtkPoints.h>
#include <vtkTriangle.h>
#include <vtkTetra.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkDoubleArray.h>
#include <vtkCellArray.h>
#include <vtkCellTreeLocator.h>
//#include <vtkGeometryFilter.h>
//#include <vtkCellLocator.h>

// local helper function
vtkIdType findClosestCell(vtkSmartPointer<vtkAbstractCellLocator> locator, double *p/*[3]*/, double offset);

void Remesher::cleanup(){
	if( mmgMesh!=NULL ){
		MMG3D_Free_all(MMG5_ARG_start, MMG5_ARG_ppMesh,&mmgMesh,MMG5_ARG_ppMet,&mmgSfcn, MMG5_ARG_end);
		mmgMesh=NULL; mmgSfcn=NULL;
	}
}

void Remesher::setDefaultOptions(){
	dOpt.clear();
	iOpt.clear();
	iOpt[MMG3D_IPARAM_verbose]=0;
	// we'll add stuff as needed here ...
	//dOpt[MMG3D_DPARAM_hausd]=1e-3;
	//iOpt[MMG3D_IPARAM_optimLES]=1;
}

void Remesher::remesh(LinearFEM& fem){
	// remesh and interpolate data from the FEM object
	// ... remesh in world space
	// ... ... when simulating plasticity, do a static solve first and remesh the result rather than the current state of a dynamic sim (see Wicke et al. 2010)
	// ... interpolate all data (including rest space coordinates)
	// ... update mass and stiffness ... where to store external load functions (not done) and boundary conditions (done?) - probably in the FEM object?

	bool remeshRestSpace=false;
	if( remeshRestSpace ){
		vtkSmartPointer<vtkPoints> tmp = fem.mesh->GetPoints();
		fem.mesh->SetPoints( fem.meshBoundary->GetPoints() );
		fem.meshBoundary->SetPoints( tmp );
	}

	buildMMGmesh(fem);

	for(std::map<int,double>::iterator opit=dOpt.begin(); opit!=dOpt.end(); ++opit)
		MMG3D_Set_dparameter(mmgMesh,mmgSfcn,opit->first, opit->second);
	for(std::map<int,int>::iterator opit=iOpt.begin(); opit!=iOpt.end(); ++opit)
		MMG3D_Set_iparameter(mmgMesh,mmgSfcn,opit->first, opit->second);

	int err = MMG3D_mmg3dlib(mmgMesh,mmgSfcn);
	if( err == MMG5_STRONGFAILURE )
		printf("BIG PROBLEM IN MMG3DLIB\n");
	if( err == MMG5_LOWFAILURE )
		printf("SMALL-ish PROBLEM IN MMG3DLIB\n");

	buildVTKmesh();

	interpolateData(fem); // rest coordinates are stored in the boundary mesh object

	buildBoundaryMesh(fem.meshBoundary);

	fem.mesh->Initialize();
	fem.mesh->ShallowCopy(newMesh);

	fem.meshBoundary->Initialize();
	fem.meshBoundary->ShallowCopy(newMeshBoundary);

	if( remeshRestSpace ){
		vtkSmartPointer<vtkPoints> tmp = fem.mesh->GetPoints();
		fem.mesh->SetPoints( fem.meshBoundary->GetPoints() );
		fem.meshBoundary->SetPoints( tmp );
	}

	fem.updateDataMaps();
	fem.precomputeMeshData();
	fem.elemMatParams=newMatParams;
	fem.rebuildExternalForceAndBoundaryData();
	if( fem.K.size()>0 ) fem.assembleForceAndStiffness( LinearFEM::REBUILD ); // if there was a stiffness matrix, rebuild it for the new mesh

	//vtkSmartPointer<vtkXMLUnstructuredGridWriter> vtkWriter = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
	//vtkWriter->SetInputData(newMesh);
	//vtkWriter->SetFileName("_out/remeshed.vtu");
	//vtkWriter->Write();

	//vtkWriter->SetInputData(newMeshBoundary);
	//vtkWriter->SetFileName("_out/remeshed_bnd.vtu");
	//vtkWriter->Write();


}

void Remesher::buildMMGmesh(LinearFEM& fem){
	vtkSmartPointer<vtkUnstructuredGrid> inputMesh = fem.mesh;
	vtkIdType n=inputMesh->GetNumberOfPoints(), m=inputMesh->GetNumberOfCells(), l=fem.meshBoundary->GetNumberOfCells();

	cleanup(); // just in case ...
	MMG3D_Init_mesh(MMG5_ARG_start, MMG5_ARG_ppMesh,&mmgMesh,MMG5_ARG_ppMet,&mmgSfcn, MMG5_ARG_end);
	MMG3D_Set_meshSize(mmgMesh,n,m,0,l,0,0);

	// MMG uses 1-based numbering for nodes and elems
	for(vtkIdType i=0; i<n; ++i){
		double p[3];
		inputMesh->GetPoint(i,p);
		MMG3D_Set_vertex(mmgMesh, p[0], p[1], p[2], 0,i+1); // MMG demands 1-numbered nodes and elements!
	}

	for(vtkIdType k=0; k<m; ++k){
		vtkIdType tmp, *e;
		inputMesh->GetCellPoints(k,tmp,e);
		MMG3D_Set_tetrahedron(mmgMesh, e[0]+1, e[1]+1, e[2]+1, e[3]+1, fem.getBodyId(k) ,k+1); // MMG demands 1-numbered nodes and elements!
	}

	for(vtkIdType k=0; k<l; ++k){
		vtkIdType tmp, *e;
		fem.meshBoundary->GetCellPoints(k,tmp,e);
		MMG3D_Set_triangle(mmgMesh, e[0]+1, e[1]+1, e[2]+1, fem.bndId(k),k+1); // MMG demands 1-numbered nodes and elements!
	}

	MMG3D_Set_solSize(mmgMesh,mmgSfcn,MMG5_Vertex,n,MMG5_Scalar);
	MMG3D_doSol(mmgMesh,mmgSfcn); // compute mean edge lengths at vertices of input mesh -> sizing function

}

void Remesher::buildVTKmesh(){
	int n, m, l;
	MMG3D_Get_meshSize(mmgMesh,&n,&m,NULL,&l,NULL,NULL);
	newMesh = vtkSmartPointer<vtkUnstructuredGrid>::New();

	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New(); points->SetDataTypeToDouble();
	points->SetNumberOfPoints(n);

	vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
	cells->Allocate( cells->EstimateSize(m, 4) );

	//printf("\n%% mmgMesh: %d nodes, %d elems\n",n,m);
	for(int i=0; i<n; ++i){
		points->SetPoint(i, mmgMesh->point[i+1].c ); // convert to 0-based node index // MMG uses 1-based numbering for nodes and elems
	}
	newMesh->SetPoints(points);

	vtkSmartPointer<vtkTetra> tetra = vtkSmartPointer<vtkTetra>::New();
	for(int k=0; k<m; ++k){	
		tetra->GetPointIds()->SetId(0, mmgMesh->tetra[k+1].v[0]-1 ); // convert to 0-based node index // MMG uses 1-based numbering for nodes and elems
		tetra->GetPointIds()->SetId(1, mmgMesh->tetra[k+1].v[1]-1 );
		tetra->GetPointIds()->SetId(2, mmgMesh->tetra[k+1].v[2]-1 );
		tetra->GetPointIds()->SetId(3, mmgMesh->tetra[k+1].v[3]-1 );
		cells->InsertNextCell(tetra);
		//printf("\n%% new tet %d has ref %d",k,mmgMesh->tetra[k+1].ref); // = body ID
	}
	newMesh->SetCells(VTK_TETRA, cells);

	//for(int k=0; k<l; ++k){	
	//	printf("\n%% new tri %d has ref %d",k,mmgMesh->tria[k+1].ref); // = bndry ID
	//}
}

void Remesher::interpolateData(LinearFEM& fem){
	newRestCoords = vtkSmartPointer<vtkPoints>::New(); newRestCoords->SetDataTypeToDouble();
	newRestCoords->SetNumberOfPoints(newMesh->GetNumberOfPoints());

	//vtkSmartPointer<vtkCellLocator> locator = vtkSmartPointer<vtkCellLocator>::New();
	vtkSmartPointer<vtkCellTreeLocator> locator = vtkSmartPointer<vtkCellTreeLocator>::New();
	locator->SetDataSet(fem.mesh);
	locator->Update();

	// pw-linear interpolation of point data
	newMesh->GetPointData()->InterpolateAllocate(fem.mesh->GetPointData());

	vtkSmartPointer<vtkIdList> e = vtkSmartPointer<vtkIdList>::New();
	double p[3], c[3], b[3], w[4], d; vtkIdType id, unused;
	Eigen::Map<Eigen::Vector3d> pm(p),cm(c);

	double meshBBox[6];
	fem.mesh->GetBounds(meshBBox); // format xmin xmax ymin ymax zmin zmax
	double offset = std::cbrt( ((meshBBox[1]-meshBBox[0])*(meshBBox[3]-meshBBox[2])*(meshBBox[4]-meshBBox[5]))/fem.mesh->GetNumberOfCells() ); // cbrt( avg. cell volume in input mesh ) ~ avg. edge length in input mesh

	for(vtkIdType i=0; i<newMesh->GetNumberOfPoints(); ++i){
		newMesh->GetPoint(i,p);			
		//printf("\n%% point (%7.4lf %7.4lf %7.4lf) ",p[0],p[1],p[2]);

		//does not work reliably: locator->FindClosestPoint(p,c,id,(int&)unused,d); //id = locator->FindCell(p) ... sometimes returns -1 probably when p is outside of the input mesh, don't use it
		id = findClosestCell(locator, p, offset);
		if( id>=0 && id<fem.mesh->GetNumberOfCells() ){
			fem.mesh->GetCell(id)->EvaluatePosition(p,c,(int&)unused,b,d,w); // eval at p allows for extrapolation, at c evaluates on or inside of the cell
			//fem.mesh->GetCell(id)->EvaluatePosition(c,p,(int&)unused,b,d,w); // "parametric" (ie. barycentric?) coords in b, weights in w (assume pw-linear?), p==c should hold afterwards
			//printf(" --> (%7.4lf %7.4lf %7.4lf) dist %7.2lg weights (%7.4lf %7.4lf %7.4lf  %7.4lf) sum=%.4lf",p[0],p[1],p[2], d, w[0],w[1],w[2],w[3], w[0]+w[1]+w[2]+w[3]);

			fem.mesh->GetCellPoints(id,e);
			newMesh->GetPointData()->InterpolatePoint(fem.mesh->GetPointData(), i, e,w);

			// pw-linear interpolation of rest coords
			pm.setZero();
			for(int j=0; j<4; ++j){
				fem.meshBoundary->GetPoint(e->GetId(j),c);
				pm+=w[j]*cm;
			}
			newRestCoords->SetPoint(i,p);
		}else
			printf("\n%% FAILED TO INTERPOLATE DATA FOR NEW NODE %d", i);
	}

	// pw-constant interpolation of cell data and all material parameters
	newMesh->GetCellData()->CopyAllocate(fem.mesh->GetCellData());
	newMatParams.resize( fem.elemMatParams.rows(), newMesh->GetNumberOfCells() );

	for(vtkIdType k=0; k<newMesh->GetNumberOfCells(); ++k){
		newMesh->GetCellPoints(k,e);
		cm.setZero();
		for(int i=0; i<4; ++i){
			newMesh->GetPoint(e->GetId(i),p);
			cm+=pm;
		}
		cm*=0.25; // c == cm contains the cell centroid now
		//locator->FindClosestPoint(c,p,id,(int&)unused,d); // find the cell in the input mesh that contains the closest point to the centroid of the new mesh
		id = findClosestCell(locator, c, offset);
		if( id>=0 && id<fem.mesh->GetNumberOfCells() ){


			// get all cell data from the input mesh
			newMesh->GetCellData()->CopyData(fem.mesh->GetCellData(), id,k);

			// copy material parameters
			for(unsigned int i=0; i<newMatParams.rows(); ++i)
				newMatParams(i,k) = fem.elemMatParams(i,id);
		}else
			printf("\n%% FAILED TO INTERPOLATE DATA FOR NEW ELEM %d", k);
		// overwrite body ID from the MMG "reference" field
		((vtkIdTypeArray*) newMesh->GetCellData()->GetAbstractArray( fieldNames[BODY_NAME].c_str() ))->SetValue(k, mmgMesh->tetra[k+1].ref);
	}
}

vtkIdType findClosestCell(vtkSmartPointer<vtkAbstractCellLocator> locator, double *p/*[3]*/, double offset){
	double c[3], b[3], w[4], d; int unused;
	vtkIdType id = locator->FindCell(p);
	if( id<0 ){ // p is likely outside of the locator's data set ... try to find a cell within the bounding box p+-offset
		double bbox[6], offset=1e-3; //newMesh->GetPoints()->GetBounds();
		bbox[0]=p[0]-offset; bbox[1]=p[0]+offset; // x-min, x-max
		bbox[2]=p[1]-offset; bbox[3]=p[1]+offset; // y-min, y-max
		bbox[4]=p[2]-offset; bbox[5]=p[2]+offset; // z-min, z-max
		vtkSmartPointer<vtkIdList> candidates = vtkSmartPointer<vtkIdList>::New();

		locator->FindCellsWithinBounds(bbox,candidates);

		double minD=0.0; vtkIdType minId=-1;
		for(vtkIdType k=0; k<candidates->GetNumberOfIds(); ++k){
			//fem.mesh->GetCell(candidates->GetId(k))->EvaluatePosition(p,c,(int&)unused,b,d,w);
			locator->GetDataSet()->GetCell(candidates->GetId(k))->EvaluatePosition(p,c,unused,b,d,w);
			if( minId<0 || d<minD ){
				minD=d; minId=candidates->GetId(k);
			}
		}
		id=minId;
	}
	if( id<0 ){ // the locator totally failed ... let's brute force it ...
		double minD=0.0; vtkIdType minId=-1;
		for(vtkIdType k=0; k<locator->GetDataSet()->GetNumberOfCells(); ++k){
			locator->GetDataSet()->GetCell(k)->EvaluatePosition(p,c,unused,b,d,w);
			if( minId<0 || d<minD ){
				minD=d; minId=k;
			}
		}
		id=minId;
	}
	return id;
}

void Remesher::buildBoundaryMesh(vtkSmartPointer<vtkUnstructuredGrid> inputBoundary){
	newMeshBoundary = vtkSmartPointer<vtkUnstructuredGrid>::New();
	// old version extracting from volume mesh ... now we read directly from MMG
	//vtkSmartPointer<vtkGeometryFilter> extractor = vtkSmartPointer<vtkGeometryFilter>::New();
	//extractor->SetInputData(newMesh);
	//extractor->Update();

	newMeshBoundary->SetPoints( newRestCoords ); // we always store all the rest coords in the boundary file, so we can re-start a simulation from a set of tet-mesh + boundary-mesh files

	//ToDo: if the input mesh has multiple body IDs, MMG might produce interface triangles as well
	//      we'd need to make sure we don't mix up the "reference" fields for these with boundary IDs!
	vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
	cells->Allocate( mmgMesh->nt , 3 );

	vtkSmartPointer<vtkTriangle> tri = vtkSmartPointer<vtkTriangle>::New();
	vtkSmartPointer<vtkIdTypeArray> newBndId = vtkSmartPointer<vtkIdTypeArray>::New();
	newBndId->SetName( fieldNames[BOUNDARY_NAME].c_str() );
	newBndId->SetNumberOfComponents(1);
	newBndId->SetNumberOfTuples( mmgMesh->nt );
	for(vtkIdType k=0; k< mmgMesh->nt; ++k){
		tri->GetPointIds()->SetId(0, mmgMesh->tria[k+1].v[0]-1 ); // convert to 0-based node index // MMG uses 1-based numbering for nodes and elems
		tri->GetPointIds()->SetId(1, mmgMesh->tria[k+1].v[1]-1 );
		tri->GetPointIds()->SetId(2, mmgMesh->tria[k+1].v[2]-1 );
		cells->InsertNextCell( tri );

		newBndId->SetValue(k, mmgMesh->tria[k+1].ref );
	}
	newMeshBoundary->SetCells(VTK_TRIANGLE, cells);
	newMeshBoundary->GetCellData()->AddArray(newBndId);

	// old version ... now we only do boundary IDs from MMG
	//vtkSmartPointer<vtkCellLocator> locator = vtkSmartPointer<vtkCellLocator>::New();
	//locator->SetDataSet(inputBoundary);
	//locator->BuildLocator();
	//newMeshBoundary->GetCellData()->CopyAllocate(inputBoundary->GetCellData());
	//vtkSmartPointer<vtkIdList> e = vtkSmartPointer<vtkIdList>::New();
	//double p[3], c[3], b[3], d; vtkIdType id, unused;
	//Eigen::Map<Eigen::Vector3d> pm(p),cm(c);
	//for(vtkIdType k=0; k<newMeshBoundary->GetNumberOfCells(); ++k){
	//	newMeshBoundary->GetCellPoints(k,e);
	//	cm.setZero();
	//	for(int i=0; i<3; ++i){
	//		newMeshBoundary->GetPoint(e->GetId(i),p);
	//		cm+=pm;
	//	}
	//	cm/=3.0; // c == cm contains the cell centroid now
	//	locator->FindClosestPoint(c,p,id,(int&)unused,d); // find the cell in the input mesh that contains the closest point to the centroid of the new mesh
	//	
	//	newMeshBoundary->GetCellData()->CopyData(inputBoundary->GetCellData(), id,k);

	//}
	//newMeshBoundary->GetCellData()->RemoveArray( fieldNames[PARENT_NAME].c_str() ); // no point in copying over old parent indices - those tets are gone anyway
	////ToDo: currently we do not compute parent information on the boundary mesh (i.e. the cell ID of the tetrahedron in the new mesh that contains a given triangle in the extracted boundary)
	////      add this if needed ...
}