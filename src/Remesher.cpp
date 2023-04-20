
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
#include <vtkExtractEdges.h>
#include <vtkCellSizeFilter.h>
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
	targetEdgeLength=1.0;
	// we'll add stuff as needed here ...
	//dOpt[MMG3D_DPARAM_hausd]=1e-3;
}

void Remesher::setTargetEdgeLengthFromMesh(vtkSmartPointer<vtkUnstructuredGrid> sample){
	vtkSmartPointer<vtkExtractEdges> edg = vtkSmartPointer<vtkExtractEdges>::New();
	vtkSmartPointer<vtkCellSizeFilter> size = vtkSmartPointer<vtkCellSizeFilter>::New();
	edg->SetInputData(sample);
	size->ComputeAreaOff();
	size->ComputeVertexCountOff();
	size->ComputeVolumeOff();
	size->ComputeLengthOn();
	size->ComputeSumOn();
	size->SetInputConnection(edg->GetOutputPort());
	size->Update();
	double sumLength = ((vtkDoubleArray*)size->GetOutput()->GetFieldData()->GetArray("Length"))->GetValue(0);
	targetEdgeLength = sumLength / edg->GetOutput()->GetNumberOfCells();
	//printf("\n%% target edge length set to %.4lg (found %d edges in input mesh)", targetEdgeLength, edg->GetOutput()->GetNumberOfCells());
}

void Remesher::remesh(LinearFEM& fem){
	// remesh and interpolate data from the FEM object
	// ... remesh in world space
	// ... ... when simulating plasticity, do a static solve first and remesh the result rather than the current state of a dynamic sim (see Wicke et al. 2010)
	// ... interpolate all data (including rest space coordinates)
	// ... update mass and stiffness ... where to store external load functions (not done) and boundary conditions (done?) - probably in the FEM object?

	buildMMGmesh(fem);
	buildSizingFunction(fem);

	for(std::map<int,double>::iterator opit=dOpt.begin(); opit!=dOpt.end(); ++opit)
		MMG3D_Set_dparameter(mmgMesh,mmgSfcn,opit->first, opit->second);
	for(std::map<int,int>::iterator opit=iOpt.begin(); opit!=iOpt.end(); ++opit)
		MMG3D_Set_iparameter(mmgMesh,mmgSfcn,opit->first, opit->second);

	int err = MMG3D_mmg3dlib(mmgMesh,mmgSfcn);
	if( err == MMG5_STRONGFAILURE ){
		printf("BIG PROBLEM IN MMG3DLIB\n");
	}else
	if( err == MMG5_LOWFAILURE ){
		printf("SMALL-ish PROBLEM IN MMG3DLIB\n");
	}else{

		buildVTKmesh();
		buildBoundaryMesh();

		interpolateData(fem); // rest coordinates are stored in the boundary mesh object
	
		fem.mesh->Initialize();
		fem.mesh->ShallowCopy(newMesh);

		fem.meshBoundary->Initialize();
		fem.meshBoundary->ShallowCopy(newMeshBoundary);

		fem.updateDataMaps();
		fem.precomputeMeshData();
		fem.elemMatParams=newMatParams;
		fem.elemViscParams=newViscParams; //cout << endl << "% *** " << newViscParams.transpose() << endl;
		fem.updateAllBoundaryData();
		//force re-assembly of all matrices ... to be done by caller
		fem.getMassMatrix().resize(0,0);
		fem.getStiffnessMatrix().resize(0,0);
		fem.getDampingMatrix().resize(0,0);

		//ToDo: interpolate BDF2 fields along with other point data -- quick fix: reset here:
		fem.x_old.resize(0); fem.v_old.resize(0);
		// same for reset coords
		fem.x_reset.resize(0);
		printf("\n%% WARNING: important ToDo in Remesher::remesh ... ");


		//vtkSmartPointer<vtkXMLUnstructuredGridWriter> vtkWriter = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
		//vtkWriter->SetInputData(newMesh);
		//vtkWriter->SetFileName("_out/remeshed.vtu");
		//vtkWriter->Write();

		//vtkWriter->SetInputData(newMeshBoundary);
		//vtkWriter->SetFileName("_out/remeshed_bnd.vtu");
		//vtkWriter->Write();

		double tmp=targetEdgeLength;
		setTargetEdgeLengthFromMesh(fem.mesh);
		printf("\n%% remeshed mean edge length is %.2lg, target was %.2lg", targetEdgeLength,tmp);
		targetEdgeLength=tmp;
	}
	cleanup();
}

void Remesher::buildMMGmesh(LinearFEM& fem){
	unsigned int n=fem.getNumberOfNodes(), m=fem.getNumberOfElems(), l=fem.getNumberOfBndryElems();

	cleanup(); // just in case ...
	MMG3D_Init_mesh(MMG5_ARG_start, MMG5_ARG_ppMesh,&mmgMesh,MMG5_ARG_ppMet,&mmgSfcn, MMG5_ARG_end);
	MMG3D_Set_meshSize(mmgMesh,n,m,0,l,0,0);

	// MMG uses 1-based numbering for nodes and elems
	for(unsigned int i=0; i<n; ++i){
		double p[3];
		fem.meshBoundary->GetPoint(i,p); // read rest space positions into MMG mesh
		MMG3D_Set_vertex(mmgMesh, p[0], p[1], p[2], 0,i+1); // MMG demands 1-numbered nodes and elements!
	}

	for(unsigned int k=0; k<m; ++k){
		vtkIdType tmp, *e;
		fem.mesh->GetCellPoints(k,tmp,e);
		MMG3D_Set_tetrahedron(mmgMesh, e[0]+1, e[1]+1, e[2]+1, e[3]+1, fem.getBodyId(k) ,k+1); // MMG demands 1-numbered nodes and elements!
	}

	for(unsigned int k=0; k<l; ++k){
		vtkIdType tmp, *e;
		fem.meshBoundary->GetCellPoints(k,tmp,e);
		MMG3D_Set_triangle(mmgMesh, e[0]+1, e[1]+1, e[2]+1, fem.bndId(k),k+1); // MMG demands 1-numbered nodes and elements!
	}
}

void Remesher::buildSizingFunction(LinearFEM& fem){
	unsigned int n=fem.getNumberOfNodes(), m=fem.getNumberOfElems();
	//// let's try a uniform remeshing adapted to the deformation
	//MMG3D_Set_solSize(mmgMesh,mmgSfcn,MMG5_Vertex,n,MMG5_Scalar);
	//MMG3D_doSol(mmgMesh,mmgSfcn); // compute mean edge lengths at vertices of input mesh -> sizing function
	////printf("\n%% input sizes: ");
	////for(unsigned int i=0; i<n; ++i){
	////	printf("%.2lg ",mmgSfcn->m[i]);
	////}
	//// compute the determinant of the deformation gradient and interpolate to the nodes ~ avg. volume ratio around node
	//Eigen::VectorXd localDeformedVolume(n); localDeformedVolume.setZero();
	//Eigen::Matrix3d F; double totalDeformedVolume=0.0,totalRestVolume=0.0;
	//for(unsigned int k=0; k<m; ++k){
	//	unsigned int tmp, *e;
	//	fem.mesh->GetCellPoints(k,tmp,e);
	//	fem.computeDeformationGradient(F,k);
	//	totalRestVolume     += fem.getVolume(k);
	//	totalDeformedVolume += fem.getVolume(k) * F.determinant();
	//	for(int i=0; i<4; ++i) localDeformedVolume( e[i] ) += 1.0/4.0* fem.getVolume(k) * F.determinant();
	//}
	//printf("\n%% sum of local volumes %.2lg, rest volume %.2lg", localDeformedVolume.sum(), totalRestVolume);
	//// adjust the sizing function by the cube root of the volume ratio
	//for(unsigned int i=0; i<n; ++i){
	//	mmgSfcn->m[i] = 3.0*std::cbrt( (localDeformedVolume.mean()/localDeformedVolume(i)) *( totalRestVolume*totalRestVolume / ( targetNumElems * totalDeformedVolume)) );
	//}
	////printf("\n%% target sizes: ");
	////for(unsigned int i=0; i<n; ++i){
	////	printf("%.2lg ",mmgSfcn->m[i]);
	////}

	// let's try an anisotropic meshing

	//// with this we'd preserve the mean edge lengths (~ resolution) of the input mesh (isotropic, but stored in tensor form)
	//MMG3D_Set_solSize(mmgMesh,mmgSfcn,MMG5_Vertex,n,MMG5_Tensor);
	//MMG3D_doSol(mmgMesh,mmgSfcn); 
	//for(unsigned int i=0; i<n; ++i){
	//	double s00=0.0,s01=0.0,s02=0.0,s11=0.0,s12=0.0,s22=0.0;
	//	MMG3D_Get_tensorSol(mmgSfcn, &s00,&s01,&s02,&s11,&s12,&s22);
	//	printf("\n%% %d: ( %.2lg  %.2lg  %.2lg  %.2lg  %.2lg  %.2lg )",i, s00,s01,s02,s11,s12,s22 );
	//}
	// from the MMG forum: (https://forum.mmgtools.org/t/mesh-refinement-to-adapt-raster-field/156/7)
	// ... M = tQ D Q with D the diagonal matrix of the eigenvalues and Q the matrix of the associated eigenvectors. If I note d an eigenvalue and q the associated eigenvector, the matrix M asks to have edges of length 1/sqrt(d) in the direction q.
	// this thread talks about a 2d mesh, similarly, another thread makes the same statement about 3d (https://forum.mmgtools.org/t/anisotropic-refinement-example/83/2)
	// ... i also understand that if v0, v1, v2 are the main axis of the ellipsoid, and h0,h1,h2 are the sizes you whish to achieve, then the matrix you describe would correspond to 1/h0^2 * v0\otimes v0 + 1/h1^2 * v1\otimes v1 + 1/h2^2 * v2\otimes v2

	std::vector<Eigen::Matrix3d> nodeF; nodeF.assign( n, Eigen::Matrix3d::Zero() );
	Eigen::VectorXd localVolume(n); localVolume.setZero();
	Eigen::Matrix3d F;
	for(unsigned int k=0; k<m; ++k){
		vtkIdType tmp, *e;
		fem.mesh->GetCellPoints(k,tmp,e);
		fem.computeDeformationGradient(F,k);
		for(int i=0; i<4; ++i){
			localVolume( e[i] ) += 0.25*fem.getVolume(k); // integral of pw-linear shape functions at each node
			nodeF[ e[i] ] += 0.25*fem.getVolume(k)* F; // integral of pw-const deformation gradient tensor field
		}
	}

	Eigen::Matrix3d s; //printf("\n%% aniso metric ...");
	MMG3D_Set_solSize(mmgMesh,mmgSfcn,MMG5_Vertex,n,MMG5_Tensor);
	//MMG3D_doSol(mmgMesh,mmgSfcn);
	for(unsigned int i=0; i<n; ++i){
		nodeF[i] /= localVolume(i); // volume-weighted average of deformation around this node
		//MMG3D_Get_tensorSol(mmgSfcn, &s(0,0),&s(0,1),&s(0,2), &s(1,1),&s(1,2), &s(2,2));
		//s(1,0)=s(0,1); s(2,0)=s(0,2); s(2,1)=s(1,2);

		s.setIdentity(); s*=1.0/(targetEdgeLength*targetEdgeLength);
		s=(nodeF[i].transpose()*s*nodeF[i]).eval();

		MMG3D_Set_tensorSol(mmgSfcn, s(0,0),s(0,1),s(0,2), s(1,1),s(1,2), s(2,2) ,i+1);

		//printf("\n%% %d: ( %.2lg  %.2lg  %.2lg  %.2lg  %.2lg  %.2lg )",i,  s(0,0),s(0,1),s(0,2), s(1,1),s(1,2), s(2,2));
	}
	//printf("\n%% raw metric:\n");
	//for(int i=0; i<6*n; ++i) printf(" %.2lg ", mmgSfcn->m[i]);
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
	vtkSmartPointer<vtkPoints> oldDeformedCoords = fem.mesh->GetPoints();
	fem.mesh->SetPoints( fem.meshBoundary->GetPoints() ); // reset the old mesh to rest configuration before we try to interpolate stuff

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

	for(unsigned int i=0; i<newMesh->GetNumberOfPoints(); ++i){
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
				oldDeformedCoords->GetPoint(e->GetId(j),c);
				pm+=w[j]*cm;
			}
			newMeshBoundary->GetPoints()->SetPoint(i,p); // write the deformed coordinate to the boundary mesh for temporary storage - we'll switch them around later
		}else
			printf("\n%% FAILED TO INTERPOLATE DATA FOR NEW NODE %d", i);
	}

	// pw-constant interpolation of cell data and all material parameters
	newMesh->GetCellData()->CopyAllocate(fem.mesh->GetCellData());
	newMatParams.resize( fem.elemMatParams.rows(), newMesh->GetNumberOfCells() );
	newViscParams.resize( fem.elemViscParams.rows(), newMesh->GetNumberOfCells() );

	for(unsigned int k=0; k<newMesh->GetNumberOfCells(); ++k){
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
			for(unsigned int i=0; i<newViscParams.rows(); ++i)
				newViscParams(i,k) = fem.elemViscParams(i,id);
		}else
			printf("\n%% FAILED TO INTERPOLATE DATA FOR NEW ELEM %d", k);
		// overwrite body ID from the MMG "reference" field
		((vtkIntArray*) newMesh->GetCellData()->GetAbstractArray( fieldNames[BODY_NAME].c_str() ))->SetValue(k, mmgMesh->tetra[k+1].ref);
	}

	// write deformed coords to the new mesh and rest coords to the new boundary mesh (currently switched)
	vtkSmartPointer<vtkPoints> newDeformedCoords = newMeshBoundary->GetPoints();
	newMeshBoundary->SetPoints( newMesh->GetPoints() ); // these are the rest coords, we'll keep them in the boundary mesh
	newMesh->SetPoints( newDeformedCoords ); // these are the deformed coords we just interpolated
}

void Remesher::buildBoundaryMesh(){
	newMeshBoundary = vtkSmartPointer<vtkUnstructuredGrid>::New();

	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New(); points->SetDataTypeToDouble();
	points->DeepCopy( newMesh->GetPoints() ); // we always store all the rest coords in the boundary file, so we can re-start a simulation from a set of tet-mesh + boundary-mesh files
	newMeshBoundary->SetPoints( points );

	//ToDo: if the input mesh has multiple body IDs, MMG might produce interface triangles as well
	//      we'd need to make sure we don't mix up the "reference" fields for these with boundary IDs!
	vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
	cells->Allocate( mmgMesh->nt , 3 );

	vtkSmartPointer<vtkTriangle> tri = vtkSmartPointer<vtkTriangle>::New();
	vtkSmartPointer<vtkIntArray> newBndId = vtkSmartPointer<vtkIntArray>::New();
	newBndId->SetName( fieldNames[BOUNDARY_NAME].c_str() );
	newBndId->SetNumberOfComponents(1);
	newBndId->SetNumberOfTuples( mmgMesh->nt );
	vtkSmartPointer<vtkIntArray> newBndParent = vtkSmartPointer<vtkIntArray>::New();
	newBndParent->SetName( fieldNames[PARENT_NAME].c_str() );
	newBndParent->SetNumberOfComponents(1);
	newBndParent->SetNumberOfTuples( mmgMesh->nt );
	vtkSmartPointer<vtkDoubleArray> newBndArea = vtkSmartPointer<vtkDoubleArray>::New();
	newBndArea->SetName( fieldNames[AREA_NAME].c_str() ); // areas are not assigned until FEM object recomputes mesh data ...
	newBndArea->SetNumberOfComponents(1);
	newBndArea->SetNumberOfTuples( mmgMesh->nt );
	for(unsigned int k=0; k< mmgMesh->nt; ++k){
		tri->GetPointIds()->SetId(0, mmgMesh->tria[k+1].v[0]-1 ); // convert to 0-based node index // MMG uses 1-based numbering for nodes and elems
		tri->GetPointIds()->SetId(1, mmgMesh->tria[k+1].v[1]-1 );
		tri->GetPointIds()->SetId(2, mmgMesh->tria[k+1].v[2]-1 );
		cells->InsertNextCell( tri );

		newBndId->SetValue(k, mmgMesh->tria[k+1].ref );

		int parent,faceOfParent;
		MMG3D_Get_tetFromTria(mmgMesh, k+1, &parent, &faceOfParent);
		newBndParent->SetValue(k, parent-1 );
	}

	newMeshBoundary->SetCells(VTK_TRIANGLE, cells);
	newMeshBoundary->GetCellData()->AddArray(newBndId);
	newMeshBoundary->GetCellData()->AddArray(newBndParent);
	newMeshBoundary->GetCellData()->AddArray(newBndArea);
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
