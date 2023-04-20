#include "LinearFEM.h"
#include "ElmerReader.h"
#include "Materials.h"
#include "fieldNames.h"

using namespace MyFEM;

#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkDoubleArray.h>
#include <vtkCellArray.h>

// static ...
unsigned int LinearFEM::MAX_SOLVER_ITERS  = 20;
double       LinearFEM::FORCE_BALANCE_EPS = 1e-10;

LinearFEM::LinearFEM() :
	elems(NULL,4,0),restCoords(NULL,3,0),deformedCoords(NULL,3,0),velocities(NULL,3,0),
	x(NULL,0,1),f(NULL,0,1),f_ext(NULL,0,1),v(NULL,0,1),
	vol(NULL,0,1), bodyId(NULL,0,1), area(NULL,0,1), bndElems(NULL,3,0), bndId(NULL,0,1), bndParent(NULL,0,1)
{
	simTime=0.0;

	mesh = vtkSmartPointer<vtkUnstructuredGrid>::New();
	meshBoundary = vtkSmartPointer<vtkUnstructuredGrid>::New();


	vtkWriter = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
	doPrintResiduals=true;
	doWriteDiriDataAfterSolve=false; // should not be neccessary if the solver converges properly (can make FD-checks fail for sensitivity analysis, as it is not implemented there)
	useBDF2=false;
}

int LinearFEM::loadMeshFromElmerFiles(std::string fileName){
	ElmerReader reader( fileName );
	int n = reader.readModel( mesh , meshBoundary );

	if( n>0 ){
		printf("\n%% read mesh files, have %d nodes, %d elems, %d boundary elems ", mesh->GetNumberOfPoints(), mesh->GetNumberOfCells(), meshBoundary->GetNumberOfCells());

		// let's see if we have a .names file and print it to stdout for convenience
		std::string line;
		ifstream in((fileName+".names").c_str());
		if( in.is_open() ) cout << endl << "% --- --- --- " << fileName << ".names --- --- --- " << endl;
		while (in.good()){
			getline(in, line);
			cout << "% " << line << endl;
		}
		if( in.is_open() ) in.close();

		initializeMeshStorage();
		precomputeMeshData();
		reset();
	}else{
		return -1;
	}
	return n;
}

void LinearFEM::loadMeshFromVTUFile(std::string fileName, std::string boundaryFileName){
	vtkSmartPointer<vtkXMLUnstructuredGridReader> reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
	reader->GlobalWarningDisplayOff(); // there seem to be some deprecation warnings in VTK 8.1 that should not affect the way we use the reader here, so silence them

	reader->SetFileName(fileName.c_str());
	reader->Update();
	mesh->Initialize();
	mesh->ShallowCopy(reader->GetOutput());

	reader->SetFileName(boundaryFileName.c_str());
	reader->Update();
	meshBoundary->Initialize();
	meshBoundary->ShallowCopy(reader->GetOutput());

	updateDataMaps(); // no need to initializeMeshStorage() because the vtu file should contain all the data of the mesh object already ...
	precomputeMeshData(); // this takes care of external data not stored in the mesh object
	reset();
}

std::string LinearFEM::saveMeshToVTUFile(std::string fileName, bool alsoWriteBoundaryFile){
	std::string bndFileName(fileName);

	vtkWriter->SetFileName(fileName.append(".vtu").c_str());
	vtkWriter->SetInputData(mesh);
	vtkWriter->Write();

	if( alsoWriteBoundaryFile ){
		vtkWriter->SetFileName(bndFileName.append("_bnd.vtu").c_str());
		vtkWriter->SetInputData(meshBoundary);
		vtkWriter->Write();
	}
	return bndFileName;
}

void LinearFEM::initializeMeshStorage(){
	// nodal quantities (vtkPointData)
	// - velocities
	// - internal forces
	// - external forces
	vtkSmartPointer<vtkDoubleArray> data;

	data = vtkSmartPointer<vtkDoubleArray>::New();
	data->SetNumberOfComponents(3); data->SetNumberOfTuples( getNumberOfNodes() );
	data->SetName( fieldNames[VELOCITY_NAME].c_str() );
	mesh->GetPointData()->AddArray(data); // store the (smart) pointer in the mesh)

	data = vtkSmartPointer<vtkDoubleArray>::New();
	data->SetNumberOfComponents(3); data->SetNumberOfTuples( getNumberOfNodes() );
	data->SetName( fieldNames[FORCE_NAME].c_str() );
	mesh->GetPointData()->AddArray(data); // store the (smart) pointer in the mesh)

	data = vtkSmartPointer<vtkDoubleArray>::New();
	data->SetNumberOfComponents(3); data->SetNumberOfTuples( getNumberOfNodes() );
	data->SetName( fieldNames[FORCE_EXT_NAME].c_str());
	mesh->GetPointData()->AddArray(data); // store the (smart) pointer in the mesh)

	// per-element quantities
	// - rest volume

	data = vtkSmartPointer<vtkDoubleArray>::New();
	data->SetNumberOfComponents(1); data->SetNumberOfTuples( getNumberOfElems() );
	data->SetName( fieldNames[VOLUME_NAME].c_str());
	mesh->GetCellData()->AddArray(data); // store the (smart) pointer in the mesh)

	data = vtkSmartPointer<vtkDoubleArray>::New();
	data->SetNumberOfComponents(1); data->SetNumberOfTuples( getNumberOfBndryElems() );
	data->SetName( fieldNames[AREA_NAME].c_str());
	meshBoundary->GetCellData()->AddArray(data); // store the (smart) pointer in the mesh)

	updateDataMaps(); // sets v, f, etc. to map into the newly created fields on the mesh
	v.setZero();
	f.setZero();
	f_ext.setZero();
	vol.setZero();
}

void LinearFEM::precomputeMeshData(){
	elemBasisInv.resize( 9, getNumberOfElems() );
	elemDFdx.resize( 9*12 , getNumberOfElems() );

	for(unsigned int k=0; k<getNumberOfElems(); ++k){
		Tetra ek = getElement(k);
		Vect3 x0 = getRestCoord(ek[0]);
		Vect3 x1 = getRestCoord(ek[1]);
		Vect3 x2 = getRestCoord(ek[2]);
		Vect3 x3 = getRestCoord(ek[3]);

		// compute inverse basis matrix
		Eigen::Map<Eigen::Matrix3d> Dm = getBasisInv(k);
		Dm.col(0) = x1-x0;
		Dm.col(1) = x2-x0;
		Dm.col(2) = x3-x0;
		// compute element volume (Dm columns are edge-vectors here)
		getVolume(k) = 1.0/6.0*Dm.col(0).dot(Dm.col(1).cross(Dm.col(2)));
		if( getVolume(k) < 0.0 ) getVolume(k)=0.0; // disables elements that have an inverted rest configuration
		// now invert the matrix
		Dm = Dm.inverse().eval();

		// compute dF/dx (deriv. of deformation gradient wrt. node positions)
		Eigen::Map<Eigen::Matrix<double,9,12> > dF = getDFdx(k);
		{	// auto-generated code, writes "dF", reads x0,x1,x2,x3
			dF.setZero(); // only update non-zero components below
			#include "codegen_linTet_dFdx.h"
		}

	}

	for(unsigned int k=0; k<getNumberOfBndryElems(); ++k){
		getArea(k) = 1.0/2.0*(getRestCoord( getBoundaryElement(k)(1) )-getRestCoord( getBoundaryElement(k)(0) )).cross(getRestCoord( getBoundaryElement(k)(2) )-getRestCoord( getBoundaryElement(k)(0) )).norm();
	}
	//printf("\n%% sum of precomp areas %.4lg", area.sum()); //cout << endl << "precompAreas=[" << endl << area << endl << "];" << endl;
}

void LinearFEM::initializeMaterialModel(unsigned int bodyId, MAT_TYPE matType, double lameLambda, double lameMu, double density){
	Eigen::VectorXd params;
	vtkSmartPointer<HomogeneousMaterial> elastModel;
	if( matType == ISOTROPIC_LINEAR_ELASTIC ){
		elastModel = vtkSmartPointer<HomogeneousIsotropicLinearElasticMaterial>::New();
	}else
	if( matType == ISOTROPIC_NEOHOOKEAN ){
		elastModel = vtkSmartPointer<HomogeneousIsotropicNeohookeanMaterial>::New();
	}

	params.resize( elastModel->getNumberOfParameters() );
	elastModel->setElasticParams(params.data(), lameLambda, lameMu);
	elastModel->setDensity( params.data(), density);

	initializeMaterialModel(bodyId, elastModel, params.data());
}

void LinearFEM::initializeMaterialModel( unsigned int bodyId, vtkSmartPointer<HomogeneousMaterial> elastModel, double* defaultParams){
	materialModel[bodyId] = elastModel;
	Eigen::VectorXd params( materialModel[bodyId]->getNumberOfParameters() );
	if( defaultParams==NULL ){
		params.setZero();
	}else{
		params = Eigen::Map<Eigen::VectorXd>(defaultParams, materialModel[bodyId]->getNumberOfParameters());
	}

	unsigned int maxNrParams=0;
	for(std::map<unsigned int,vtkSmartPointer<HomogeneousMaterial> >::iterator it=materialModel.begin(); it!=materialModel.end(); ++it)
		maxNrParams = std::max( it->second->getNumberOfParameters(), maxNrParams);

	elemMatParams.conservativeResize( maxNrParams, getNumberOfElems() );
	for(unsigned int k=0; k<getNumberOfElems(); ++k) if( bodyId==getBodyId(k)) {
		elemMatParams.col(k) = params;
	}
}

void LinearFEM::initializeViscosityModel(unsigned int bodyId, vtkSmartPointer<ViscosityModel> viscModel, double* defaultParams){
	viscosityModel[bodyId] = viscModel;
	Eigen::VectorXd params( viscosityModel[bodyId]->getNumberOfParameters() );
	if( defaultParams==NULL ){
		params.setZero();
	}else{
		params = Eigen::Map<Eigen::VectorXd>(defaultParams, viscosityModel[bodyId]->getNumberOfParameters());
	}

	unsigned int maxNrParams=0;
	for(std::map<unsigned int,vtkSmartPointer<ViscosityModel> >::iterator it=viscosityModel.begin(); it!=viscosityModel.end(); ++it)
		maxNrParams = std::max( it->second->getNumberOfParameters(), maxNrParams);

	elemViscParams.conservativeResize( maxNrParams, getNumberOfElems() );
	for(unsigned int k=0; k<getNumberOfElems(); ++k) if( bodyId==getBodyId(k)) {
		elemViscParams.col(k) = params;
	}
}

void LinearFEM::computeDeformationGradient(Eigen::Matrix3d& F, unsigned int k){
	// compute deformation gradient F = Dw*Dm^-1 for element k
	Tetra ek = getElement(k);
	Vect3 v0 = getDeformedCoord(ek[0]);
	// compute inverse basis matrix
	Eigen::Map<Eigen::Matrix3d> DmInv = getBasisInv(k);
	F.col(0) = getDeformedCoord(ek[1])-v0;
	F.col(1) = getDeformedCoord(ek[2])-v0;
	F.col(2) = getDeformedCoord(ek[3])-v0;
	F *= DmInv;
}

void LinearFEM::computeVelocityGradient(Eigen::Matrix3d& dv, unsigned int k, VectXMap& vi){
	Tetra ek = getElement(k);
	Eigen::Block<VectXMap,3,1> v0 = vi.block<3,1>( 3*ek[0],0 );
	// compute inverse basis matrix
	Eigen::Map<Eigen::Matrix3d> DmInv = getBasisInv(k);
	dv.col(0) = vi.block<3,1>( 3*ek[1],0 )-v0;
	dv.col(1) = vi.block<3,1>( 3*ek[2],0 )-v0;
	dv.col(2) = vi.block<3,1>( 3*ek[3],0 )-v0;
	dv *= DmInv;
}

void LinearFEM::setBoundaryCondition(unsigned int boundaryId, const VectorField& g, BC_TYPE type, DOF_MASK dofs){
	bndCnd[boundaryId].data = &g;
	bndCnd[boundaryId].type = type;
	bndCnd[boundaryId].dofs = dofs;

	updateBoundaryData(boundaryId);
}

void LinearFEM::updateBoundaryData(unsigned int bcId){
	Eigen::Vector3d gx;
	if( bndCnd.count(bcId)==0 ) return; // invalid ID given
	if( bndCnd[bcId].type == BC_DIRICHLET ){
		for(unsigned int k = 0; k < getNumberOfBndryElems(); ++k){
			//printf("\n%% %d %d %d (id %d) %s ", getBoundaryElement(k)(0),getBoundaryElement(k)(1),getBoundaryElement(k)(2), bndId(k), bndId(k)==boundaryId?"***":"");

			if( bndId(k)==bcId ){
				for(int i=0; i<3; ++i){ // for each node of triangle k
					bndCnd[bcId].data->eval( gx, getRestCoord( getBoundaryElement(k)(i) ), getDeformedCoord(getBoundaryElement(k)(i)) ,simTime);

					if( bndCnd[bcId].dofs & X_MASK ) diriData[ getNodalDof(getBoundaryElement(k)(i), X_DOF) ] = gx[0];
					if( bndCnd[bcId].dofs & Y_MASK ) diriData[ getNodalDof(getBoundaryElement(k)(i), Y_DOF) ] = gx[1];
					if( bndCnd[bcId].dofs & Z_MASK ) diriData[ getNodalDof(getBoundaryElement(k)(i), Z_DOF) ] = gx[2];
				}
			}
		}
	
	}else
	if( bndCnd[bcId].type == BC_NEUMANN ){
		// Neumann BCs ...
		std::set<unsigned int> initializedDofs;
		// f+= int( g(x)*phi_j ) dx = (assm. pw-const g) ... area(tri_k)*g(x_c)*1/3
		for(unsigned int k = 0; k < getNumberOfBndryElems(); ++k){
			if( bndId(k)==bcId ){
				Eigen::Vector3d xc,xc0;
				double areaOver3 = 1.0/6.0*(getRestCoord( getBoundaryElement(k)(1) )-getRestCoord( getBoundaryElement(k)(0) )).cross(getRestCoord( getBoundaryElement(k)(2) )-getRestCoord( getBoundaryElement(k)(0) )).norm();
				xc0 = ( getRestCoord( getBoundaryElement(k)(0) )+getRestCoord( getBoundaryElement(k)(1) )+getRestCoord( getBoundaryElement(k)(2) ) )/3.0;
				xc  = ( getDeformedCoord( getBoundaryElement(k)(0) )+getDeformedCoord( getBoundaryElement(k)(1) )+getDeformedCoord( getBoundaryElement(k)(2) ) )/3.0;
				bndCnd[bcId].data->eval( gx, xc0, xc ,simTime);

				for(int i=0; i<3; ++i){ // for each node of triangle k
					//ToDo: not sure DOF-masking makes much sense here ... maybe simplify
					if( neumData.count( getNodalDof(getBoundaryElement(k)(i), X_DOF) )==0 || initializedDofs.count( getNodalDof(getBoundaryElement(k)(i), X_DOF) )==0 ) neumData[ getNodalDof(getBoundaryElement(k)(i), X_DOF) ] = 0.0;
					if( neumData.count( getNodalDof(getBoundaryElement(k)(i), Y_DOF) )==0 || initializedDofs.count( getNodalDof(getBoundaryElement(k)(i), Y_DOF) )==0 ) neumData[ getNodalDof(getBoundaryElement(k)(i), Y_DOF) ] = 0.0;
					if( neumData.count( getNodalDof(getBoundaryElement(k)(i), Z_DOF) )==0 || initializedDofs.count( getNodalDof(getBoundaryElement(k)(i), Z_DOF) )==0 ) neumData[ getNodalDof(getBoundaryElement(k)(i), Z_DOF) ] = 0.0;

					if( bndCnd[bcId].dofs & X_MASK ){ neumData[ getNodalDof(getBoundaryElement(k)(i), X_DOF) ] += areaOver3*gx[0]; initializedDofs.insert(getNodalDof(getBoundaryElement(k)(i), X_DOF)); }
					if( bndCnd[bcId].dofs & Y_MASK ){ neumData[ getNodalDof(getBoundaryElement(k)(i), Y_DOF) ] += areaOver3*gx[1]; initializedDofs.insert(getNodalDof(getBoundaryElement(k)(i), Y_DOF)); }
					if( bndCnd[bcId].dofs & Z_MASK ){ neumData[ getNodalDof(getBoundaryElement(k)(i), Z_DOF) ] += areaOver3*gx[2]; initializedDofs.insert(getNodalDof(getBoundaryElement(k)(i), Z_DOF)); }
				}
			}
		}
	}else
		printf("\n%% UNKNOWN BOUNDARY CONDITION TYPE %d", bndCnd[bcId].type);
}

void LinearFEM::updateAllBoundaryData(){
	diriData.clear();
	neumData.clear();
	for(std::map<unsigned int,BoundaryCondition>::iterator it=bndCnd.begin(); it!=bndCnd.end(); ++it)
		updateBoundaryData(it->first);
}

void LinearFEM::setExternalAcceleration(unsigned int bodyId, const VectorField& g){
	fExtFields[bodyId]=&g;
}

void LinearFEM::assembleForceAndStiffness(UPDATE_MODE mode){
	class StiffnessAssemblyOp{
	public:
		UPDATE_MODE mode;
		std::vector<Eigen::Triplet<double> > K_triplets;
		Eigen::Matrix3d F;
		double energy; Vector9d stress; Matrix9d hessian;
		Eigen::Matrix<double, 12,1> force; Eigen::Matrix<double, 12,12> stiffness;

		StiffnessAssemblyOp(UPDATE_MODE mode_) : mode(mode_) {}

		inline void initialize(LinearFEM& fem){
			fem.f.setZero();
			if( fem.K.size()==0 ) mode=REBUILD; // if K is empty force assembly from scratch
			if( mode == UPDATE ){ // zero out coeffs but keep memory
				for(int k=0; k<fem.K.data().size(); ++k) fem.K.data().value(k)=0.0;
			}
		}

		inline void calculateElement(LinearFEM& fem, unsigned int k){
			Tetra ek = fem.getElement(k);

			// compute local force and stiffness
			fem.computeDeformationGradient(F, k);

			//if( materialModel.count(getBodyId(k)) == 0) printf("\n%% material missing for body %d on element %d", getBodyId(k) ,k);

			fem.getMaterialModel(k)->computeEnergyStressAndHessian(F,fem.getMaterialParameters(k).data() ,energy,stress,hessian);
			force     = -fem.getVolume(k)*fem.getDFdx(k).transpose()*stress; //equivalent to -getVolume(k)*stress.transpose()*getDFdx(k)
			if(  mode == REBUILD || mode == UPDATE ){
				stiffness =  fem.getVolume(k)*fem.getDFdx(k).transpose()*hessian*fem.getDFdx(k);
			}

			// force is the local internal force vector wrt. to the elements DOFs
			// 12 components, layout (v1x v1y v1z v2x v2y ...)
			// similar for stiffness (12x12) ...
		}

		inline void assembleVectorDOF(LinearFEM& fem, unsigned int k, unsigned int i, unsigned int idof, unsigned int lidof, unsigned int gidof){
			fem.f(gidof) += force(lidof);
		}

		inline void assembleMatrixDOF(LinearFEM& fem, unsigned int k, unsigned int i, unsigned int idof, unsigned int lidof, unsigned int gidof, unsigned int j, unsigned int jdof, unsigned int ljdof, unsigned int gjdof){
			if( mode == UPDATE ){
				fem.K.coeffRef(gidof,gjdof) += stiffness(lidof,ljdof);
			}else if( mode == REBUILD ){
				K_triplets.push_back(
					Eigen::Triplet<double>(gidof,gjdof, stiffness(lidof,ljdof) )
				); //ToDo: optimize: preallocate triplet storage, store only upper (or lower) part of K, use .selfAdjointView ... depends also on linear solver choice
			}
		}

		inline void finalize(LinearFEM& fem){
			if( mode == REBUILD ){ // build from triplets
				fem.K.resize(N_DOFS*fem.getNumberOfNodes(),N_DOFS*fem.getNumberOfNodes());
				fem.K.setFromTriplets(K_triplets.begin(),K_triplets.end());
			}
		}

	} stiffop(mode);

	LinearFEM::assemblyLoop(*this, stiffop);

	addNeumannBoundaryForces();
}

void LinearFEM::assembleViscousForceAndDamping(Eigen::VectorXd& vi, UPDATE_MODE mode){
	if( mode == UPDATE ){ // zero out coeffs but keep memory
		for(unsigned int k=0; k<D.data().size(); ++k)  D.data().value(k)=0.0;
	}
	for(std::map<unsigned int,vtkSmartPointer<ViscosityModel> >::iterator it=viscosityModel.begin(); it!=viscosityModel.end(); ++it )
		it->second->assembleForceAndDampingMatrix(vi,*this,it->first,mode);
}

void LinearFEM::assembleMassAndExternalForce(){
	class ConsistentMassAssemblyOp{
	public:
		double elemMass;
		Eigen::Vector3d gx,x0,xc;
		std::vector<Eigen::Triplet<double> > M_triplets;

		inline void initialize(LinearFEM& fem){
			fem.M.resize(N_DOFS*fem.getNumberOfNodes(),N_DOFS*fem.getNumberOfNodes());
			fem.f_ext.setZero();
		}
		inline void calculateElement(LinearFEM& fem, unsigned int k){
			Tetra ek = fem.getElement(k);
			if( fem.fExtFields.count( fem.getBodyId(k) ) ){
				x0 = 1.0/4.0*(fem.getRestCoord(ek(0)) + fem.getRestCoord(ek(1)) + fem.getRestCoord(ek(2)) + fem.getRestCoord(ek(3)));
				xc = 1.0/4.0*(fem.getDeformedCoord(ek(0)) + fem.getDeformedCoord(ek(1)) + fem.getDeformedCoord(ek(2)) + fem.getDeformedCoord(ek(3)));
				fem.fExtFields[fem.getBodyId(k)]->eval(gx,x0,xc, fem.simTime); // evaluate external acceleration at element centroid
			}else{
				gx.setZero();
			}
			elemMass = fem.getVolume(k)*( fem.materialModel[fem.getBodyId(k)]->getDensity( fem.getMaterialParameters(k).data() ) );
		}
		inline void assembleVectorDOF(LinearFEM& fem, unsigned int k, unsigned int i, unsigned int idof, unsigned int lidof, unsigned int gidof){
			fem.f_ext(gidof) += 1.0/4.0* elemMass * gx(idof); //ToDo: support non-const external acceleration --> use quadrature over element
		}
		inline void assembleMatrixDOF(LinearFEM& fem, unsigned int k, unsigned int i, unsigned int idof, unsigned int lidof, unsigned int gidof, unsigned int j, unsigned int jdof, unsigned int ljdof, unsigned int gjdof){
			if( idof==jdof ) M_triplets.push_back(
				Eigen::Triplet<double>(gidof,gjdof, elemMass* 1.0/((i==j)?10.0:20.0) )
			);
		}
		inline void finalize(LinearFEM& fem){
			fem.M.setFromTriplets(M_triplets.begin(), M_triplets.end());
		}
	} cmop;

	LinearFEM::assemblyLoop(*this, cmop);
}

void LinearFEM::applyDirichletBoundaryConditions(SparseMatrixD& S, double weight){
//#pragma omp parallel for schedule(guided)
	for(Eigen::Index k=0; k < S.outerSize(); ++k){
		for(SparseMatrixD::InnerIterator it(S,k); it; ++it){
			if( diriData.count(it.row()) ){
				it.valueRef() = (it.row()==it.col()) ? weight : 0.0;
			}
		}
	}
}

void LinearFEM::applyStaticBoundaryConditions(SparseMatrixD& S, Eigen::VectorXd& g, const Eigen::VectorXd& x0, double alpha){
	//moved to assembly: addNeumannBoundaryForces(g);

	applyDirichletBoundaryConditions(S);

	for(std::map<unsigned int,double>::iterator it=diriData.begin(); it!=diriData.end(); ++it){
		g(it->first) = alpha * (it->second - x0(it->first));
	}
}

void LinearFEM::applyDynamicBoundaryConditions(SparseMatrixD& S, Eigen::VectorXd& g, const Eigen::VectorXd& x0, const Eigen::VectorXd& vi, double dt){
	// apply Dirichlet boundary conditions by setting corresponding rows in S to identity
	// and write given data into g
	//ToDo: maybe reduce the system size? (not sure if it will increase the solver speed ...)
	//ToDo: allow per-node Dirichlet constraints to arbitrary plane or line (currently only per-dof i.e. axis-aligned implemented)

	double weight = 1.0; //S.diagonal().sum() / S.rows(); // average of diagonal //ToDo: test if weight screws up in sensitivity analysis

	applyDirichletBoundaryConditions(S, weight);

	for(std::map<unsigned int,double>::iterator it=diriData.begin(); it!=diriData.end(); ++it){
		g(it->first) = weight*( (it->second - x0(it->first))/dt - vi(it->first) );
	}
}

void LinearFEM::addNeumannBoundaryForces(){
	for(std::map<unsigned int,double>::iterator it=neumData.begin(); it!=neumData.end(); ++it){
		f(it->first) += it->second;
	}
}

void LinearFEM::staticSolveStep(double regularize){
	S = K;
	if( regularize > 0 ){
		SparseMatrixD I(K.rows(),K.cols());
		I.setIdentity();
		S+=regularize*I;
	}
	Eigen::VectorXd g = f+f_ext;
	applyStaticBoundaryConditions(S,g,x);

	//cout << endl << "cpp_g = [" << endl << (g) << endl << "];";
	//cout << endl << "cpp_S = [" << endl << (S) << endl << "];";

	//SparseSolver solver(S);
	linearSolver.compute(S);
	x += linearSolver.solve(g);

	// overwrite Dirichlet BCs with target values (just in case the linear solver introduced some numerical errors)
	if( doWriteDiriDataAfterSolve ) for(std::map<unsigned int,double>::iterator it=diriData.begin(); it!=diriData.end(); ++it){
		x(it->first) = it->second;
	}
}

int LinearFEM::staticSolve(double regularize, double eps){
	assembleForceAndStiffness();
	Eigen::VectorXd g=getVectorInternalForces()+getVectorExternalForces();
	Eigen::VectorXd x0=x; double firstResidual=freeDOFnorm(g);
	int iter;
	for(iter=0; iter<MAX_SOLVER_ITERS && freeDOFnorm(g)>eps || iter==0; ++iter){
		staticSolveStep(regularize);
		assembleForceAndStiffness();
		g=getVectorInternalForces()+getVectorExternalForces();
		if( doPrintResiduals) printf(" (%7.2lg)",freeDOFnorm(g));
		if(iter>3 && freeDOFnorm(g) > 1e6*std::max(1.0,firstResidual) || isnan(freeDOFnorm(g))){x=x0; return -1;}
	}
	if( (iter>=MAX_SOLVER_ITERS) && (freeDOFnorm(g) > 1e3 * eps) ) {x=x0; return -1;} 
	mesh->GetPoints()->Modified(); // mark VTK structure as modified so rendering etc. can update properly
	return iter;
}

int LinearFEM::staticSolvePreregularized(double preEps, double regularize, unsigned int maxPreIters, double eps){
	if( regularize > 0.0 ){
		unsigned int currentMaxIters = MAX_SOLVER_ITERS;
		MAX_SOLVER_ITERS = maxPreIters;
		int r = staticSolve(regularize, preEps);
		MAX_SOLVER_ITERS = currentMaxIters;
		if( r<0 ) return r; // pre-solve failed
		if( doPrintResiduals ) printf("\n%% ");
	}
	return staticSolve(0.0, eps);
}


void LinearFEM::dynamicExplicitEulerSolve(double dt){
	S = M;
	Eigen::VectorXd dv = dt* (f+f_ext);
	applyDynamicBoundaryConditions(S,dv,x,v,dt);
	linearSolver.compute(S);
	currentAcceleration = linearSolver.solve(dv);
	v += currentAcceleration;
	currentAcceleration /= dt;
	x += dt*v;
	simTime += dt;
	
	if( doWriteDiriDataAfterSolve ) for(std::map<unsigned int,double>::iterator it=diriData.begin(); it!=diriData.end(); ++it){
		x(it->first) = it->second;
	}
	mesh->GetPoints()->Modified(); // mark VTK structure as modified so rendering etc. can update properly
}

int LinearFEM::dynamicImplicitTimestep(double dt, double eps){
	Eigen::VectorXd vi( v.size() );
	Eigen::VectorXd x0 = x, g;

	if( useBDF2 && (x_old.size()==0 || v_old.size()==0) ){ // x_old or v_old for BDF2 integration need initializing ...
		x_old = x - dt*v;
		v_old = v; // assume zero acceleration for the start
	}

	vi.setZero(); // x == x0 here
	if( useBDF2 ){
		//x = x0 + 2.0/3.0*dt*vi + 1.0/3.0*(x0-x_old);
		vi = 0.5*(x_old-x0)/dt;
	}

	double firstResidual=-1.0;
	bool done=false; int iter;

	if( M.size()==0 ) assembleMassAndExternalForce(); // just in case ...

	if( doPrintResiduals) printf(" (t = %7.4lg) \t", simTime);
	for(iter=0; iter<MAX_SOLVER_ITERS && !done; ++iter){

		if( useBDF2 )
			x = x0 + 2.0/3.0*dt*vi + 1.0/3.0*(x0-x_old);
		else
			x = x0+dt*vi;

		assembleForceAndStiffness( UPDATE ); // overwrites f vector (does setZero() first), also writes K = -df/dx
		assembleViscousForceAndDamping( vi, UPDATE ); // adds to f vector - keep order of assembly fcn. calls! - also writes D = -df_visc/dv

		if( useBDF2 )
			g = (f+f_ext) - (1.5/dt)*(M*(vi-v)) + 0.5*(M*(v-v_old))/dt;
		else
			g = (f+f_ext) - (M*(vi-v))/dt; // solve 0 = f(x0+dt*vi) + f_ext - M/dt*(vi-v)

		if( firstResidual < 0.0 ) firstResidual = freeDOFnorm(g);
		else if(iter>3 && freeDOFnorm(g) > 1e6*std::max(1.0,firstResidual) || isnan(freeDOFnorm(g))){x=x0; v.setZero(); return -1;} // kill the solver if it has diverged

		if( iter>0 && (freeDOFnorm(g) < eps) ) done=true;
		else{
			if( useBDF2 )
				S =  2.0/3.0*K*dt + 1.5*M/dt; //note: df/dv = df/dx*dx/dv, where dx/dv = 2/3*dt for BDF2
			else
				S =  K*dt + M/dt;
			if( D.size()>0 ) S += D; // add viscous damping matrix if it exists

			applyDynamicBoundaryConditions(S,g,x0,vi,dt);
			//SparseSolver solver( S );
			linearSolver.compute(S);
			Eigen::VectorXd dv = linearSolver.solve( g );
			vi += dv; //could stabilize with line-search if freeDOFnorm(g) has not decreased ...
		}
		if( doPrintResiduals) printf(" (%7.2lg)",freeDOFnorm(g));
	}
	if( !done && (freeDOFnorm(g) > 1e3 * eps) ) {x=x0; v.setZero(); return -1;} // kill the solver if it has failed to converge

		if( useBDF2 )
			currentAcceleration = (1.5/dt)*(vi-v) - (0.5/dt)*(v-v_old);
		else
			currentAcceleration = (vi-v)/dt;
	

	if( useBDF2 ){ x_old=x0; v_old=v; }
	v=vi; // x is already set to end-of-timestep positions
	simTime += dt;
	
	if( doWriteDiriDataAfterSolve ) for(std::map<unsigned int,double>::iterator it=diriData.begin(); it!=diriData.end(); ++it){
		x(it->first) = it->second;
	}
	mesh->GetPoints()->Modified(); // mark VTK structure as modified so rendering etc. can update properly
	return iter;
}

void LinearFEM::computeTotalInternalForceOnBoundary(unsigned int bndry, double& totalArea, Eigen::Vector3d& fB){
	// compute the internal force on the given boundary as sum_tri ( stress*normal*area )
	fB.setZero();
	totalArea=0.0;
	for(unsigned int k=0; k<getNumberOfBndryElems(); ++k) if(getBoundaryId(k)==bndry){
		unsigned int parent = getBoundaryParent(k);
		// compute the stress of the parent element ...
		double energy; Vector9d stress; Matrix9d hessian; Eigen::Matrix3d F;
		computeDeformationGradient(F, parent);
		materialModel[getBodyId(parent)]->computeEnergyStressAndHessian(F,getMaterialParameters(parent).data() ,energy,stress,hessian);
		// compute the normal of the boundary element ...
		Eigen::Vector3d n = ( getRestCoord(getBoundaryElement(k)(1)) - getRestCoord(getBoundaryElement(k)(0)) ).cross( getRestCoord(getBoundaryElement(k)(2))-getRestCoord(getBoundaryElement(k)(0)) ).normalized();
		// sum up ...
		Eigen::Map<Eigen::Matrix3d> stressMatrix(stress.data());
		fB += stressMatrix*n*getArea(k);
		totalArea += getArea(k);
	}
}

//ToDo: combine the following computeAverage... functions into one ...
void LinearFEM::computeAverageRestCoordinateOfBoundary(unsigned int bndry, Eigen::Vector3d& x0B){
	x0B.setZero();
	double totalArea=0.0;
	for(unsigned int k=0; k<getNumberOfBndryElems(); ++k) if(getBoundaryId(k)==bndry){
		Tri ek = getBoundaryElement(k);
		totalArea += getArea(k);
		for(int i=0; i<3; ++i){
			x0B += 1.0/3.0*getArea(k) * getRestCoord(ek(i));
		}
	}
	x0B /= totalArea;
}
void LinearFEM::computeAverageDeformedCoordinateOfBoundary(unsigned int bndry, Eigen::Vector3d& xB){
	xB.setZero();
	double totalArea=0.0;
	for(unsigned int k=0; k<getNumberOfBndryElems(); ++k) if(getBoundaryId(k)==bndry){
		Tri ek = getBoundaryElement(k);
		totalArea += getArea(k);
		for(int i=0; i<3; ++i){
			xB += 1.0/3.0*getArea(k) * getDeformedCoord(ek(i));
		}
	}
	xB /= totalArea;
}
void LinearFEM::computeAverageDisplacementOfBoundary(unsigned int bndry, Eigen::Vector3d& uB){
	uB.setZero();
	double totalArea=0.0;
	for(unsigned int k=0; k<getNumberOfBndryElems(); ++k) if(getBoundaryId(k)==bndry){
		Tri ek = getBoundaryElement(k);
		totalArea += getArea(k);
		for(int i=0; i<3; ++i){
			uB += 1.0/3.0*getArea(k) * ( getDeformedCoord(ek(i)) - getRestCoord(ek(i)) );
		}
	}
	uB /= totalArea;
}
void LinearFEM::computeAverageVelocityOfBoundary(unsigned int bndry, Eigen::Vector3d& vB){
	vB.setZero();
	double totalArea=0.0;
	for(unsigned int k=0; k<getNumberOfBndryElems(); ++k) if(getBoundaryId(k)==bndry){
		Tri ek = getBoundaryElement(k);
		totalArea += getArea(k);
		for(int i=0; i<3; ++i){
			vB += 1.0/3.0*getArea(k) * ( getVelocity(ek(i)) );
		}
	}
	vB /= totalArea;
}

double LinearFEM::computeBodyVolume(unsigned int bodyID){
	double vol=0.0;
	for(unsigned int k=0; k<getNumberOfElems(); ++k){
		if( bodyID==-1 || bodyID==getBodyId(k) ) vol += getVolume(k);
	}
	return vol;
}

double LinearFEM::computeBodyVolumeAndCentreOfMass(Eigen::Vector3d& com, unsigned int bodyID){
	com.setZero();
	double vol=0.0, volK;
	for(unsigned int k=0; k<getNumberOfElems(); ++k) if( bodyID==-1 || bodyID==getBodyId(k) ){
		volK = getVolume(k);
		vol += volK;
		com += volK*(1.0/4.0)*(getRestCoord(getElement(k)(0))+getRestCoord(getElement(k)(1))+getRestCoord(getElement(k)(2))+getRestCoord(getElement(k)(3)));
	}
	com /= vol;
	//printf("\n%% mesh centre of mass: [ "); cout << com.transpose() << " ]" << endl;
	return vol;
}

void LinearFEM::updateDataMaps(){
	new (&deformedCoords) Vect3Map((double*)mesh->GetPoints()->GetVoidPointer(0), 3,mesh->GetPoints()->GetNumberOfPoints());
	new (&restCoords)     Vect3Map((double*)meshBoundary->GetPoints()->GetVoidPointer(0)    , 3,meshBoundary->GetPoints()->GetNumberOfPoints());
	new (&x)              VectXMap((double*)mesh->GetPoints()->GetVoidPointer(0), 3*mesh->GetPoints()->GetNumberOfPoints());

	new (&elems)          TetraMap(mesh->GetCells()->GetPointer()+1 ,4,mesh->GetCells()->GetNumberOfCells()); // +1 because we need to skip the first element in the raw array (each row starts with the number of indices ==4 of the following element)
	new (&bndElems)         TriMap(meshBoundary->GetCells()->GetPointer()+1 ,3,meshBoundary->GetCells()->GetNumberOfCells()); // +1 because we need to skip the first element in the raw array (each row starts with the number of indices ==3 of the following element)

	vtkAbstractArray* data;

	data = mesh->GetPointData()->GetAbstractArray( fieldNames[VELOCITY_NAME].c_str() );
	if( data==NULL ) printf("\n%% MISSING VELOCITIES IN MESH DATA");
	else{
		new (&velocities) Vect3Map((double*)data->GetVoidPointer(0), 3,mesh->GetPoints()->GetNumberOfPoints());
		new (&v) VectXMap((double*)data->GetVoidPointer(0), 3*getNumberOfNodes());
	}

	data = mesh->GetPointData()->GetAbstractArray( fieldNames[FORCE_NAME].c_str() );
	if( data==NULL ) printf("\n%% MISSING FORCES IN MESH DATA");
	else
		new (&f) VectXMap((double*) data->GetVoidPointer(0), 3*getNumberOfNodes());

	data = mesh->GetPointData()->GetAbstractArray( fieldNames[FORCE_EXT_NAME].c_str() );
	if( data==NULL ) printf("\n%% MISSING LOADS IN MESH DATA");
	else
		new (&f_ext) VectXMap((double*)data->GetVoidPointer(0), 3*getNumberOfNodes());

	data = mesh->GetCellData()->GetAbstractArray( fieldNames[VOLUME_NAME].c_str() );
	if( data==NULL ) printf("\n%% MISSING VOLUMES IN MESH DATA");
	else
		new (&vol) VectXMap((double*) data->GetVoidPointer(0),getNumberOfElems());

	data = meshBoundary->GetCellData()->GetAbstractArray( fieldNames[AREA_NAME].c_str() );
	if( data==NULL ) printf("\n%% MISSING AREAS IN MESH DATA");
	else
		new (&area) VectXMap((double*) data->GetVoidPointer(0),getNumberOfBndryElems());

	data = mesh->GetCellData()->GetAbstractArray( fieldNames[BODY_NAME].c_str() );
	if( data==NULL ) printf("\n%% MISSING BODY IDs IN MESH DATA");
	else
		new (&bodyId) VectIMap((unsigned int*) data->GetVoidPointer(0), mesh->GetNumberOfCells());

	data = meshBoundary->GetCellData()->GetAbstractArray( fieldNames[BOUNDARY_NAME].c_str() );
	if( data==NULL ) printf("\n%% MISSING BOUNDARY IDs IN MESH DATA");
	else
		new (&bndId) VectIMap((unsigned int*) data->GetVoidPointer(0), meshBoundary->GetNumberOfCells());

	data = meshBoundary->GetCellData()->GetAbstractArray( fieldNames[PARENT_NAME].c_str() );
	if( data==NULL ) printf("\n%% MISSING PARENT IDs IN MESH DATA");
	else
		new (&bndParent) VectIMap((unsigned int*) data->GetVoidPointer(0), meshBoundary->GetNumberOfCells());
}

unsigned int LinearFEM::getNumberOfNodes() const { return mesh->GetPoints()->GetNumberOfPoints(); }
unsigned int LinearFEM::getNumberOfElems() const { return mesh->GetCells() ->GetNumberOfCells();  }
unsigned int LinearFEM::getNumberOfBndryElems() const { return meshBoundary->GetCells()->GetNumberOfCells();  }

const Vect3Map& LinearFEM::getRestCoords() const{
	return restCoords;
}
Vect3Map& LinearFEM::getRestCoords(){
	return restCoords;
}
Vect3Map& LinearFEM::getDeformedCoords(){
	return deformedCoords;
}
Vect3Map& LinearFEM::getVelocities(){
	return velocities;
}
TetraMap& LinearFEM::getElements(){
	return elems;
}
TriMap& LinearFEM::getBoundaryElements(){
	return bndElems;
}
Vect3 LinearFEM::getRestCoord(unsigned int i){
	return getRestCoords().block<3,1>(0,i);
}
Vect3 LinearFEM::getDeformedCoord(unsigned int i){
	return getDeformedCoords().block<3,1>(0,i);
}
Vect3 LinearFEM::getVelocity(unsigned int i){
	return getVelocities().block<3,1>(0,i);
}
Tetra LinearFEM::getElement(unsigned int k){
	return getElements().block<4,1>(0,k);
}
Tri LinearFEM::getBoundaryElement(unsigned int k){
	return getBoundaryElements().block<3,1>(0,k);
}
double& LinearFEM::getVolume(unsigned int k){
	return vol(k);
}
double& LinearFEM::getArea(unsigned int k){
	return area(k);
}
unsigned int& LinearFEM::getBodyId(unsigned int k){
	return bodyId(k); //((unsigned intArray*) mesh->GetCellData()->GetAbstractArray( fieldNames[BODY_NAME].c_str() ))->GetTuple1(k);
}
unsigned int& LinearFEM::getBoundaryId(unsigned int k){
	return bndId(k);
}
unsigned int& LinearFEM::getBoundaryParent(unsigned int k){
	return bndParent(k);
}
Eigen::Map<Eigen::Matrix3d> LinearFEM::getBasisInv(unsigned int k){
	return Eigen::Map<Eigen::Matrix3d>( elemBasisInv.block<9,1>(0,k).data() );
}

Eigen::Map<Eigen::Matrix<double,9,12> > LinearFEM::getDFdx(unsigned int k){
	return Eigen::Map<Eigen::Matrix<double,9,12> >( elemDFdx.block<9*12,1>(0,k).data() );
}

Eigen::Map<Eigen::VectorXd> LinearFEM::getMaterialParameters(unsigned int k){
	return Eigen::Map<Eigen::VectorXd>( &elemMatParams.coeffRef(0,k), materialModel[getBodyId(k)]->getNumberOfParameters() );
}
Eigen::Map<Eigen::VectorXd> LinearFEM::getViscosityParameters(unsigned int k){
	return Eigen::Map<Eigen::VectorXd>( &elemViscParams.coeffRef(0,k), viscosityModel[getBodyId(k)]->getNumberOfParameters() );
}
vtkSmartPointer<HomogeneousMaterial> LinearFEM::getMaterialModel(unsigned int k){
	if( materialModel.count(getBodyId(k))==0 ) printf("\n%% MISSING MATERIAL MODEL FOR ELEMENT %u!\n",k);
	return materialModel[ getBodyId(k) ];
}
vtkSmartPointer<ViscosityModel> LinearFEM::getViscosityModel(unsigned int k){
	if( viscosityModel.count(getBodyId(k))>0 ) return viscosityModel[ getBodyId(k) ];
	return NULL;
}


double LinearFEM::freeDOFnorm(Eigen::VectorXd& g){
	double gnorm=0.0;
	for(unsigned int i=0; i<g.size(); ++i){
		if( diriData.count(i)==0 ) gnorm += std::abs(g(i)); //g(i)*g(i);//
	}
	return gnorm;
}

void LinearFEM::reset(){
	if( x_reset.size() == x.size() )
		x = x_reset;
	else
		deformedCoords = restCoords;

	if( v_reset.size() == v.size() )
		v = v_reset;
	else
		v.setZero();

	f.setZero();
	f_ext.setZero();
	// also delete matrices, so caller is forced to assemble everything from the reset state before starting a new sim
	M.resize(0,0);
	K.resize(0,0);
	D.resize(0,0);
	// delete BDF2 previous position and velocity data
	x_old.resize(0);
	v_old.resize(0);
	simTime=0.0;
}
void LinearFEM::setResetPoseFromCurrent(){
	x_reset = x;
}
void LinearFEM::clearResetPose(){
	x_reset.resize(0);
}
void LinearFEM::setResetVelocitiesFromCurrent(){
	v_reset = v;
}
void LinearFEM::clearResetVelocities(){
	v_reset.resize(0);
}

LinearFEM::~LinearFEM(){}


// quick preview window ...
// VTK preview rendering includes ...
#include <vtkWeakPointer.h>
#include <vtkDataSetMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkCamera.h>
#include <vtkRenderWindow.h>
#include <vtkCubeAxesActor.h>
#include <vtkRenderWindowInteractor.h>
//VTK_MODULE_INIT(vtkInteractionStyle)

MeshPreviewWindow::MeshPreviewWindow(vtkWeakPointer<vtkUnstructuredGrid> mesh){
	meshMapper = vtkSmartPointer<vtkDataSetMapper>::New();
	meshMapper->SetInputData(mesh);
	meshActor = vtkSmartPointer<vtkActor>::New();
	meshActor->SetMapper(meshMapper);
	//meshActor->GetProperty()->SetRepresentationToWireframe();
	meshActor->GetProperty()->EdgeVisibilityOn();
	//meshActor->GetProperty()->SetEdgeColor(1.0,1.0,1.0);
	meshActor->GetProperty()->SetEdgeColor(0.0,0.5,0.0);
	meshActor->GetProperty()->SetOpacity(0.8); meshActor->GetProperty()->BackfaceCullingOn();
	meshActor->GetProperty()->SetColor(0.3,0.3,0.5);
	renderer = vtkSmartPointer<vtkRenderer>::New();
	renWin = vtkSmartPointer<vtkRenderWindow>::New();
	iren = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	iren->SetRenderWindow(renWin);
	renWin->AddRenderer(renderer);
	renderer->AddActor(meshActor); 

	cubeAxesActor = vtkSmartPointer<vtkCubeAxesActor>::New();
	cubeAxesActor->SetCamera(renderer->GetActiveCamera());
	cubeAxesActor->SetFlyMode(vtkCubeAxesActor::VTK_FLY_FURTHEST_TRIAD);
	cubeAxesActor->XAxisMinorTickVisibilityOff();
	cubeAxesActor->YAxisMinorTickVisibilityOff();
	cubeAxesActor->ZAxisMinorTickVisibilityOff();
	cubeAxesActor->XAxisLabelVisibilityOff();
	cubeAxesActor->YAxisLabelVisibilityOff();
	cubeAxesActor->ZAxisLabelVisibilityOff();
	renderer->AddActor(cubeAxesActor);

	renderer->GetActiveCamera()->SetPosition(0.1,-1.0,0.4);
	renderer->GetActiveCamera()->SetViewUp(0.0, 0.0, 1.0);
	renderer->GetActiveCamera()->SetClippingRange(1e-6,1e12);
	renderer->ResetCamera(); // zoom to data (maintains direction of view)
	renderer->SetInteractive(0);
	renWin->SetSize(640,480);
}
void MeshPreviewWindow::reset(){
	renWin->Finalize();
	renWin->Start();
	renWin->SetSize(640,480);
}
void MeshPreviewWindow::render(){
	double bounds[6], d[3];
	meshMapper->GetInput()->Modified();
	meshMapper->GetBounds(bounds);
	renderer->ResetCamera(bounds);
	d[0] = bounds[1]-bounds[0]; d[1] = bounds[3]-bounds[2]; d[2] = bounds[5]-bounds[4]; // (x,y,z)_max - (x,y,z)_min
	bounds[0] -= d[0]*0.25; bounds[1] += d[0]*0.25;
	bounds[2] -= d[1]*0.25; bounds[3] += d[1]*0.25;
	bounds[4] -= d[2]*0.25; bounds[5] += d[2]*0.25;
	cubeAxesActor->SetBounds(bounds);
	renWin->Start();
	renWin->Render();
	renWin->Frame();
}
