#ifndef LINEARFEM_H
#define LINEARFEM_H

#include "types.h"
#include <vtkSmartPointer.h>
#include <vtkWeakPointer.h>

class vtkPoints;
class vtkDoubleArray;
class vtkUnstructuredGrid;
class vtkXMLUnstructuredGridWriter;

// for the preview window -- consider moving to a separate file ...
class vtkDataSetMapper;
class vtkActor;
class vtkRenderer;
class vtkRenderWindow;
class vtkCubeAxesActor;
class vtkRenderWindowInteractor;

namespace MyFEM{
	class HomogeneousMaterial;
	class ViscosityModel;

	enum DOF_MASK { X_MASK=1, Y_MASK=2, Z_MASK=4, ALL_MASK=7 }; //ToDo: support arbitrary plane or line constraints ... currently only axis-aligned version implemented
	enum BC_TYPE { BC_DIRICHLET, BC_NEUMANN };
	class BoundaryCondition{
	public:
		const VectorField* data;
		BC_TYPE type;
		DOF_MASK dofs;
	};

	class LinearFEM{
	public:
		static unsigned int MAX_SOLVER_ITERS;
		static double FORCE_BALANCE_EPS;
		LinearFEM();
		virtual ~LinearFEM();

		// functions to load a FEM mesh from files
		int  loadMeshFromElmerFiles(std::string fileName);
		void loadMeshFromVTUFile(std::string fileName, std::string boundaryFileName);
		std::string saveMeshToVTUFile(std::string fileName, bool alsoWriteBoundaryFile=false);

		// allocate memory in the mesh object
		void initializeMeshStorage();

		// precompute some data on the mesh (element basis matrices, shape function gradients, rest volumes, ...)
		void precomputeMeshData();

		// set the material model (we assign one material model per mesh body, but every element has it's own set of material parameters)
		enum MAT_TYPE { ISOTROPIC_LINEAR_ELASTIC, ISOTROPIC_NEOHOOKEAN };
		void initializeMaterialModel( unsigned int bodyId, MAT_TYPE matType=ISOTROPIC_LINEAR_ELASTIC, double lameLambda=0.0, double lameMu=1.0, double density=0.1);
		void initializeMaterialModel( unsigned int bodyId, vtkSmartPointer<HomogeneousMaterial> elastModel, double* defaultParams=NULL);  // if defaultParams is NULL, all parameters will be set to zero, otherwise defaultParams must contain getNumberOfParameters() entries as specified by the given elastModel
		void initializeViscosityModel(unsigned int bodyId, vtkSmartPointer<ViscosityModel> viscModel, double* defaultParams=NULL); // if defaultParams is NULL, all parameters will be set to zero, otherwise defaultParams must contain getNumberOfParameters() entries as specified by the given viscModel

		void computeDeformationGradient(Eigen::Matrix3d& F, unsigned int k);
		void computeVelocityGradient(Eigen::Matrix3d& dv, unsigned int k, VectXMap& vi);

		void setBoundaryCondition(unsigned int boundaryId, const VectorField& g, BC_TYPE type=BC_DIRICHLET, DOF_MASK dofs=ALL_MASK);
		void setExternalAcceleration(unsigned int bodyId, const VectorField& g);

		void assembleForceAndStiffness(UPDATE_MODE mode=UPDATE);
		void assembleViscousForceAndDamping(Eigen::VectorXd& vi, UPDATE_MODE mode=UPDATE);
		void assembleMassAndExternalForce(); //ToDo: add update mode here as well?
		void rebuildExternalForceAndBoundaryData(){ updateAllBoundaryData(); assembleMassAndExternalForce(); } // can be used after remeshing to compute load and boundary data for the new mesh from previously set fields
		void staticSolveStep(double regularize=0.0); // one iteration of elastostatics solving u += (K+regularize*I)^-1 (f + f_ext) -- assumes assembly has been done before
		int  staticSolve(double regularize=0.0, double eps=FORCE_BALANCE_EPS); // solve elastostatics by iterating until convergence eps
		int  staticSolvePreregularized(double preEps, double regularize, unsigned int maxPreIters=MAX_SOLVER_ITERS, double eps=FORCE_BALANCE_EPS); // solve elastostatics by first iterating with regularization to tolerance preEps with max iterations maxPreIters, then without regularization iterating until convergence eps up to default iteration limit
		void dynamicExplicitEulerSolve(double timestep); // one timestep of explicit Euler solving v += M^-1 (f + f_ext) -- assumes assembly has been done before
		virtual int dynamicImplicitTimestep(double timestep, double eps=FORCE_BALANCE_EPS); // one timestep of implicit Euler solving v += (M+timestep^2)^-1 ( M*v0 + f + f_ext) -- assumes mass assembly has been done before, assembles internal force and stiffness, iterates until convergence (squared norm of residual force < eps) -- returns number of Newton iterations or negative on error
		void computeTotalInternalForceOnBoundary(unsigned int bndry,  double& totalArea, Eigen::Vector3d& fB);
		
		void computeAverageRestCoordinateOfBoundary(unsigned int bndry, Eigen::Vector3d& x0B);
		void computeAverageDeformedCoordinateOfBoundary(unsigned int bndry, Eigen::Vector3d& xB);
		void computeAverageDisplacementOfBoundary(unsigned int bndry, Eigen::Vector3d& uB);
		void computeAverageVelocityOfBoundary(unsigned int bndry, Eigen::Vector3d& vB);
		double computeBodyVolume(unsigned int bodyID = -1); // compute the total volume of all elements with the given body ID (by default all elements)
		double computeBodyVolumeAndCentreOfMass(Eigen::Vector3d& com, unsigned int bodyID=-1);

		// data mapping and accessors ...
		void       updateDataMaps();
		unsigned int  getNumberOfNodes() const;
		unsigned int  getNumberOfElems() const;
		unsigned int  getNumberOfBndryElems() const;
		const Vect3Map& getRestCoords() const;
		Vect3Map&  getRestCoords();
		Vect3Map&  getDeformedCoords();
		Vect3Map&  getVelocities();
		TetraMap&  getElements();
		TriMap&    getBoundaryElements();
		Vect3      getRestCoord(unsigned int i);
		Vect3      getDeformedCoord(unsigned int i);
		Vect3      getVelocity(unsigned int i);
		Tetra      getElement(unsigned int k);
		Tri        getBoundaryElement(unsigned int k);
		double&    getVolume(unsigned int k);
		double&    getArea(unsigned int k);
		unsigned int& getBodyId(unsigned int k);
		unsigned int& getBoundaryId(unsigned int k);
		unsigned int& getBoundaryParent(unsigned int k);

		inline SparseMatrixD& getMassMatrix(){return M;}
		inline SparseMatrixD& getStiffnessMatrix(){return K;}
		inline SparseMatrixD& getDampingMatrix(){return D;}
		inline SparseMatrixD& getLastSystemMatrix(){return S;} // the system matrix solved in the most recent ...Solve() function call

		// these functions return Eigen::Map objects, which directly access the internal memory!
		inline VectXMap getVectorDeformedCoords(){return x;}
		inline VectXMap getVectorVelocities(){return v;}
		inline VectXMap getVectorInternalForces(){return f;}
		inline VectXMap getVectorExternalForces(){return f_ext;}

		Eigen::Map<Eigen::Matrix3d> getBasisInv(unsigned int k);
		Eigen::Map<Eigen::Matrix<double,9,12> > getDFdx(unsigned int k);
		Eigen::Map<Eigen::VectorXd> getMaterialParameters(unsigned int k);
		Eigen::Map<Eigen::VectorXd> getViscosityParameters(unsigned int k);
		vtkSmartPointer<HomogeneousMaterial> getMaterialModel(unsigned int k);
		vtkSmartPointer<ViscosityModel>      getViscosityModel(unsigned int k);
		std::map<unsigned int,vtkSmartPointer<HomogeneousMaterial> > getMaterialModels() const {return materialModel;} // key is body-id
		std::map<unsigned int,vtkSmartPointer<ViscosityModel>      > getViscosityModels() const {return viscosityModel;} // key is body-id

		enum DOF_IDX{X_DOF, Y_DOF, Z_DOF, N_DOFS};
		inline unsigned int getNodalDof(unsigned int nodeId, unsigned int dof){return N_DOFS*nodeId+dof;}
		double freeDOFnorm(Eigen::VectorXd& g); // compute the squared norm of g while skipping all constrained DOFs

		void reset(); // reset deformed coords, velocities, internal and external forces
		void setResetPoseFromCurrent(); // store the current deformed coordinates and use them as reset
		void clearResetPose(); // delete stored reset coordinates - reverts to default behaviour of resetting to rest coordinates
		void setResetVelocitiesFromCurrent();
		void clearResetVelocities();

		// these methods modify the linear system to include active boundary conditions from stored data (not re-evaluating the input fields - 
		void applyDirichletBoundaryConditions(SparseMatrixD& S, double weight=1.0); // set rows in S to identity (1 on diagonal, 0 elsewhere) for all Dirichlet DOFs
		void applyStaticBoundaryConditions(SparseMatrixD& S, Eigen::VectorXd& g, const Eigen::VectorXd& x, double alpha=1.0); // interpret Dirichlet data as target positions and set RHS vector to difference from given x (current - assumes we're solving for an update to the positions)
		void applyDynamicBoundaryConditions(SparseMatrixD& S, Eigen::VectorXd& g, const Eigen::VectorXd& x, const Eigen::VectorXd& vi, double dt);
		void addNeumannBoundaryForces(); // writes directly to internal force vector f
		// these methods compute per-DOF intermediate data from active boundary conditions
		void updateBoundaryData(unsigned int bcId);
		void updateAllBoundaryData();

		// ==============================
		// data starts here -- only access directly if you know what you're doing

		// geometry data (VTK)
		vtkSmartPointer<vtkUnstructuredGrid> mesh;
		vtkSmartPointer<vtkUnstructuredGrid> meshBoundary; //the boundary mesh object stores rest coordinates for ALL nodes, and triangle elements on the boundary
		vtkSmartPointer<vtkXMLUnstructuredGridWriter> vtkWriter;

		// maps of VTK data to Eigen
		TetraMap elems;
		Vect3Map restCoords, deformedCoords, velocities;
		VectXMap x; // deformed coordinates mapped to single vector for solving
		VectXMap f,f_ext,v; // internal force, external force, and velocity (nodal data)
		VectXMap vol, area; // volumes of undeformed elements, area of boundary elements
		TriMap   bndElems;
		VectIMap bodyId, bndId, bndParent; // body group ID for each (volume) eleme,t boundary group ID for each boundary element, parent volume element for each boundary element

		// linear algebra data (Eigen)
		Eigen::MatrixXd elemBasisInv;  // inverse of rest pose basis matrices (x1-x0 x2-x0 x3-x0)^-1 // layout (3x3 x m)  unrolled into (9 x m)
		Eigen::MatrixXd elemDFdx;      // dF/dx = shape function gradients wrt. local DOFs per elem  // layout (9x12 x m) unrolled into (108 x m)
		Eigen::MatrixXd elemMatParams; // material parameters per element // layout (p x m) where p is the number of material parameters, m the number of elements
		Eigen::MatrixXd elemViscParams;// viscosity parameters per element - layout (p x m) where p is the number of viscous  parameters, m the number of elements

		SparseMatrixD M,K,D,S; // mass, stiffness, damping, and system matrix (S will be set by solver functions: K for statics, M for explicit dynamics, M/dt+dt*K for implicit dynamics, modified for Dirichlet boundary conditions and possibly regularized)
		SparseSolver linearSolver; // store the linear solver object, in case we solve the same system with multiple right hand sides, could eventually replace S above.
		
		//ToDo: map these into the VTK structure, so they get automatically interpolated if we remesh!
		Eigen::VectorXd currentAcceleration; // the acceleration computed in the most recent time step solve of a dynamic sim -- just in case someone needs to know it, output only - has no effect on the simulation
		Eigen::VectorXd x_old,v_old; // for BDF2 time integration
		Eigen::VectorXd x_reset, v_reset; // use this set of coordinates as deformed coordinates for reset() - if empty, rest coords will be used, same for velocities (zero by default)

		// boundary conditions, external loads, material types ...
		std::map<unsigned int,double> diriData; // map DOF IDs to BC data for Dirichlet boundaries
		std::map<unsigned int,double> neumData; // map DOF IDs to BC data for Neumann   boundaries
		std::map<unsigned int,BoundaryCondition> bndCnd; // map boundary IDs to input fields so we can rebuild boundary conditions (for example after remeshing)
		std::map<unsigned int,const VectorField*> fExtFields; // map body IDs to input fields so we can also rebuild the external forces
		std::map<unsigned int,vtkSmartPointer<HomogeneousMaterial> > materialModel; // map body IDs to elastic material model
		std::map<unsigned int,vtkSmartPointer<ViscosityModel>      > viscosityModel; // map body IDs to viscous damping  model

		// simulated time (incremented during dynamicImplicitTimestep and dynamicExplicitEulerSolve) used to evaluate fields
		double simTime;

		// some flags and stuff
		bool doPrintResiduals, doWriteDiriDataAfterSolve;
		bool useBDF2; // use BDF2 instead of implicit Euler (BDF1) in dynamicImplicitTimestep -- may degrade accuracy of adjoint sensitivity analysis


		template <typename AssemblyOperator>
		static void assemblyLoop(LinearFEM& fem, AssemblyOperator& aop){
			aop.initialize(fem);
			for(unsigned int k=0; k<fem.getNumberOfElems(); ++k){
				Tetra ek = fem.getElement(k);
				aop.calculateElement(fem, k);
				for(int i=0; i<4; ++i){
					for(int idof=0; idof<N_DOFS; ++idof){
						unsigned int lidof = fem.getNodalDof(   i ,idof); // local  dof index
						unsigned int gidof = fem.getNodalDof(ek(i),idof); // global dof index
						aop.assembleVectorDOF(fem, k,i,idof,lidof,gidof);
						for(int j=0; j<4; ++j){
							for(int jdof=0; jdof<N_DOFS; ++jdof){
								unsigned int ljdof = fem.getNodalDof(   j ,jdof); // local  dof index
								unsigned int gjdof = fem.getNodalDof(ek(j),jdof); // global dof index
								aop.assembleMatrixDOF(fem, k,i,idof,lidof,gidof, j,jdof,ljdof,gjdof);
							}
						}
					}
				}
			}
			aop.finalize(fem);
		}
	
	protected:
	};


	// VTK preview rendering ...
	// quick preview window
	class MeshPreviewWindow{
	public:
		MeshPreviewWindow(vtkWeakPointer<vtkUnstructuredGrid> mesh);
		void render();
		void reset();
	protected:
		vtkSmartPointer<vtkDataSetMapper> meshMapper;
		vtkSmartPointer<vtkActor> meshActor;
		vtkSmartPointer<vtkRenderer> renderer;
		vtkSmartPointer<vtkRenderWindow> renWin;
		vtkSmartPointer<vtkCubeAxesActor> cubeAxesActor;
		vtkSmartPointer<vtkRenderWindowInteractor> iren;
	};
}
#endif
