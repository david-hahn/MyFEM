#ifndef TYPES_H
#define TYPES_H

#include <Eigen/Eigen>
#include <vtkType.h>

namespace MyFEM{
	typedef Eigen::SparseMatrix<double,Eigen::RowMajor> SparseMatrixD;  // according to https://eigen.tuxfamily.org/dox/TopicMultiThreading.html BiCGSTAB works in parallel with a row-major sparse format - ToDo: test if it is actually faster ...
	typedef Eigen::BiCGSTAB<SparseMatrixD > SparseSolver; //used for most examples
	//typedef Eigen::BiCGSTAB<SparseMatrixD, Eigen::IncompleteLUT<double> > SparseSolver; //could be faster according to Eigen docs -- probably only for big systems

	//typedef Eigen::SparseMatrix<double,Eigen::ColMajor> SparseMatrixD; // SparseLU wants col-major https://eigen.tuxfamily.org/dox/classEigen_1_1SparseLU.html
	//typedef Eigen::SparseLU<SparseMatrixD > SparseSolver; // probably slower but could be more accurate?

	//SparseLU and BiCG seem to be a lot more robust to numerically stiff problems / not so great conditioning ... also contact penalties and rotation invariant damping can have asymmetric matrices
	//typedef Eigen::SimplicialLDLT<SparseMatrixD > SparseSolver;

	// The following types can be used to map 3D vtkPoints and vtkCells (containing strictly tetra elements) to Eigen matrices (n x 3) and (m x 4) using the internal memory directly:
	//	CoordMatrix co ((double*)vtkPointsObject->GetVoidPointer(0), vtkPointsObject->GetNumberOfPoints(),0);
	//	TetraMatrix el ( vtkCellArrayObject->GetPointer()+1 ,vtkCellArrayObject->GetNumberOfCells(),0); // +1 because we need to skip the first element in the raw array (each row starts with the number of indices ==4 of the following element)
	// Note that vtkPoints uses float internally by default unless specified before inserting points!
	// Check if the data type is double before mapping: (vtkPointsObject->GetDataType()==VTK_DOUBLE)?
	typedef Eigen::Map<Eigen::Matrix<double,   3,Eigen::Dynamic,Eigen::ColMajor>                                        > Vect3Map;
	typedef Eigen::Map<Eigen::Matrix<vtkIdType,4,Eigen::Dynamic,Eigen::ColMajor>,Eigen::Unaligned,Eigen::OuterStride<5> > TetraMap;
	typedef Eigen::Map<Eigen::VectorXd                                                                                  > VectXMap;
	typedef Eigen::Map<Eigen::Matrix<vtkIdType,3,Eigen::Dynamic,Eigen::ColMajor>,Eigen::Unaligned,Eigen::OuterStride<4> > TriMap;
	typedef Eigen::Matrix<unsigned int,Eigen::Dynamic,1>                                                                  VectorXid;
	typedef Eigen::Map<VectorXid                                                                                        > VectIMap;
	// These types can be used to reference a block of the former accessing a single node or tet without copying data
	//	Vect3 pi = co.block<3,1>(0,i);
	//	Tetra ek = el.block<4,1>(0,k);
	typedef Eigen::Block<Vect3Map,3,1> Vect3;
	typedef Eigen::Block<TetraMap,4,1> Tetra;
	typedef Eigen::Block<TriMap,3,1>   Tri;

	enum UPDATE_MODE { REBUILD , UPDATE , SKIP }; // select whether to rebuild from scratch or update the stiffness or damping matrix -- use update if mesh topology has not changed since last call

	class VectorField{ // base class for vector fields, evaluates to f=0 everywhere
	public:
		// evaluate f(x0,x,t): (R^3 x R^3 x R) -> R^3, where x0 is the rest space (undeformed) coordinate and x is the world space (deformed) coordinate
		virtual void eval(Eigen::Vector3d& f, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {f.setZero();}
		virtual bool isDataAvailable(double t){return true;}
	};


	template <typename T>
	T defaultLBFGSoptions(){
		/*LBFGSpp::LBFGSParam<double>*/T optimOptions;
		optimOptions.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_WOLFE;
		optimOptions.wolfe = 1.0-1e-3;
		optimOptions.m = 5;
		optimOptions.epsilon = 1e-16;
		optimOptions.past = 1; /*0 == off, 1 == compare to previous iteration to detect if we got stuck ...*/
		optimOptions.delta = 1e-8; /*relative change of objective function to previous iteration below which we give up*/
		optimOptions.max_iterations = 150;
		return optimOptions;
	}
}

#endif