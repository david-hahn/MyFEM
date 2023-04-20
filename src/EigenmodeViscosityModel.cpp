#include "EigenmodeViscosityModel.h"
#include "LinearFEM.h"
#include "vtkPiecewiseFunction.h"
#include "../Spectra/SymEigsSolver.h"
#include "../Spectra/SymGEigsSolver.h"

using namespace MyFEM;

// "global" linear viscosity model where we only prepare the damping matrix once, and then just compute forces f_d = D*v leaving D unchanged
//  then build D from standard Newtonian viscosity (i.e. like linear elastic stiffness)
//  run modal analysis for the first n eigenmodes and adjust viscosity per eigen-frequency
//  higher modes all get the same viscosity (end of freqency range value) ... update D accordingly
typedef Spectra::SparseSymMatProd<double,Eigen::SelfAdjoint,SparseMatrixD::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor,SparseMatrixD::StorageIndex> Kprod;
typedef Spectra::SparseRegularInverse<double,Eigen::SelfAdjoint,SparseMatrixD::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor,SparseMatrixD::StorageIndex> Minv;
typedef Spectra::SymGEigsSolver<double, Spectra::SMALLEST_MAGN,Kprod,Minv,Spectra::GEIGS_REGULAR_INVERSE> EigSolver;

void EigenmodeViscosityModel::initializeToConstantViscosity(double viscosity, LinearFEM& fem, unsigned int bodyId){
	nu_fcn = vtkSmartPointer<vtkPiecewiseFunction>::New();
	nu_fcn->ClampingOn();
	nu_fcn->AllowDuplicateScalarsOff();

	eigvals.resize(0); eigvecs.resize(0,0); // force recomputation of eigenmodes

	nu = viscosity;
	Eigen::VectorXd v( fem.v.size() ); v.setZero();
	fem.viscosityModel[bodyId]=this;
	assembleForceAndDampingMatrix(v, fem, bodyId, REBUILD);
}


void EigenmodeViscosityModel::getViscosityCoefficients(Eigen::VectorXd& c){
	c.resize( nu_fcn->GetSize() );
	for(int i=0; i<c.size(); ++i) c(i) = nu_fcn->GetValue( frqOfPoints(i) );
}
void EigenmodeViscosityModel::setViscosityCoefficients(const Eigen::VectorXd& c, LinearFEM& fem){
	bool changed=false;
	if( c.size() == nu_fcn->GetSize() ){
		for(int i=0; i<c.size(); ++i){
			if( nu_fcn->GetValue( frqOfPoints(i) ) != c(i) ){
				nu_fcn->AddPoint( frqOfPoints(i), c(i) ); // updates the value for i-th frequency to c(i)
				if( i==(c.size()-1) ) nu = c(i); // use the coefficient of the highest frequency as base viscosity for all higher modes
				changed=true;
			}
		}
		if( changed ){
			// now rebuild the damping matrix
			Eigen::VectorXd v( fem.v.size() ); v.setZero();
			assembleForceAndDampingMatrix(v, fem, 0, REBUILD); // not using body-ID here as the model is already supposed to be registered with the FEM object
		}
	}else{ printf("\n%% WRONG INPUT SIZE, IGNORED EigenmodeViscosityModel::setViscosityCoefficients \n");}
}

void EigenmodeViscosityModel::computeStressAndDerivative(Eigen::Matrix3d& dv, double* params, Vector9d& stress, Matrix9d& H){
	stress.setZero();
	H.setZero();
	matrixComponentRefs(dv);
	// reads nu already defined as member variable
	#include "codegen_NewtonianViscosity.h"
}

// this is a 3-in-1 function
// if mode==UPDATE, fem.D remains unchanged and the forces are set to -fem.D*v
// if mode==REBUILD and eigenmodes are already stored -> fem.D is adjusted according to damping coefficients in nu_fcn
// if mode==REBUILD and no eigenmodes are stored, compute them and reset nu_fcn to constant viscosity, fem.D will be assembled for standard Newtonian viscosity
void EigenmodeViscosityModel::assembleForceAndDampingMatrix(Eigen::VectorXd& v, LinearFEM& fem, unsigned int bodyId, UPDATE_MODE mode){
	if( fem.D.size()==0 || mode==REBUILD){
		if( fem.D.size()==0 ){ // the damping matrix has been cleared - rebuild everything including eigenmodes
			eigvals.resize(0); eigvecs.resize(0,0); // force recomputation of eigenmodes
		}
		bool haveAdjustments = (nu_fcn->GetSize()>0);
		if( haveAdjustments ){ // we already have adjustment data
			double nu_range[2];
			nu_fcn->GetRange(nu_range);
			nu = nu_fcn->GetValue(nu_range[1]); // set default viscosity to value for highest modal frequency
		}

		// first run the standard assembly routine ...
		Eigen::VectorXd f_tmp = fem.f, f_ext_tmp = fem.f_ext, x_tmp = fem.x;
		double nu_tmp = nu; nu=1.0; // assemble for unit viscosity, multiply later
		ViscosityModel::assembleForceAndDampingMatrix(v,fem,bodyId,mode);
		fem.deformedCoords = fem.restCoords;
		fem.assembleForceAndStiffness(mode); // assemble stiffness for undeformed state -> linear stiffness
		if( fem.M.size()==0 ) fem.assembleMassAndExternalForce(); // we'll also need an up-to-date mass matrix ...
		fem.f = f_tmp; // restore forces, we don't want them right now ...
		fem.f_ext = f_ext_tmp;
		fem.x = x_tmp; // also restore deformation
		nu = nu_tmp;

		ratioDtoK = fem.D.diagonal().sum() / fem.K.diagonal().sum(); // ratioDtoK is 1/nu times the first Rayleigh coefficient, the other one (mass term) is always zero
		fem.D = nu * ratioDtoK * fem.K; // set D to Rayleigh stiffness damping

		if( eigvals.size()==0 ){ // we don't have eigenmodes yet, assume we're initializing, compute eigenmodes and set nu_fcn to constant viscosity
			Kprod K(fem.K);
			Minv M(fem.M);
			EigSolver eigs( &K, &M, nModes, K.cols() );
			eigs.init();
			nModesComputed = eigs.compute() -6; // skip the last 6 entries in the result
			//// for eigenvectors we have: v'K*v == w^2  and v'*M*v == 1, and zero for pairs of different eigenvectors
			//// as we have not applied any Dirichlet boundary conditions, the smallest 6 eigenvalues (zero up to numerics) correspond to rigid transformation modes
			//// eigs should report them in order largest to smallest, so it should be safe to simply skip the last 6 entries
			double highestFrq = sqrt(eigs.eigenvalues()(0)); // the highest frequency spanned by the computed modes
			double lowestFrq = sqrt(eigs.eigenvalues()(nModesComputed-1)); // the highest frequency spanned by the computed modes
			//printf("\n%% frequency range %.4lg - %.4lg rad/s in %d modes (D/K ratio %.4lg) ", lowestFrq, highestFrq, nModesComputed, ratioDtoK );
			//cout << endl << eigs.eigenvalues().transpose() << endl;

			if(!haveAdjustments ){ //initialize the adjustment function
				nu_fcn->Initialize(); // reset the adjustment function
				frqOfPoints.resize( nNuPoints ); frqOfPoints.setLinSpaced(lowestFrq, highestFrq);
				for(int i=0; i<nNuPoints; ++i){
					nu_fcn->AddPoint(frqOfPoints(i), nu);
				}
			}

			eigvals = eigs.eigenvalues();
			eigvecs = eigs.eigenvectors();

			//// debug output ...
			//Eigen::VectorXd t( nModes ); t.setLinSpaced(lowestFrq, highestFrq);
			//printf("\n nu_fcn = [ ");
			//for(int i=0; i<t.size(); ++i)
			//	printf("\n %.4lg   %.4lg", t(i), nu_fcn->GetValue(t(i)));
			//printf(" ];\n");

		}
		if( haveAdjustments ){ //update D (currently assembled to standard Newtonian viscosity) with coefficients in nu_fcn ...
			//printf("nnz in D %u ", fem.D.nonZeros());
			Eigen::MatrixXd tmp( fem.D.rows(), fem.D.cols() ); tmp.setZero();
			for(int i=0; i<nModesComputed; ++i){ // adjust damping for mode i -- i==0 corresponds to the highest frequency mode we computed, i==n-1 to the lowest possible frequency (not counting rigid modes)
				//printf("\n%% efrq %d: %.4lg, ev-norm %.4lg", i, sqrt(eigs.eigenvalues()(i)), eigs.eigenvectors().col(i).norm());
				Eigen::VectorXd evn = fem.M*eigvecs.col(i);
				double frq = sqrt(eigvals(i));
				//printf("\n%% dFactor = nu_fcn ( frq ) / nu = %.4lg (%.4lg) / %.4lg ", nu_fcn->GetValue( frq ), frq, nu);
				double dFactor = nu_fcn->GetValue( frq ) / nu; // ratio of requested viscosity for this frequency to base viscosity value
				tmp += (dFactor-1.0)*nu*ratioDtoK*eigvals(i)* (evn*evn.transpose());
				//tmp += ( nu_fcn->GetValue( frq ) - nu )*ratioDtoK*eigvals(i)* (evn*evn.transpose());
				//fem.D += tmp; // figure out how to do this properly with sparse matrices ...
				//SparseMatrixD st ( tmp.sparseView() ); 
				//fem.D += st; // this sort of ruins the sparsity of D and makes things super slow -- maybe better keep the adjustment separate? but what about the derivatives needed in the newton solver???
				// use the Woodbury formula? --- See Barbic et al 2017, eq (26) --- or Wikipedia --- could introduce aux. variable y and solve [S V ; V^T -I]\[v; y]=[f; 0], where V*V^T=tmp above
			}
			tmp += fem.D;
			//double refVal = fem.D.diagonal().sum() / fem.D.rows();
			fem.D = tmp.sparseView();//(refVal,1e-3); // if we use eps 1e-16, D will be fully dense, at 1e-4 we save about 1/3, at 1e-3 we see serious numerical artefacts
			//printf("now %u ", fem.D.nonZeros());
			// ...
		}
	}
	// finally add the damping forces from the adjusted matrix ...
	fem.f -= fem.D * v; // damping force is -D*v
}
