#include "Materials.h"
#include "LinearFEM.h"

using namespace MyFEM;

void HomogeneousIsotropicLinearElasticMaterial::computeEnergyStressAndHessian(Eigen::Matrix3d& F, double* params, double& energy, Vector9d& PK1stress, Matrix9d& H){
	// auto-generated code, writes "energy", "PK1stress", and "H", reads F,la,mu
	matrixComponentRefs(F);
	const double &la = params[0], &mu = params[1];
	energy = 0.0;
	PK1stress.setZero();
	H.setZero();
	#include "codegen_linElast.h"
	//printf("\n iso lin elast with la=%7.4lf and mu=%7.4lf \n", la,mu);
}
void HomogeneousIsotropicNeohookeanMaterial::computeEnergyStressAndHessian(Eigen::Matrix3d& F, double* params, double& energy, Vector9d& PK1stress, Matrix9d& H){
	// auto-generated code, writes "energy", "PK1stress", and "H", reads F,la,mu
	matrixComponentRefs(F);
	const double &la = params[0], &mu = params[1];
	energy = 0.0;
	PK1stress.setZero();
	H.setZero();
	#include "codegen_Neohookean.h"
}




void ViscosityModel::assembleForceAndDampingMatrix(Eigen::VectorXd& v, LinearFEM& fem, unsigned int bodyId, UPDATE_MODE mode){
	class ViscosityAssemblyOp{
	public:
		VectXMap vi; UPDATE_MODE mode;
		std::vector<Eigen::Triplet<double> > D_triplets;
		Eigen::Matrix3d dv;
		Vector9d stress; Matrix9d dStress;
		Eigen::Matrix<double, 12,1> force; Eigen::Matrix<double, 12,12> dForce;

		ViscosityAssemblyOp(Eigen::VectorXd& vi_, UPDATE_MODE mode_) : vi(vi_.data(),vi_.size()), mode(mode_) {}
		inline void initialize(LinearFEM& fem){
			if( fem.D.size()==0 ) mode=REBUILD; // prevent update if D is empty
		}
		inline void calculateElement(LinearFEM& fem, unsigned int k){
			if( fem.getViscosityModel(k)==NULL ){
				force.setZero(); dForce.setZero(); // no viscosity on this element
			}else{
				fem.computeVelocityGradient(dv,k,vi);
				fem.getViscosityModel(k)->computeStressAndDerivative(dv,fem.getViscosityParameters(k).data(),stress,dStress);
				force  = -fem.getVolume(k)*fem.getDFdx(k).transpose()* stress;
				if(  mode == REBUILD || mode == UPDATE ){
					dForce =  fem.getVolume(k)*fem.getDFdx(k).transpose()*dStress*fem.getDFdx(k);
				}
			}
		}
		inline void assembleVectorDOF(LinearFEM& fem, unsigned int k, unsigned int i, unsigned int idof, unsigned int lidof, unsigned int gidof){
			fem.f(gidof) += force(lidof);
		}
		inline void assembleMatrixDOF(LinearFEM& fem, unsigned int k, unsigned int i, unsigned int idof, unsigned int lidof, unsigned int gidof, unsigned int j, unsigned int jdof, unsigned int ljdof, unsigned int gjdof){
			if( mode == UPDATE ){
				fem.D.coeffRef(gidof,gjdof) += dForce(lidof,ljdof);
			}else if( mode == REBUILD ){
				D_triplets.push_back(
					Eigen::Triplet<double>(gidof,gjdof, dForce(lidof,ljdof) )
				);
			}
		}
		void finalize(LinearFEM& fem){
			if( mode == REBUILD ){ // build from triplets
				fem.D.resize(fem.N_DOFS*fem.getNumberOfNodes(), fem.N_DOFS*fem.getNumberOfNodes());
				fem.D.setFromTriplets(D_triplets.begin(), D_triplets.end());
			}
		}
	} viscop(v,mode);

	LinearFEM::assemblyLoop(fem, viscop);
}

void PowerLawViscosityModel::computeStressAndDerivative(Eigen::Matrix3d& dv, double* params, Vector9d& stress, Matrix9d& H){
	stress.setZero();
	H.setZero();
	if( dv.squaredNorm()>sqrt(DBL_MIN) ){	// auto generated code ...
		matrixComponentRefs(dv);
		const double &nu = params[0], &h = params[1];
		#include "codegen_PowerLawViscosity.h"
	}
}

void PowerSeriesViscosityModel::computeStressAndDerivative(Eigen::Matrix3d& dv, double* params, Vector9d& stress_out, Matrix9d& H_out){
	Vector9d stress; Matrix9d H;
	stress.setZero(); H.setZero();
	stress_out.setZero(); H_out.setZero();
	if( dv.squaredNorm()>sqrt(DBL_MIN) ){	// auto generated code ...
		matrixComponentRefs(dv);
		for(unsigned int i=0; i<getNumberOfParameters(); ++i){
			const double &nu = params[i], &h = powerIndices[i];
			#include "codegen_PowerLawViscosity.h"
			//printf("\n%% viscPS i=%d (nu=%.4lg, h=%.4lg) ",i,nu,h); //cout << stress.transpose() << " --> " << stress_out.transpose();
			stress_out += stress; H_out += H;
		}
	}
}

void RotationInvariantViscosityModel::computeStressAndDerivative(Eigen::Matrix3d& dv, double* params, Vector9d& stress, Matrix9d& H){
	stress.setZero();
	H.setZero();
	HF.setZero();
	if( dv.squaredNorm()>FLT_EPSILON && F.determinant()>DBL_EPSILON ){	// auto generated code ...
		matrixComponentRefs(F);
		const double
			&dv_1_1 = dv(0,0), &dv_1_2 = dv(0,1), &dv_1_3 = dv(0,2),
			&dv_2_1 = dv(1,0), &dv_2_2 = dv(1,1), &dv_2_3 = dv(1,2),
			&dv_3_1 = dv(2,0), &dv_3_2 = dv(2,1), &dv_3_3 = dv(2,2);

		const double &nu = params[0];
		#include "codegen_RotInvariantViscosity_pot.h"
	}
}
void RotationInvariantViscosityModel::assembleForceAndDampingMatrix(Eigen::VectorXd& v, LinearFEM& fem, unsigned int bodyId, UPDATE_MODE mode){
	class RotationInvariantViscosityAssemblyOp{
	public:
		VectXMap vi; UPDATE_MODE mode;
		std::vector<Eigen::Triplet<double> > D_triplets;
		Eigen::Matrix3d dv,F;
		Vector9d stress; Matrix9d dStress;
		Eigen::Matrix<double, 12,1> force; Eigen::Matrix<double, 12,12> dForce, dForceF;

		RotationInvariantViscosityAssemblyOp(Eigen::VectorXd& vi_, UPDATE_MODE mode_) : vi(vi_.data(),vi_.size()), mode(mode_) {}
		inline void initialize(LinearFEM& fem){
			if( fem.D.size()==0 ) mode=REBUILD; // prevent update if D is empty
			if( mode == UPDATE ){ // zero out coeffs but keep memory
				for(unsigned int k=0; k<fem.D.data().size(); ++k)  fem.D.data().value(k)=0.0;
			}
		}
		inline void calculateElement(LinearFEM& fem, unsigned int k){
			vtkSmartPointer<RotationInvariantViscosityModel> vm;
			if( vm=RotationInvariantViscosityModel::SafeDownCast( fem.getViscosityModel(k) )){
				fem.computeVelocityGradient(dv,k,vi);
				fem.computeDeformationGradient(F,k);
				vm->setCurrentDefGrad(F);
				//Eigen::Matrix<double, 12,1> v_el; for(int i=0;i<4;++i) v_el.segment<3>(3*i)=vi.segment<3>( 3*fem.getElement(k)(i) );
				//Vector9d dv_vect = fem.getDFdx(k)*v_el;
				//dv.row(0) = dv_vect.segment<3>(0);
				//dv.row(1) = dv_vect.segment<3>(3);
				//dv.row(2) = dv_vect.segment<3>(6); // same result as fem.computeVelocityGradient(dv,k,vi);

				vm->computeStressAndDerivative(dv,fem.getViscosityParameters(k).data(),stress,dStress);
				force  = -fem.getVolume(k)*fem.getDFdx(k).transpose()* stress;
				if(  mode == REBUILD || mode == UPDATE ){
					dForce =  fem.getVolume(k)*fem.getDFdx(k).transpose()*dStress*fem.getDFdx(k);
					dForceF = fem.getVolume(k)*fem.getDFdx(k).transpose()*(vm->getDstressDF())*fem.getDFdx(k);
				}
			}else{
				force.setZero(); dForce.setZero(); dForceF.setZero(); // no viscosity on this element
			}	//printf(" (%d|%.2lg)", k, force.norm());
		}
		inline void assembleVectorDOF(LinearFEM& fem, unsigned int k, unsigned int i, unsigned int idof, unsigned int lidof, unsigned int gidof){
			fem.f(gidof) += force(lidof);
		}
		inline void assembleMatrixDOF(LinearFEM& fem, unsigned int k, unsigned int i, unsigned int idof, unsigned int lidof, unsigned int gidof, unsigned int j, unsigned int jdof, unsigned int ljdof, unsigned int gjdof){
			if( mode == UPDATE ){
				fem.D.coeffRef(gidof,gjdof) += dForce(lidof,ljdof);
				fem.K.coeffRef(gidof,gjdof) += dForceF(lidof,ljdof);
			}else if( mode == REBUILD ){
				D_triplets.push_back(
					Eigen::Triplet<double>(gidof,gjdof, dForce(lidof,ljdof) )
				);
				if( fem.K.nonZeros()==0 ){ printf("\n%% WARNING: RotationInvariantViscosityModel requires stiffness to be assembled first");
				}else{ fem.K.coeffRef(gidof,gjdof) += dForceF(lidof,ljdof); }
			}
		}
		void finalize(LinearFEM& fem){
			if( mode == REBUILD ){ // build from triplets
				fem.D.resize(fem.N_DOFS*fem.getNumberOfNodes(), fem.N_DOFS*fem.getNumberOfNodes());
				fem.D.setFromTriplets(D_triplets.begin(), D_triplets.end());
			}
		}
	} viscop(v,mode);

	LinearFEM::assemblyLoop(fem, viscop);
}