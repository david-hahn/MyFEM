#ifndef MATERIALS_H
#define MATERIALS_H

#include "types.h"
#include <vtkObject.h>
#include <vtkSetGet.h>
#include <vtkSmartPointer.h>

// For vectorized stress and stress derivative wrt. deformation gradient
typedef Eigen::Matrix<double,9,1> Vector9d;
typedef Eigen::Matrix<double,9,9> Matrix9d;

#define matrixComponentRefs(F) \
	const double \
		&f_1_1 = F(0,0), &f_1_2 = F(0,1), &f_1_3 = F(0,2), \
		&f_2_1 = F(1,0), &f_2_2 = F(1,1), &f_2_3 = F(1,2), \
		&f_3_1 = F(2,0), &f_3_2 = F(2,1), &f_3_3 = F(2,2); // deformation gradient in MuPad notation

namespace MyFEM{
	class LinearFEM;
	class DifferentiableSpline;

	class HomogeneousMaterial : public vtkObject{
	public:
		vtkTypeMacro(HomogeneousMaterial, vtkObject)
		// implement this function to compute the material response to a given deformation such that it
		// takes the deformation gradient
		// and material parameters in a double array of size getNumberOfParameters(), then
		// writes the strain energy density into energy
		// writes the stress (all 9 components of a symmetric 3x3 matrix) into stress(0) to stress(8) in order (xx,xy,xz,yx,yy,yz,zx,zy,zz), and
		// writes the energy hessian = stress derivative wrt. deformation into hessian (rows as in stress vector, columns in order (xx,xy,xz,yx,yy,yz,zx,zy,zz) ~> components of F)
		virtual void computeEnergyStressAndHessian(Eigen::Matrix3d& F, double* params, double& energy, Vector9d& stress, Matrix9d& hessian)=0;

		virtual void setElasticParams(double* params, double lameLambda=0.0, double lameMu=1.0)=0;
		virtual void setDensity(double* params, double density=0.1)=0;

		virtual void getElasticParams(double* params, double& lameLambda, double& lameMu)=0;
		virtual double getDensity(double* params)=0;

		virtual unsigned int getNumberOfParameters()=0;
	};

	class HomogeneousIsotropicLinearElasticMaterial : public HomogeneousMaterial{
	public:
		vtkTypeMacro(HomogeneousIsotropicLinearElasticMaterial, HomogeneousMaterial)
		static HomogeneousIsotropicLinearElasticMaterial* New(){ return new HomogeneousIsotropicLinearElasticMaterial(); } // for compatibility with vtkSmartPointer
		virtual void Delete(){} // for compatibility with vtkSmartPointer
		virtual ~HomogeneousIsotropicLinearElasticMaterial(){Delete();}

		virtual void computeEnergyStressAndHessian(Eigen::Matrix3d& F, double* params, double& energy, Vector9d& PK1stress, Matrix9d& H);

		virtual inline void setElasticParams(double* params, double lameLambda=0.0, double lameMu=1.0){
			params[0]=lameLambda; params[1]=lameMu;
		}
		virtual inline void getElasticParams(double* params, double& lameLambda, double& lameMu){
			lameLambda=params[0]; lameMu=params[1];
		}
		virtual inline void   setDensity(double* params, double density=0.1){params[2]=density;}
		virtual inline double getDensity(double* params){return params[2];}

		virtual inline unsigned int getNumberOfParameters(){return 3;};
	};

	class HomogeneousIsotropicNeohookeanMaterial : public HomogeneousIsotropicLinearElasticMaterial{
	public:
		vtkTypeMacro(HomogeneousIsotropicNeohookeanMaterial, HomogeneousIsotropicLinearElasticMaterial)
		static HomogeneousIsotropicNeohookeanMaterial* New(){ return new HomogeneousIsotropicNeohookeanMaterial(); } // for compatibility with vtkSmartPointer
		virtual void Delete(){} // for compatibility with vtkSmartPointer
		virtual ~HomogeneousIsotropicNeohookeanMaterial(){Delete();}

		virtual void computeEnergyStressAndHessian(Eigen::Matrix3d& F, double* params, double& energy, Vector9d& PK1stress, Matrix9d& H);
	};


	// this is a globally homogeneous elastic material, where the stress-strain relation is defined by functions of principal stretches
	// the only per-element parameter is density, getElasticParams always returns zeros.
	// setElasticParams(...) initializes the stress-strain functions to be equivalent to a Neohookean material (ToDo!)
	// implementation in PrincipalStretchMaterial.cpp
	class PrincipalStretchMaterial : public HomogeneousIsotropicLinearElasticMaterial{
	public:
		vtkTypeMacro(PrincipalStretchMaterial, HomogeneousIsotropicLinearElasticMaterial)
		static PrincipalStretchMaterial* New(){ return new PrincipalStretchMaterial(); } // for compatibility with vtkSmartPointer
		virtual void Delete(){} // for compatibility with vtkSmartPointer
		virtual ~PrincipalStretchMaterial(){Delete();}

		virtual void computeEnergyStressAndHessian(Eigen::Matrix3d& F, double* params, double& energy, Vector9d& PK1stress, Matrix9d& H);

		virtual void setElasticParams(double* params, double lameLambda=0.0, double lameMu=1.0);
		virtual inline void getElasticParams(double* params, double& lameLambda, double& lameMu){ lameLambda=0.0; lameMu=0.0; }
		virtual inline void   setDensity(double* params, double density=0.1){params[0]=density;}
		virtual inline double getDensity(double* params){return params[0];}

		virtual inline unsigned int getNumberOfParameters(){return 1;};

		vtkSmartPointer<DifferentiableSpline> fPrime, hPrime; // stress-strain curves for uniaxial and triaxial stretch
		Eigen::VectorXd xp; // points on x-axis of fPrime and hPrime
	};


	class ViscosityModel : public vtkObject{
	public:
		vtkTypeMacro(ViscosityModel, vtkObject)
		// implement this function to compute the material response to a given deformation rate such that it
		// takes the velocity gradient dv
		// and material parameters in a double array of size getNumberOfParameters(), then
		// writes the stress (all 9 components of a symmetric 3x3 matrix) into stress(0) to stress(8) in order (xx,xy,xz,yx,yy,yz,zx,zy,zz), and
		// writes stress derivative wrt. the velocity gradient into dStress (rows as in stress vector, columns in order (xx,xy,xz,yx,yy,yz,zx,zy,zz))
		virtual void computeStressAndDerivative(Eigen::Matrix3d& dv, double* params, Vector9d& stress, Matrix9d& dStress)=0;
		virtual unsigned int getNumberOfParameters()=0;
		// run the default assembly loop for viscous force and damping matrix
		//ToDo: not tested for multi-body meshes yet - might need to combine sparse matrix data in REBUILD mode ...
		virtual void assembleForceAndDampingMatrix(Eigen::VectorXd& v, LinearFEM& fem, unsigned int bodyId, UPDATE_MODE mode=UPDATE);
	};

	class PowerLawViscosityModel : public ViscosityModel{
	public:
		vtkTypeMacro(PowerLawViscosityModel, ViscosityModel)
		static PowerLawViscosityModel* New(){ return new PowerLawViscosityModel(); } // for compatibility with vtkSmartPointer
		virtual void Delete(){} // for compatibility with vtkSmartPointer
		virtual ~PowerLawViscosityModel(){Delete();}

		virtual void computeStressAndDerivative(Eigen::Matrix3d& dv, double* params, Vector9d& stress, Matrix9d& H);

		virtual void setViscosity(double* params, double  viscosity=0.0){params[0]=viscosity;}
		virtual void setPowerLawH(double* params, double  powerLawH=1.0){params[1]=powerLawH;}

		virtual void getViscosity(double* params, double& viscosity    ){viscosity = params[0];}
		virtual void getPowerLawH(double* params, double& powerLawH    ){powerLawH = params[1];}

		virtual unsigned int getNumberOfParameters(){return 2;}
	};

	class PowerSeriesViscosityModel : public ViscosityModel{
	public:
		vtkTypeMacro(PowerSeriesViscosityModel, ViscosityModel)
		static PowerSeriesViscosityModel* New(){ return new PowerSeriesViscosityModel(); } // for compatibility with vtkSmartPointer
		virtual void Delete(){} // for compatibility with vtkSmartPointer
		virtual ~PowerSeriesViscosityModel(){Delete();}

		virtual void computeStressAndDerivative(Eigen::Matrix3d& dv, double* params, Vector9d& stress, Matrix9d& H);

		virtual void setViscosityCoeffs(double* params, double* viscosityCoeffs){
			for(unsigned int i=0; i<getNumberOfParameters(); ++i) params[i]=viscosityCoeffs[i];
		}

		virtual unsigned int getNumberOfParameters(){return powerIndices.size();}

		Eigen::VectorXd powerIndices;
	};

	class RotationInvariantViscosityModel : public ViscosityModel{
	public:
		vtkTypeMacro(RotationInvariantViscosityModel, ViscosityModel)
		static RotationInvariantViscosityModel* New(){ return new RotationInvariantViscosityModel(); } // for compatibility with vtkSmartPointer
		virtual void Delete(){} // for compatibility with vtkSmartPointer
		virtual ~RotationInvariantViscosityModel(){Delete();}

		virtual void computeStressAndDerivative(Eigen::Matrix3d& dv, double* params, Vector9d& stress, Matrix9d& H);
		virtual inline Matrix9d& getDstressDF(){ return HF; }

		virtual void setViscosity(double* params, double  viscosity=0.0){params[0]=viscosity;}
		virtual void getViscosity(double* params, double& viscosity    ){viscosity = params[0];}
		virtual void setTimestep(double dt){} // deprecated -- ToDo: remove
		virtual void setCurrentDefGrad(Eigen::Matrix3d& F_){F=F_;}

		virtual unsigned int getNumberOfParameters(){return 1;}

		virtual void assembleForceAndDampingMatrix(Eigen::VectorXd& v, LinearFEM& fem, unsigned int bodyId, UPDATE_MODE mode=UPDATE);
	protected:
		//double timestep;
		Eigen::Matrix3d F;
		Matrix9d HF;
	};
}

#endif