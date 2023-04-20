#ifndef EIGENMODEVISCOSITYMODEL_H
#define EIGENMODEVISCOSITYMODEL_H

#include "Materials.h"
#include "vtkSmartPointer.h"
class vtkPiecewiseFunction; // fwd decl


namespace MyFEM{
	class EigenmodeViscosityModel : public ViscosityModel{
	public:
		vtkTypeMacro(EigenmodeViscosityModel, ViscosityModel)
		static EigenmodeViscosityModel* New(){ return new EigenmodeViscosityModel(); } // for compatibility with vtkSmartPointer
		EigenmodeViscosityModel(){ nModes=100; nNuPoints=5; }
		virtual void Delete(){/*printf("\n%% EigenmodeViscosityModel::Delete called\n");*/} // for compatibility with vtkSmartPointer
		virtual ~EigenmodeViscosityModel(){Delete();}

		// here we compute standard Newtonian viscosity from internally stored parameters
		virtual void computeStressAndDerivative(Eigen::Matrix3d& dv, double* params, Vector9d& stress, Matrix9d& H);
		
		// if mode=REBUILD, assemble standard Newtonian damping matrix, compute and store eigenmodes
		// if mode=REBUILD, and eigenmodes are already stored update damping matrix with new viscosity coefficients
		// if mode=UPDATE, simply compute fem.D*v and leave the matrix unchanged
		virtual void assembleForceAndDampingMatrix(Eigen::VectorXd& v, LinearFEM& fem, unsigned int bodyId, UPDATE_MODE mode=UPDATE);

		virtual void setNumberOfModes(unsigned int nModes_){nModes=nModes_;}
		virtual void setNumberOfAdjustmentPoints(unsigned int nNuPoints_){nNuPoints = nNuPoints_;}
		virtual void initializeToConstantViscosity(double viscosity, LinearFEM& fem, unsigned int bodyId);
		virtual void getAdjustmentFrequencies(Eigen::VectorXd& f){ f = frqOfPoints; }
		virtual void getViscosityCoefficients(Eigen::VectorXd& c);
		virtual void setViscosityCoefficients(const Eigen::VectorXd& c, LinearFEM& fem);
		//ToDo: add methods to access/modify frequency-damping params ...

		virtual unsigned int getNumberOfParameters(){return 0;} // no per-element parameters
		virtual unsigned int getNumberOfComputedModes(){return nModesComputed;} // may be less than nModes requested - typically by 6 rigid modes that are skipped, but it could be even less if the Eigensolver did not converge properly
		virtual double getBaseViscosity(){return nu;}
		virtual double getBaseDampingRatio(){return ratioDtoK;}

		virtual const Eigen::MatrixXd& getEigenmodes()  const {return eigvecs;}
		virtual const Eigen::VectorXd& getEigenvalues() const {return eigvals;}

		unsigned int nModes, nNuPoints;
	protected:
		vtkSmartPointer<vtkPiecewiseFunction> nu_fcn; // adjustment of damping per frequency -- using a piecewise linear function as VTK splines don't easily provide derivatives of function, or derivatives of coefficents wrt. control point values
		double nu; // used for initial assembly (viscosity of modes exceeding nModes)
		unsigned int nModesComputed; // store how many modes we've actually computed (may be less than requested nModes, skipping rigid modes)
		Eigen::MatrixXd eigvecs;
		Eigen::VectorXd eigvals;
		Eigen::VectorXd frqOfPoints;
		double ratioDtoK;
	};
}

#endif