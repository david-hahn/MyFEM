#ifndef DIRECTSENSITIVITY_H
#define DIRECTSENSITIVITY_H

#include "SensitivityAnalysis.h"

namespace MyFEM{
	/* Sensitivity analysis for a given objective function using the adjoint method
	 *
	 * We compute d(phi)/dq for a scalar objective function phi(xv,f,q,t) with respect to a parameter vector q
	 * Here x are nodal positions, v are nodal velocities, and "xv" denotes the concatenated state vector (x, v),
	 * f are internal forces, q are the design parameters, and t is time.
	 * As all of these are related through a FEM simulation, imposing a constraint of the form g(xv,xv_t,q,t)=0,
	 */

	class DirectSensitivity : public SensitivityAnalysis{
	public:
		DirectSensitivity(
			ObjectiveFunction& objective,
			ParameterHandler&  parameters,
			LinearFEM&         simulation
		) : SensitivityAnalysis(objective,parameters,simulation) {}
		virtual ~DirectSensitivity(){}

		// solve the static problem and return the objective function value and gradient wrt. parameters
		virtual double solveStaticFEM(Eigen::VectorXd& dphi_dq);

		virtual int dynamicImplicitTimeStep(unsigned int step, double timestep, double& phiVal, Eigen::VectorXd& dphi_dq, double eps=-1.0); // passes through return value from FEM solver, if eps<0 LinearFEM::FORCE_BALANCE_EPS will be used

		inline ObjectiveFunction& getObjectiveFunction(){return phi;}
		inline ParameterHandler& getParameterHandler(){return qHdl;}
		inline LinearFEM& getSimObject(){ return fem; }
		inline Eigen::MatrixXd& getDxDq(){return dx_dq;}
		inline Eigen::MatrixXd& getDvDq(){return dv_dq;}
		inline Eigen::MatrixXd& getHessian(){
			if( H.size()==0 ) printf("\n%% WARNING: HESSIAN NOT SET -- ONLY AVAILABLE FOR SOME OBJECTIVE FUNCTION TYPES\n"); // Hessian only available for BoundaryFieldObjectiveFunction at the moment (ToDo: generalize ...)
			return H;
		}

	protected:
		// direct sensitivities and Hessian
		Eigen::MatrixXd dx_dq, dv_dq, H;

		// additional storage for BDF2 time integration
		Eigen::MatrixXd dx_dq_old, dv_dq_old, H_old;

		virtual void dynamicSimStart();
		virtual void dynamicSimStaticInit();
	};
}

#endif