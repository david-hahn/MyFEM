#ifndef ADJOINTSENSITIVITY_H
#define ADJOINTSENSITIVITY_H

#include "SensitivityAnalysis.h"
#include <set>
#include <map>

namespace MyFEM{
	class MeshPreviewWindow;

	/* Sensitivity analysis for a given objective function using the adjoint method
	 *
	 * We compute d(phi)/dq for a scalar objective function phi(xv,f,q,t) with respect to a parameter vector q
	 * Here x are nodal positions, v are nodal velocities, and "xv" denotes the concatenated state vector (x, v),
	 * f are internal forces, q are the design parameters, and t is time.
	 * As all of these are related through a FEM simulation, imposing a constraint of the form g(xv,xv_t,q,t)=0,
	 * we can apply the adjoint method to compute d(phi)/dq more efficiently. (Here xv_t denotes the time-derivative of the state xv.)
	 *
	 * In the adjoint method, we'll need three types of partial derivatives:
	 * - derivs. of the objective function wrt. its arguments -> to be computed by the objective function object
	 * - derivs. of the constraint wrt. the design parameters (basically change of internal forces wrt. parameters) -> to be computed by the parameter handler object
	 * - derivs. of the constraint wrt. the state (basically mass and stiffness matrices) -> already available within the FEM simulation object
	 *
	 * AdjointSensitivity drives both the forward simulation and adjoint computation and stores the required data
	 */

	class AdjointSensitivity : public SensitivityAnalysis{
	public:
		AdjointSensitivity(
			ObjectiveFunction& objective,
			ParameterHandler&  parameters,
			LinearFEM&         simulation,
			bool assumeSymmetricMatrices_ = true
		) : SensitivityAnalysis(objective,parameters,simulation) { assumeSymmetricMatrices=assumeSymmetricMatrices_; }
		virtual ~AdjointSensitivity(){}

		// solve the static problem and return the objective function value and gradient wrt. parameters
		virtual double solveStaticFEM(Eigen::VectorXd& dphi_dq);

		virtual int dynamicImplicitTimeStep(unsigned int step, double timestep, double& phiVal, Eigen::VectorXd& dphi_dq, double eps=-1.0); // passes through return value from FEM solver, if eps<0 LinearFEM::FORCE_BALANCE_EPS will be used
		void dynamicImplicitAdjointStep(unsigned int step, double& phiVal, Eigen::VectorXd& dphi_dq);

		inline ObjectiveFunction& getObjectiveFunction(){return phi;}
		inline ParameterHandler& getParameterHandler(){return qHdl;}
		inline LinearFEM& getSimObject(){ return fem; }

		inline void setAssumeSymmetricMatrices( bool value ){ assumeSymmetricMatrices=value; }

	protected:
		bool assumeSymmetricMatrices; // for most standard FEM models all matrices will be symmetrc - skip transpositions if flag is set (default true)
		// storage for adjoint computation
		std::map<unsigned int, double> dts, phiVals; // timestep durations
		std::map<unsigned int, SparseMatrixD > Ks, Ms, Ds, Rs; //ToDo: unless we remesh, M will remain unchanged -> store only once; similarly for linear elasticity K will be constant ... figure out storage handling ...
		std::map<unsigned int, Eigen::MatrixXd> g_qs; //we treat all parameters as one long vector here - a ParameterHandler must map them to FEM parameters (materials, boundaries, etc ...)
		std::map<unsigned int, SparseMatrixD> spg_qs; //as above if parameter handler uses sparse format instead
		std::map<unsigned int, std::set<unsigned int> > stickDOFs;

		SparseMatrixD   staticK, static_spg_q;
		Eigen::MatrixXd static_g_q;
		Eigen::MatrixXd dx0_dq, dv0_dq; //direct initial condition derivatives, ToDo: could handle ICs as implicit function g0(x0,v0,q)=0 and take partial derivs here instead ...


		std::map<unsigned int, Eigen::VectorXd> phi_xs, phi_vs, phi_fs, phi_qs;
		Eigen::VectorXd lambdaX, lambdaV; // adjoint states
		Eigen::VectorXd lambdaX_old, lambdaV_old; // old values for BDF2 time integration

		virtual void dynamicSimStart();
		virtual void dynamicSimStaticInit();
		virtual void dynamicSimFinalize(double& phiVal, Eigen::VectorXd& dphi_dq);
	};
}

#endif