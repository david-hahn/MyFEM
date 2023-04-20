#ifndef SENSITIVITYANALYSIS_H
#define SENSITIVITYANALYSIS_H

#include "types.h"

namespace MyFEM{
	class LinearFEM;
	class MeshPreviewWindow;

	/*
	 * SensitivityAnalysis: abstract base class for either adjoint or direct sensitivity analysis
	 * ObjectiveFunction:   defines the objective function and computes its partial derivatives
	 * ParameterHandler:    manages how design parameters are communicated to the simulation object and computes partial derivatives of internal forces
	 */

	// Create subclasses of ObjectiveFunction and ParameterHandler to adapt functionality ...
	class ObjectiveFunction{
	public:
		ObjectiveFunction(){ uTarget.setZero(); }
		// define a scalar objective function phi(x,v,f,q,t) --> R
		// returns the objective function value and writes partial derivatives \partial phi / \partial x = phi_x, similar for phi_v, phi_f, and phi_q
		virtual double evaluate( LinearFEM& fem,
			Eigen::VectorXd& phi_x, Eigen::VectorXd& phi_v, Eigen::VectorXd& phi_f, Eigen::VectorXd& phi_q
		); // example implementation in AdjointSensitivity.cpp
		virtual void reset( LinearFEM& fem ){} // should be called once at the start of a dynamic sim to allow for initialization tasks in subclasses
		Eigen::Vector3d uTarget;
	};
	class ParameterHandler{
	public:
		const unsigned int bodyId; // the body of the FEM mesh on which this parameter handler operates
		ParameterHandler(unsigned int bodyId_) : bodyId(bodyId_) {}
		virtual bool useSparseDgDq(){return false;} // override to use sparse format for g_q in computeConstraintDerivatives
		virtual void getCurrentParams(  Eigen::VectorXd& q, LinearFEM& fem){} // writes q, does not change fem
		virtual void setNewParams(const Eigen::VectorXd& q, LinearFEM& fem){} // writes to fem, does not change q
		virtual unsigned int getNumberOfParams(const LinearFEM& fem){return 0;} // how many parameters are we working with (rows in q)
		virtual unsigned int getNumberOfDOFs(const LinearFEM& fem); // how many unknowns are there in the sim  (rows in x, v, f) -- implementation in AdjointSensitivity.cpp
		virtual double computeConstraintDerivatives(Eigen::MatrixXd& g_q, Eigen::VectorXd& phiQ_q, LinearFEM& fem){return 0.0;}; // partial derivatives of the constraint function wrt. the design parameters + regularizer term phiQ(q) and dphiQ_dq
		virtual double computeConstraintDerivatives(SparseMatrixD&   g_q, Eigen::VectorXd& phiQ_q, LinearFEM& fem){return 0.0;}; // use this version for sparse output format

		static double regFcn(double qi, double& dr_dqi, double& d2r_dqi2, double rc = 0.0, double alpha=1e-5){ // a soft regularizer function to push parameters to positive values
			if( qi >= rc ) return 0.0;
			double s = rc-qi; // > 0.0 or we returned already
			d2r_dqi2 += 2.0*alpha*s;// linear curvature d^2 r / dqi^2 = d(-a*s^2)/ds * ds/dqi = (-2as) *(-1)
			dr_dqi -= alpha*s*s;    // quadratic gradient dr/dqi = dr/ds * ds/dqi = a*s^2 *(-1)
			return alpha/3.0*s*s*s; // cubic regularizer function  r = a/3*s^3, s=rc-qi
		}
		static double barFcn(double qi, double& dr_dqi, double& d2r_dqi2, double rc = 0.0, double alpha=1e-5){ // a log-barrier function to ensure positive parameter values
			if( qi <= rc ) return DBL_MAX;
			if( qi-rc > 1.0 ) return 0.0;
			double s = qi-rc; // > 0.0 here
			d2r_dqi2 += alpha/s/s;
			dr_dqi -= alpha/s;
			return -alpha*log(s);
		}

		//ToDo: add a verifyParameters function that allows for range checks of parameters, so we don't run simulations for useless values
		// experimental for Gauss-Newton solvers: compute (diagonal of) phiQ hessian (i.e. second derivatives of regularization functions)
		Eigen::VectorXd phiQ_qq; // by default left empty!
	};

	class SensitivityAnalysis{
	public:
		SensitivityAnalysis(
			ObjectiveFunction& objective,
			ParameterHandler&  parameters,
			LinearFEM&         simulation
		) : 
			phi(objective), qHdl(parameters), fem(simulation)
		{
			timestep = 0.0; numberOfTimesteps = 0;
			solverEps = -1; // default from LinearFEM will be used instead
			preReg = 0.0;
			maxPreIters = 0; preEps = solverEps; // these are not used if preReg is zero
			outFileName=""; wnd=NULL; evalCounter=0; t0=0.0;
			bestRunPhiValue=DBL_MAX;
		}
		virtual ~SensitivityAnalysis(){}

		// run a simulation and compute objective function value and gradient from given parameters params
		virtual double operator()(const Eigen::VectorXd& params, Eigen::VectorXd& objectiveFunctionGradient);

		virtual double finiteDifferenceTest(const Eigen::VectorXd& params, const double fd_h=1e-6);

		virtual void setupDynamicSim( double timestep, unsigned int numberOfTimesteps, bool startWithStaticSolve=false, std::string outFileName="", MeshPreviewWindow* wnd=NULL, double solverEps=-1.0 ); // leave outFileName empty to skip writing output files and wnd==NULL to skip preview render, leave solverEps<0.0 to use default from LinearFEM		// solve the dynamic problem and return the objective function value and gradient wrt. parameters
		virtual void setupStaticPreregularizer(double regularize, unsigned int maxPreIters, double preEps);
		virtual double getSimulatedTime(){ return timestep * ((double)numberOfTimesteps); }
		virtual double solveImplicitDynamicFEM(Eigen::VectorXd& dphi_dq);
		virtual int    dynamicImplicitTimeStep(unsigned int step, double timestep, double& phiVal, Eigen::VectorXd& dphi_dq, double eps=-1.0)=0; // passes through return value from FEM solver, if eps<0 LinearFEM::FORCE_BALANCE_EPS will be used

		// solve the static problem and return the objective function value and gradient wrt. current simulation parameters
		virtual double solveStaticFEM(Eigen::VectorXd& dphi_dq)=0;

		inline void resetEvalCounter(){ evalCounter=0; }
		inline unsigned int getEvalCounter(){ return evalCounter; }

		double bestRunPhiValue;
		Eigen::VectorXd bestRunParameters, bestRunGradient;

	protected:
		// the FEM object defining the forward simulation
		LinearFEM& fem;

		// objective function and parameter handler
		ObjectiveFunction& phi;
		ParameterHandler&  qHdl;

		// dynamic sim properties
		double timestep; unsigned int numberOfTimesteps;
		// static pre-regularization settings
		double preEps, preReg; unsigned int maxPreIters;
		// flags and stuff
		std::string outFileName; MeshPreviewWindow* wnd;
		double solverEps;
		bool startWithStaticSolve;
		unsigned int evalCounter; // count how many times operator() has been called
		double t0; // omp_get_wtime at first operator() call after evalCounter reset for timings
		double phiQ; // for output/debug store the regularizer contribution to the objective function value internally

		virtual void dynamicSimStart(){} // can be used in subclasses for initialization tasks at the start of a dynamic simulation
		virtual void dynamicSimStaticInit(){ phi.reset(fem); } // similarly, used for computations when starting from a static solve (startWithStaticSolve==true)
		virtual void dynamicSimFinalize(double& phiVal, Eigen::VectorXd& dphi_dq){} // can be used in subclasses for wrapping things up at the end of a dynamic simulation
	};
}

#endif