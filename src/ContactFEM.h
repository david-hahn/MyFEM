#ifndef CONTACTFEM_H
#define CONTACTFEM_H

#include "LinearFEM.h"
#include <set>
#include <vector>

namespace MyFEM{
	class DiffableScalarField;
	class QPSolver;

	class ContactFEM : public LinearFEM{
	public:
		static double FORCE_BALANCE_DELTA; // negative = disabled, otherwise stop solvers if residual delta between iterations is less than specified threshold
		static bool printConvergenceDebugInfo; // default disabled, more solver debug output ...
		enum CONTACT_METHOD { CONTACT_IGNORE , CONTACT_CLAMP_PENALTY , CONTACT_TANH_PENALTY , CONTACT_HYBRID , CONTACT_QP=91 };
		enum CONTACT_STATE { CONTACT_NONE , CONTACT_STICK , CONTACT_SLIP };

		ContactFEM();
		virtual ~ContactFEM();

		int addRigidObstacle(const DiffableScalarField& g, double frictionCoeffient);
		inline void clearRigidObstacles(){rigidObstacles.clear(); frictionCoefficient.clear();}
		inline void setPenaltyFactors(double normalPenalty, double tangentialPenalty){
			normalPenaltyFactor=normalPenalty; tangentialPenaltyFactor=tangentialPenalty;
		}
		inline void setSlipDirectionRegularizer(double eps){ if(eps>=0.0) epsD = eps; else epsD=0.0; }
		inline void setContactMethod(CONTACT_METHOD m){ method = m; }

		virtual int dynamicImplicitTimestep(double dt, double eps=LinearFEM::FORCE_BALANCE_EPS);
		int dynamicImplicitTanhPenaltyContactTimestep(double dt, double eps=LinearFEM::FORCE_BALANCE_EPS);
		int dynamicImplicitClampedLinearPenaltyContactTimestep(double dt, double eps=LinearFEM::FORCE_BALANCE_EPS);
		int dynamicImplicitClassificationLinearPenaltyContactTimestep(double dt, double eps=LinearFEM::FORCE_BALANCE_EPS);
		int dynamicImplicitQPContactTimestep(QPSolver& qpSolver, double dt, double eps=LinearFEM::FORCE_BALANCE_EPS);
		int dynamicImplicitHybridPenaltyContactTimestep(double dt, double eps=LinearFEM::FORCE_BALANCE_EPS);

		void assembleTanhPenaltyForceAndStiffness(const Eigen::VectorXd& vi, const double dt, UPDATE_MODE mode=UPDATE);
		bool assembleClampedLinearPenaltyForceAndStiffness(Eigen::VectorXd& fc, const Eigen::VectorXd& vi, const unsigned int iter, const double dt, UPDATE_MODE mode=UPDATE);
		void assembleLinearPenaltyForceAndStiffness(Eigen::VectorXd& fc, const Eigen::VectorXd& vi, const std::vector<CONTACT_STATE>& contactState, const std::vector<unsigned int>& contactObstacle, const double dt, UPDATE_MODE mode=UPDATE);
		void assembleNormalAndSlipPenaltyForceAndStiffness(Eigen::VectorXd& fc, const Eigen::VectorXd& fsDir, const Eigen::VectorXd& vi, const std::vector<CONTACT_STATE>& contactState, const std::vector<unsigned int>& contactObstacle, const double dt, UPDATE_MODE mode=UPDATE);
		
		std::vector<const DiffableScalarField*> rigidObstacles; // keep a list of rigid obstacles
		std::vector<double> frictionCoefficient; // store a coefficient of friction for each rigidObstacles
		std::map<unsigned int, std::vector<CONTACT_STATE> > contactStateByObstacle; // store contact state of each (boundary) node per rigid obstacle - not used by all contact methods though
		unsigned int method;
		double epsD; // regularization of direction normalization
		double normalPenaltyFactor, tangentialPenaltyFactor;

		SparseMatrixD Sr, R; // local rotations for tangential stick constraints, Sr = R*S*R.transpose() (used for CONTACT_HYBRID)
		std::set<unsigned int> stickDOFs; // constrained DOFs for tangential stick (used for CONTACT_HYBRID)

	protected:
		// internal penalty force functions
		void nodalTanhPenaltyForceAndStiffness(
			Eigen::Vector3d& f_n, Eigen::Matrix3d& Kn, Eigen::Vector3d& f_t, Eigen::Matrix3d& Kt, Eigen::Matrix3d& Dt,
			const double pF, const double pFt, const double cF, const double g, const Eigen::Vector3d& n,
			const Eigen::Vector3d& v, const double dt, UPDATE_MODE mode
		);
		CONTACT_STATE nodalClampedLinearPenaltyForceAndStiffness(
			Eigen::Vector3d& f_n, Eigen::Matrix3d& Kn, Eigen::Vector3d& f_t, Eigen::Matrix3d& Kt, Eigen::Matrix3d& Dt,
			const double pF, const double pFt, const double cF, const double g, const Eigen::Vector3d& n, bool forceStick,
			const Eigen::Vector3d& v, const double dt, UPDATE_MODE mode
		);
		void nodalLinearPenaltyForceAndStiffness(
			Eigen::Vector3d& f_n, Eigen::Matrix3d& Kn, Eigen::Vector3d& f_t, Eigen::Matrix3d& Kt, Eigen::Matrix3d& Dt,
			const double pF, const double pFt, const double cF, const double g, const Eigen::Vector3d& n,
			const Eigen::Vector3d& v, CONTACT_STATE contactState, const double dt, UPDATE_MODE mode
		);
		void nodalNormalAndSlipPenaltyForceAndStiffness(
			Eigen::Vector3d& f_n, Eigen::Matrix3d& Kn, Eigen::Vector3d& f_t, Eigen::Matrix3d& Kt, Eigen::Matrix3d& Dt,
			const double pF, const double cF, const double g, const Eigen::Vector3d& n,
			const Eigen::Vector3d& v, const Eigen::Vector3d& d, CONTACT_STATE contactState, const double dt, UPDATE_MODE mode
		);
		bool classifyStickSlipNodes(std::vector<ContactFEM::CONTACT_STATE>& contactState, std::vector<unsigned int>& contactObstacle, Eigen::VectorXd& fs, const std::vector<bool>& contactConstraintDone, const Eigen::VectorXd& vi, const Eigen::VectorXd& fc, unsigned int iter, double eps);
	};


	class DiffableScalarField{ // base class for differentiable scalar fields, evaluates to g=0 everywhere
	public:
		// evaluate g(x0,x,t): (R^3 x R^3 x R) -> R, where x0 is the rest space (undeformed) coordinate and x is the world space (deformed) coordinate
		virtual double eval(Eigen::Vector3d& dg_dx, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {dg_dx.setZero(); return 0.0;}
	};

	class QPSolver{
	public:
		QPSolver();
		~QPSolver();

		// solve: vi = arg min (0.5 v' S v + v' r) s.t. cLower <= C v <= cUpper
		// when using mosek only cLower is used for inequalities or set cLower==cUpper for equalities
		int solve(
			Eigen::VectorXd& vi, Eigen::VectorXd& lambda, double eps, unsigned int numConstraints,
			SparseMatrixD& S, Eigen::VectorXd& r, SparseMatrixD& C, Eigen::VectorXd& cLower, Eigen::VectorXd& cUpper
		);

	protected:
		void* mosekEnv; // only used if compiled with QPSOLVE_USE_MOSEK
	};

}

#endif
