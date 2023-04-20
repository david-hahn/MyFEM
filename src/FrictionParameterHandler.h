#ifndef FRICTIONPARAMETERHANDLER_H
#define FRICTIONPARAMETERHANDLER_H

#include "SensitivityAnalysis.h"

namespace MyFEM{
	class ContactFEM;
	class FrictionParameterHandler : public ParameterHandler{
	public:
		FrictionParameterHandler(unsigned int bodyId, const double timestep_) : ParameterHandler(bodyId), timestep(timestep_) {}
		virtual void getCurrentParams(  Eigen::VectorXd& q, LinearFEM& fem);
		virtual void setNewParams(const Eigen::VectorXd& q, LinearFEM& fem);
		virtual unsigned int getNumberOfParams(const LinearFEM& fem);
		virtual double computeConstraintDerivatives(Eigen::MatrixXd& g_q, Eigen::VectorXd& phiQ_q, LinearFEM& fem); // partial derivatives of the constraint function wrt. the design parameters + regularizer term phiQ(q) and dphiQ_dq

	protected:
		virtual double linearPenaltyForceDerivatives(Eigen::MatrixXd& g_q, Eigen::VectorXd& phiQ_q, ContactFEM& fem);
		virtual double tanhPenaltyForceDerivatives(Eigen::MatrixXd& g_q, Eigen::VectorXd& phiQ_q, ContactFEM& fem);
		const double timestep;
	};
}

#endif // !FRICTIONPARAMETERHANDLER_H
