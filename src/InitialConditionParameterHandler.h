#ifndef INITIALCONDITIONPARAMETERHANDLER_H
#define INITIALCONDITIONPARAMETERHANDLER_H

#include "SensitivityAnalysis.h"

namespace MyFEM{
	class InitialConditionParameterHandler : public ParameterHandler{
	public:
		InitialConditionParameterHandler(unsigned int bodyId) : ParameterHandler(bodyId) { setPostion=setOrientation=setVelocity=setAngularVelocity=positionGradients=orientationGradients=velocityGradients=angularVelocityGradients=false; angularVelocityScale=1.0; }
		virtual void getCurrentParams(  Eigen::VectorXd& q, LinearFEM& fem){ q = currentQ; }
		virtual void setNewParams(const Eigen::VectorXd& q, LinearFEM& fem){ currentQ = q; }
		virtual unsigned int getNumberOfParams(const LinearFEM& fem){ return expectedNumberOfParams(); }
		virtual double computeConstraintDerivatives(Eigen::MatrixXd& g_q, Eigen::VectorXd& phiQ_q, LinearFEM& fem){ g_q.resize( getNumberOfDOFs(fem), getNumberOfParams(fem) ); g_q.setZero(); phiQ_q.setZero(); } // partial derivatives of the constraint function wrt. the design parameters + regularizer term phiQ(q) and dphiQ_dq
		// new for InitialConditionParameterHandler
		virtual void applyInitialConditions( LinearFEM& fem );
		virtual void computeInitialDerivatives( Eigen::MatrixXd& dx_dq, Eigen::MatrixXd& dv_dq, LinearFEM& fem );

		inline unsigned int getPositionIndex(){ return 0; }
		inline unsigned int getOrientationIndex(){ return 3*( (setPostion?1:0) ); }
		inline unsigned int getVelocityIndex(){ return 3*( (setPostion?1:0)+(setOrientation?1:0) ); }
		inline unsigned int getAngularVelocityIndex(){ return 3*( (setPostion?1:0)+(setOrientation?1:0)+(setVelocity?1:0) ); }

		bool setPostion, setOrientation, setVelocity, setAngularVelocity, positionGradients, orientationGradients, velocityGradients, angularVelocityGradients;
		double angularVelocityScale;
	protected:
		Eigen::VectorXd currentQ;
		bool checkParamSize(){ return currentQ.size()==expectedNumberOfParams(); }
		inline unsigned int expectedNumberOfParams(){ return (3*( (setPostion?1:0)+(setOrientation?1:0)+(setVelocity?1:0)+(setAngularVelocity?1:0) )); }
	};
}

#endif // !INITIALCONDITIONPARAMETERHANDLER_H
