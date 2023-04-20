#ifndef VISCOUSMATERIALPARAMETERHANDLER_H
#define VISCOUSMATERIALPARAMETERHANDLER_H

#include "SensitivityAnalysis.h"

namespace MyFEM{

	// Parameter handler for PowerLawViscosityModel
	class GlobalPLViscosityMaterialParameterHandler : public ParameterHandler{
	public:
		GlobalPLViscosityMaterialParameterHandler(unsigned int bodyId) : ParameterHandler(bodyId) {}
		virtual void getCurrentParams(  Eigen::VectorXd& q, LinearFEM& fem); // writes q, does not change fem
		virtual void setNewParams(const Eigen::VectorXd& q, LinearFEM& fem); // writes to fem, does not change q
		virtual unsigned int getNumberOfParams(const LinearFEM& fem){ return 2; } // how many parameters are we working with (rows in q)
		virtual double computeConstraintDerivatives(Eigen::MatrixXd& g_q, Eigen::VectorXd& phiQ_q, LinearFEM& fem); // partial derivatives of the constraint function wrt. the design parameters + regularizer term phiQ(q) and dphiQ_dq
	};

	// Parameter handler for PowerSeriesViscosityModel
	class GlobalPSViscosityMaterialParameterHandler : public GlobalPLViscosityMaterialParameterHandler{
	public:
		GlobalPSViscosityMaterialParameterHandler(unsigned int bodyId) : GlobalPLViscosityMaterialParameterHandler(bodyId) {}
		virtual void getCurrentParams(  Eigen::VectorXd& q, LinearFEM& fem); // writes q, does not change fem
		virtual void setNewParams(const Eigen::VectorXd& q, LinearFEM& fem); // writes to fem, does not change q
		virtual unsigned int getNumberOfParams(const LinearFEM& fem); // how many parameters are we working with (rows in q)
		virtual double computeConstraintDerivatives(Eigen::MatrixXd& g_q, Eigen::VectorXd& phiQ_q, LinearFEM& fem); // partial derivatives of the constraint function wrt. the design parameters + regularizer term phiQ(q) and dphiQ_dq
	protected:
		unsigned int nParams=0;
	};

	// Parameter handler for RotationInvariantViscosityModel
	class GlobalRotationInvariantViscosityMaterialParameterHandler : public ParameterHandler{
	public:
		GlobalRotationInvariantViscosityMaterialParameterHandler(unsigned int bodyId) : ParameterHandler(bodyId) {}
		virtual void getCurrentParams(  Eigen::VectorXd& q, LinearFEM& fem); // writes q, does not change fem
		virtual void setNewParams(const Eigen::VectorXd& q, LinearFEM& fem); // writes to fem, does not change q
		virtual unsigned int getNumberOfParams(const LinearFEM& fem){ return 1; } // how many parameters are we working with (rows in q)
		virtual double computeConstraintDerivatives(Eigen::MatrixXd& g_q, Eigen::VectorXd& phiQ_q, LinearFEM& fem); // partial derivatives of the constraint function wrt. the design parameters + regularizer term phiQ(q) and dphiQ_dq
	};

	// Parameter handler for EigenmodeViscosityModel
	class EigenmodeViscosityMaterialParameterHandler : public ParameterHandler{
	public:
		EigenmodeViscosityMaterialParameterHandler(unsigned int bodyId) : ParameterHandler(bodyId) {}
		//EigenmodeViscosityMaterialParameterHandler(){ nParams=0; printf("\n%% WARNING: EigenmodeViscosityMaterialParameterHandler is inaccurate when used with AdjointSensitiviy AND BDF2 time integration.");}
		virtual void getCurrentParams(  Eigen::VectorXd& q, LinearFEM& fem); // writes q, does not change fem
		virtual void setNewParams(const Eigen::VectorXd& q, LinearFEM& fem); // writes to fem, does not change q
		virtual unsigned int getNumberOfParams(const LinearFEM& fem); // how many parameters are we working with (rows in q)
		virtual double computeConstraintDerivatives(Eigen::MatrixXd& g_q, Eigen::VectorXd& phiQ_q, LinearFEM& fem); // partial derivatives of the constraint function wrt. the design parameters + regularizer term phiQ(q) and dphiQ_dq
	protected:
		unsigned int nParams=0;
	};
}
#endif
