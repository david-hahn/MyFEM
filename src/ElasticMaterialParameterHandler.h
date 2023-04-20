#ifndef ELASTICMATERIALPARAMETERHANDLER_H
#define ELASTICMATERIALPARAMETERHANDLER_H

#include "SensitivityAnalysis.h"
#include "InitialConditionParameterHandler.h"

namespace MyFEM{

	class GlobalElasticMaterialParameterHandler : public ParameterHandler{
	public:
		GlobalElasticMaterialParameterHandler(unsigned int bodyId) : ParameterHandler(bodyId) {}
		virtual void getCurrentParams(  Eigen::VectorXd& q, LinearFEM& fem); // writes q, does not change fem
		virtual void setNewParams(const Eigen::VectorXd& q, LinearFEM& fem); // writes to fem, does not change q
		virtual unsigned int getNumberOfParams(const LinearFEM& fem); // how many parameters are we working with (rows in q)
		virtual double computeConstraintDerivatives(Eigen::MatrixXd& g_q, Eigen::VectorXd& phiQ_q, LinearFEM& fem); // partial derivatives of the constraint function wrt. the design parameters + regularizer term phiQ(q) and dphiQ_dq
	protected:
		unsigned int firstElemInActiveBody; bool initialized=false;
	};

	class PerElementElasticMaterialParameterHandler : public ParameterHandler{
	public:
		PerElementElasticMaterialParameterHandler(unsigned int bodyId) : ParameterHandler(bodyId) {}
		virtual bool useSparseDgDq(){return true;}
		virtual void getCurrentParams(  Eigen::VectorXd& q, LinearFEM& fem); // writes q, does not change fem
		virtual void setNewParams(const Eigen::VectorXd& q, LinearFEM& fem); // writes to fem, does not change q
		virtual unsigned int getNumberOfParams(const LinearFEM& fem); // how many parameters are we working with (rows in q)
		virtual double computeConstraintDerivatives(SparseMatrixD& g_q, Eigen::VectorXd& phiQ_q, LinearFEM& fem); // partial derivatives of the constraint function wrt. the design parameters + regularizer term phiQ(q) and dphiQ_dq
	};

	class GlobalDensityParameterHandler : public ParameterHandler{
	public:
		GlobalDensityParameterHandler(unsigned int bodyId) : ParameterHandler(bodyId) {}
		virtual void getCurrentParams(  Eigen::VectorXd& q, LinearFEM& fem); // writes q, does not change fem
		virtual void setNewParams(const Eigen::VectorXd& q, LinearFEM& fem); // writes to fem, does not change q
		virtual unsigned int getNumberOfParams(const LinearFEM& fem); // how many parameters are we working with (rows in q)
		virtual double computeConstraintDerivatives(Eigen::MatrixXd& g_q, Eigen::VectorXd& phiQ_q, LinearFEM& fem); // partial derivatives of the constraint function wrt. the design parameters + regularizer term phiQ(q) and dphiQ_dq
	protected:
		unsigned int firstElemInActiveBody; bool initialized=false;
	};

	class PerElementDensityMaterialParameterHandler : public ParameterHandler{
	public:
		PerElementDensityMaterialParameterHandler(unsigned int bodyId, bool useReferenceMassRegularizer_=false, double regularizerWeight_=1.0) : ParameterHandler(bodyId) {
			useReferenceMassRegularizer=useReferenceMassRegularizer_; regularizerWeight=regularizerWeight_; referenceMass=-1.0;  // if useReferenceMassRegularizer is true, referenceMass will be calculated from the state of the FEM object before overwriting with the first call to setNewParams()
		}
		virtual bool useSparseDgDq(){return true;}
		virtual void getCurrentParams(  Eigen::VectorXd& q, LinearFEM& fem); // writes q, does not change fem
		virtual void setNewParams(const Eigen::VectorXd& q, LinearFEM& fem); // writes to fem, does not change q
		virtual unsigned int getNumberOfParams(const LinearFEM& fem); // how many parameters are we working with (rows in q)
		virtual double computeConstraintDerivatives(SparseMatrixD& g_q, Eigen::VectorXd& phiQ_q, LinearFEM& fem); // partial derivatives of the constraint function wrt. the design parameters + regularizer term phiQ(q) and dphiQ_dq
	protected:
		bool useReferenceMassRegularizer;
		double referenceMass, regularizerWeight;
	};



	// implementation in PrincipalStretchMaterial.cpp
	class GlobalPrincipalStretchMaterialParameterHandler : public ParameterHandler{
	public:
		unsigned int debugTmp=0;
		GlobalPrincipalStretchMaterialParameterHandler(unsigned int bodyId) : ParameterHandler(bodyId) {}
		virtual void getCurrentParams(  Eigen::VectorXd& q, LinearFEM& fem); // writes q, does not change fem
		virtual void setNewParams(const Eigen::VectorXd& q, LinearFEM& fem); // writes to fem, does not change q
		virtual unsigned int getNumberOfParams(const LinearFEM& fem); // how many parameters are we working with (rows in q)
		virtual double computeConstraintDerivatives(Eigen::MatrixXd& g_q, Eigen::VectorXd& phiQ_q, LinearFEM& fem); // partial derivatives of the constraint function wrt. the design parameters + regularizer term phiQ(q) and dphiQ_dq
		double alpha = 1e-10; // regularizer weight (used in computeConstraintDerivatives)
	};

	// use two parameter handlers in conjunction ...
	class CombinedParameterHandler : public InitialConditionParameterHandler{
	protected:
		ParameterHandler& qh1;
		ParameterHandler& qh2;
	public:
		bool useLogOfParams=false;
		virtual bool useSparseDgDq(){return qh1.useSparseDgDq() || qh2.useSparseDgDq();}
		CombinedParameterHandler(ParameterHandler& handler1, ParameterHandler& handler2) : qh1(handler1), qh2(handler2), InitialConditionParameterHandler(0) {}
		virtual void getCurrentParams(  Eigen::VectorXd& q, LinearFEM& fem); // writes q, does not change fem
		virtual void setNewParams(const Eigen::VectorXd& q, LinearFEM& fem); // writes to fem, does not change q
		virtual unsigned int getNumberOfParams(const LinearFEM& fem); // how many parameters are we working with (rows in q)
		virtual double computeConstraintDerivatives(Eigen::MatrixXd& g_q, Eigen::VectorXd& phiQ_q, LinearFEM& fem); // partial derivatives of the constraint function wrt. the design parameters + regularizer term phiQ(q) and dphiQ_dq
		virtual double computeConstraintDerivatives(SparseMatrixD&   g_q, Eigen::VectorXd& phiQ_q, LinearFEM& fem);
		// new for compatibility with InitialConditionParameterHandler (only one of the two can be a subclass of InitialConditionParameterHandler)
		virtual void applyInitialConditions( LinearFEM& fem );
		virtual void computeInitialDerivatives( Eigen::MatrixXd& dx_dq, Eigen::MatrixXd& dv_dq, LinearFEM& fem );

	};

}
#endif
