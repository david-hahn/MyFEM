#ifndef BOUNDARYFIELDOBJECTIVEFUNCTION_H
#define BOUNDARYFIELDOBJECTIVEFUNCTION_H

#include "SensitivityAnalysis.h"

namespace MyFEM{
	class VectorField;

	class BoundaryFieldObjectiveFunction : public ObjectiveFunction{
	public:
		BoundaryFieldObjectiveFunction(double scale_=1.0) : scale(scale_) { targetFields.clear(); }

		virtual double evaluate( LinearFEM& fem,
			Eigen::VectorXd& phi_x, Eigen::VectorXd& phi_v, Eigen::VectorXd& phi_f, Eigen::VectorXd& phi_q
		);

		void addTargetField(unsigned int bndId, VectorField& g){ targetFields[bndId] = &g; }
		void addTargetField(unsigned int bndId, VectorField* g){ targetFields[bndId] =  g; }

		Eigen::VectorXd phi_xx; // use phi_xx as diagonal matrix ... this objective function is = sum( a_i (u_i - u_g) ^2 ), so we can easily compute the second derivative wrt. x = diag(a_i), phi_xx will be (over-)written in evaluate()
		Eigen::VectorXd phi_vv; // same as above for velocity terms

		std::string debugFileName; unsigned int debugCount=0; // write some debug info if debugFileName!=""
	protected:
		std::map<unsigned int, VectorField*> targetFields;
		double scale=1.0;
	};

	class AverageBoundaryValueObjectiveFunction : public BoundaryFieldObjectiveFunction{
	public:
		enum TARGET_MODE { TARGET_LOCATION, TARGET_DISPLACEMENT, TARGET_VELOCITY } targetMode = TARGET_LOCATION;
		enum RESET_MODE { RESET_NO_CHANGE, RESET_FIELD_OFFSETS } resetMode = RESET_NO_CHANGE;
		AverageBoundaryValueObjectiveFunction(double scale_=1.0) : BoundaryFieldObjectiveFunction(scale_) {}
		virtual double evaluate( LinearFEM& fem,
			Eigen::VectorXd& phi_x, Eigen::VectorXd& phi_v, Eigen::VectorXd& phi_f, Eigen::VectorXd& phi_q
		);
		virtual ~AverageBoundaryValueObjectiveFunction(){
			for(std::map<unsigned int,VectorField*>::iterator it=trackedLocations.begin(); it!=trackedLocations.end(); ++it) if(it->second!=NULL) delete it->second;
		}
		virtual void writeTrackedLocations(std::string fname, unsigned int samples);
		virtual void reset( LinearFEM& fem );
	protected:
		bool resetOnNextEval=false;
		std::map<unsigned int,VectorField*> trackedLocations, trackedEvals; // for debug output: store recent results of eval() calls in target fields
	};

}
#endif
