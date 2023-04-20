#ifndef DIFFERENTIABLESPLINE_H
#define DIFFERENTIABLESPLINE_H

#include "types.h"
#include <string>

#include "vtkObject.h"
#include "vtkSmartPointer.h"
#include "vtkKochanekSpline.h"

namespace MyFEM{
	// implementation currently in TemporalInterpolationField.cpp ... move eventually
	class DifferentiableSpline : public vtkKochanekSpline{
		vtkTypeMacro(DifferentiableSpline, vtkKochanekSpline)
		static DifferentiableSpline* New(){ return new DifferentiableSpline(); } // for compatibility with vtkSmartPointer
		virtual void Delete(){} // for compatibility with vtkSmartPointer
		virtual ~DifferentiableSpline(){Delete();}

		int    EvaluateCoefficients(double& t, double& c0, double& c1, double& c2, double& c3); // returns interval index
		double Evaluate(double t);
		double EvaluateTDerivative(double t); // compute derivative df/dt of spline value f(t) with respect to spline parameter t
		void   EvaluateYDerivative(double t, Eigen::VectorXd& dfdy); // compute derivative df/dy of spline value f(t) with respect to input control point values y at time t

		double firstDerivRegularizer(Eigen::VectorXd& phiQ_y); // a regularizer function that penalizes negative first derivatives of this spline and it's derivative wrt. control point values

		void Compute(); // override compute function to also store coefficient-derivatives

		Eigen::MatrixXd coeffDerivs;
	};		

}

#endif
