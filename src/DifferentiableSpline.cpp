#include "DifferentiableSpline.h"

// for derivatives of splines ...
#include "vtkPiecewiseFunction.h"
#include <unsupported\Eigen\AutoDiff>

using namespace MyFEM;

// adapted from vtkKochanekSpline ... Closed flag ignored and assumed false
int DifferentiableSpline::EvaluateCoefficients(double& t, double& c0, double& c1, double& c2, double& c3){
	int index = 0;
	if (this->ComputeTime < this->GetMTime ()){ this->Compute ();}
	int size = this->PiecewiseFunction->GetSize ();
	if (size < 2) { c0=c1=c2=c3=0.0; return 0; }// make sure we have at least 2 points
	if (t < Intervals[0]){ t = Intervals[0]; }// clamp the function at both ends
	if (t > Intervals[size - 1]){ t = Intervals[size - 1]; }
	index = this->FindIndex(size,t); // find pointer to cubic spline coefficient
	t = (t - Intervals[index]) / (Intervals[index+1] - Intervals[index]); // calculate offset within interval
	c0 = Coefficients[index*4  ];
	c1 = Coefficients[index*4+1];
	c2 = Coefficients[index*4+2];
	c3 = Coefficients[index*4+3];
	return index;
}
double DifferentiableSpline::Evaluate(double t){
	double c0, c1, c2, c3;
	EvaluateCoefficients(t,c0,c1,c2,c3); // also offsets t to interval
	return (t * (t * (t * c3 + c2) + c1) + c0);  // evaluate y
}

double DifferentiableSpline::EvaluateTDerivative(double t){
	double c0, c1, c2, c3;
	int index = EvaluateCoefficients(t,c0,c1,c2,c3); // applies t = (t - Intervals[index]) / (Intervals[index+1] - Intervals[index]); // calculate offset within interval
	return 1.0/(Intervals[index+1] - Intervals[index]) * (t*(t*3.0*c3+2.0*c2)+c1);  // evaluate dy/dt = 3t^2*c3 + 2t*c2 + c1, y = (t * (t * (t * c3 + c2) + c1) + c0) = t^3*c3 + t^2*c2 + t*c1 + c0
}

void DifferentiableSpline::EvaluateYDerivative(double t, Eigen::VectorXd& dfdy){
	double c0, c1, c2, c3;
	int index = EvaluateCoefficients(t,c0,c1,c2,c3);
	// spline value  (t * (t * (t * c3 + c2) + c1) + c0) == (c3*t^3 + c2*t^2 + c1*t + c0)
	dfdy.resize(PiecewiseFunction->GetSize());
	Eigen::Block<Eigen::MatrixXd,-1,1,!Eigen::MatrixXd::IsRowMajor>
		dc0dy = coeffDerivs.col(index*4  ),
		dc1dy = coeffDerivs.col(index*4+1),
		dc2dy = coeffDerivs.col(index*4+2),
		dc3dy = coeffDerivs.col(index*4+3);
	dfdy = dc3dy*(t*t*t) + dc2dy*(t*t) + dc1dy*t + dc0dy;
}
//----------------------------------------------------------------------------
// Compute Kochanek Spline coefficients.
void DifferentiableSpline::Compute(){
	typedef Eigen::AutoDiffScalar<Eigen::VectorXd> adouble;
	
	double *ts, *xs;
	int size;
	int i;

	size = this->PiecewiseFunction->GetSize (); // get the size of the independent variables

	if(size < 2){
	vtkErrorMacro("Spline requires at least 2 points. # of points is: " <<size);
	return;
	}

	// copy the independent variables
	delete [] this->Intervals;
	this->Intervals = new double[size];
	ts = this->PiecewiseFunction->GetDataPointer ();
	for (i = 0; i < size; i++){
		this->Intervals[i] = *(ts + 2*i);
	}

	
	delete [] this->Coefficients; // allocate memory for coefficients
	this->Coefficients = new double [4*size];
	
	adouble* dependent = new adouble[size]; //dependent = new double [size]; // allocate memory for dependent variables

	adouble (*coefficients)[4] = (adouble (*)[4])(new adouble [4*size]); //coefficients = this->Coefficients; // get start of coefficients for this dependent variable

	// get the dependent variable values
	xs = this->PiecewiseFunction->GetDataPointer () + 1;
	for (int j = 0; j < size; j++){
		dependent[j].value() = *(xs + 2*j);
		dependent[j].derivatives() = Eigen::VectorXd::Unit( size, j );

		coefficients[j][0].derivatives().resize(size);
		coefficients[j][1].derivatives().resize(size);
		coefficients[j][2].derivatives().resize(size);
		coefficients[j][3].derivatives().resize(size);
	}
 
	double &tension=DefaultTension, &bias=DefaultBias, &continuity=DefaultContinuity;

	const double VTK_EPSILON=0.0001;
	adouble        cs;             /* source chord                 */
	adouble        cd;             /* destination chord            */
	adouble        ds;             /* source deriviative           */
	adouble        dd;             /* destination deriviative      */
	adouble        n0, n1;         /* number of frames btwn nodes  */
	cs.derivatives().resize( size );
	cd.derivatives().resize( size );
	ds.derivatives().resize( size );
	dd.derivatives().resize( size );
	n0.derivatives().resize( size );
	n1.derivatives().resize( size );
	int           N;              /* top point number             */

	N = size - 1;

	for (i=1; i < N; i++){
		cs = dependent[i] - dependent[i-1];
		cd = dependent[i+1] - dependent[i];

		ds = cs*((1 - tension)*(1 - continuity)*(1 + bias)) / 2.0 + cd*((1 - tension)*(1 + continuity)*(1 - bias)) / 2.0;

		dd = cs*((1 - tension)*(1 + continuity)*(1 + bias)) / 2.0 + cd*((1 - tension)*(1 - continuity)*(1 - bias)) / 2.0;

		// adjust deriviatives for non uniform spacing between nodes
		n1  = Intervals[i+1] - Intervals[i];
		n0  = Intervals[i] - Intervals[i-1];

		ds *= (2 * n0 / (n0 + n1));
		dd *= (2 * n1 / (n0 + n1));

		coefficients[i][0] = dependent[i];
		coefficients[i][1] = dd;
		coefficients[i][2] = ds;
	}

	// Calculate the deriviatives at the end points
	coefficients[0][0] = dependent[0];
	coefficients[N][0] = dependent[N];
	coefficients[N][1] = 0.0;
	coefficients[N][2] = 0.0;
	coefficients[N][3] = 0.0;

	switch (LeftConstraint){
		case 0: // desired slope at leftmost point is leftValue
		coefficients[0][1] = this->ComputeLeftDerivative();
		break;
		case 1: // desired slope at leftmost point is leftValue
		coefficients[0][1] = LeftValue;
		break;
		case 2: // desired second derivative at leftmost point is leftValue
		coefficients[0][1] = (6*(dependent[1] - dependent[0]) - 2*coefficients[1][2] - LeftValue)
				/ 4.0;
		break;
		case 3: // desired secord derivative at leftmost point is leftValue times secod derivative at first interior point
		if ((LeftValue > (-2.0 + VTK_EPSILON)) ||
			(LeftValue < (-2.0 - VTK_EPSILON))){
			coefficients[0][1] = (3*(1 + LeftValue)*(dependent[1] - dependent[0]) - (1 + 2*LeftValue)*coefficients[1][2]) / (2 + LeftValue);
		}else{
			coefficients[0][1] = 0.0;
		}
		break;
	}

	switch (RightConstraint){
		case 0: // desired slope at rightmost point is rightValue
		coefficients[N][2] = this->ComputeRightDerivative();
		break;
		case 1: // desired slope at rightmost point is rightValue
		coefficients[N][2] = RightValue;
		break;
		case 2: // desired second derivative at rightmost point is rightValue
			coefficients[N][2] = (6*(dependent[N] - dependent[N-1]) - 2*coefficients[N-1][1] + RightValue) / 4.0;
			break;
		case 3: // desired secord derivative at rightmost point is rightValue times secord derivative at last interior point
		if ((RightValue > (-2.0 + VTK_EPSILON)) ||
			(RightValue < (-2.0 - VTK_EPSILON))){
			coefficients[N][2] = (3*(1 + RightValue)*(dependent[N] - dependent[N-1]) - (1 + 2*RightValue)*coefficients[N-1][1]) / (2 + RightValue);
		}else{
			coefficients[N][2] = 0.0;
		}
		break;
	}

	// Compute the Coefficients
	for (i=0; i < N; i++){
		coefficients[i][2] = (-3 * dependent[i])        + ( 3 * dependent[i+1]) + (-2 * coefficients[i][1]) + (-1 * coefficients[i+1][2]);
		coefficients[i][3] = ( 2 * dependent[i])        + (-2 * dependent[i+1]) + ( 1 * coefficients[i][1]) + ( 1 * coefficients[i+1][2]);
	}

	// store data
	coeffDerivs.resize(size,4*size);
	for(int i=0; i<size; ++i){ for(int j=0; j<4; ++j){
		Coefficients[   4*i+j] = coefficients[i][j].value();
		coeffDerivs.col(4*i+j) = coefficients[i][j].derivatives();
	}}

	// free the dependent variable storage
	delete [] dependent;
	delete [] coefficients;

	// update compute time
	this->ComputeTime = this->GetMTime();
}

double DifferentiableSpline::firstDerivRegularizer(Eigen::VectorXd& phiQ_y){
	// compute regularizer function value and add derivatives to phiQ_y
	// parameters are the spline control point values
	// we penalize negative first derivative
	// -- at each control point
	// -- and at the inflection point of each spline segment (assuming cubic spline)
	int size = this->PiecewiseFunction->GetSize (); // get the size of the independent variables -- this is the number of control points; we have size-1 intervals
	double phiVal = 0.0;
	phiQ_y.resize( size );
	phiQ_y.setZero();

	Eigen::Vector4d dPhidc; // derivative of phi wrt. the four coefficients of this interval

	// for each control point i, we have four Coefficients[4*i+j] and derivatives of these coeffs wrt. control point values coeffDerivs.col(4*i+j)
	// spline value is (t * (t * (t * c3 + c2) + c1) + c0) == (c3*t^3 + c2*t^2 + c1*t + c0), where t is in (0,1), ie. scaled by 1.0/(Intervals[i+1] - Intervals[i])
	for(int i = 0; i<(size-1); ++i){ // for each interval (i, i+1) ... skip the last interval as we assume an open spline
		dPhidc.setZero();
		// penalize if constant value at i is higher than at i+1
		if((Coefficients[4*(i+1)] - Coefficients[4*i]) < 0.0 ){
			phiVal -= (Coefficients[4*(i+1)] - Coefficients[4*i]) * 10.0/(Intervals[i+1] - Intervals[i]);
			dPhidc[0] += 10.0/(Intervals[i+1] - Intervals[i]);
		}

		// first point i -- first derivative is c1 * 1.0/(Intervals[i+1] - Intervals[i]), and c1 == Coefficients[4*i+1]
		if( Coefficients[4*i+1]<0.0 ){
			phiVal -=  Coefficients[4*i+1] * 1.0/(Intervals[i+1] - Intervals[i]);
			dPhidc[1] -= 1.0/(Intervals[i+1] - Intervals[i]);
		}
		// second point i+1 -- first derivative is (c1 + 2c2 + 3c3) * 1.0/(Intervals[i+1] - Intervals[i]) (t==1 at i+1 for this interval)
		double dfdt = Coefficients[4*i+1] + 2.0*Coefficients[4*i+2] + 3.0*Coefficients[4*i+3];
		if( dfdt < 0.0){
			phiVal -= dfdt * 1.0/(Intervals[i+1] - Intervals[i]);
			dPhidc[1] -= 1.0/(Intervals[i+1] - Intervals[i]);
			dPhidc[2] -= 2.0/(Intervals[i+1] - Intervals[i]);
			dPhidc[3] -= 3.0/(Intervals[i+1] - Intervals[i]);
		}
		// inflection point -- if the inflection point at t=-1/3*c2/c3 is within (0,1), also check first derivative there
		double tip = -1.0/3.0*Coefficients[4*i+2]/Coefficients[4*i+3];
		if( tip > 0.0 && tip < 1.0 ){
			dfdt = Coefficients[4*i+1] + 2.0*Coefficients[4*i+2]*tip + 3.0*Coefficients[4*i+3]*tip*tip;
			if( dfdt < 0.0 ){
				phiVal -= dfdt * 1.0/(Intervals[i+1] - Intervals[i]);
				dPhidc[1] -= 1.0/(Intervals[i+1] - Intervals[i]);
				dPhidc[2] -= 2.0*tip/(Intervals[i+1] - Intervals[i]);
				dPhidc[3] -= 3.0*tip*tip/(Intervals[i+1] - Intervals[i]);
			}
		}

		// now we have added all penalties for this interval to phiVal and the derivatives of the added amount wrt. coefficients is in dPhidc
		phiQ_y += coeffDerivs.block(0,4*i,size,4) * dPhidc;
		phiQ_y -= coeffDerivs.col(4*(i+1)) * dPhidc[0]; // dPhidc[0] affected by c0 of control point i+1 analogous to c0 at i
	}
	phiQ_y /= (double)size;
	return (phiVal / (double)size);
}

// old version for vtkCardinalSpline base class ....
//double DifferentiableSpline::EvaluateTDerivative(double t){
//	// from vtkSpline we have the following members in DifferentiableSpline
//	//int ClampValue;
//	//double *Intervals;
//	//double *Coefficients;
//	//int LeftConstraint;
//	//double LeftValue;
//	//int RightConstraint;
//	//double RightValue;
//	//vtkPiecewiseFunction *PiecewiseFunction;
//	//int Closed;
//	if (ComputeTime < GetMTime ()){ Compute(); } // check to see if we need to recompute the spline
//	int size = PiecewiseFunction->GetSize(); // make sure we have at least 2 points
//	if (size < 2){ return 0.0; }
//	if ( Closed ){ size = size + 1; }
//	if (t < Intervals[0]){ t = Intervals[0]; } // clamp the function at both ends
//	if (t > Intervals[size - 1]){ t = Intervals[size - 1]; }
//	int index = FindIndex(size,t); // find pointer to cubic spline coefficient using bisection method
//	t = (t - Intervals[index]); // calculate offset within interval
//	const double //&c0 = *(Coefficients + index * 4   ), // c0 not needed for derivative
//		&c1 = *(Coefficients + index * 4 +1),
//		&c2 = *(Coefficients + index * 4 +2),
//		&c3 = *(Coefficients + index * 4 +3);
//	return c1 + t*(2.0*c2 + t*3.0*c3); // derivative of spline value (c0 + t*(c1 + t*(c2 + c3*t))) == (c3*t^3 + c2*t^2 + c1*t + c0) wrt. t
//}
//
//void DifferentiableSpline::EvaluateYDerivative(double t, Eigen::VectorXd& dfdy){
//	if (ComputeTime < GetMTime ()){ Compute(); } // check to see if we need to recompute the spline
//	int size = PiecewiseFunction->GetSize(); // make sure we have at least 2 points
//	if (size < 2){ return; }
//	if ( Closed ){ size = size + 1; }
//	if (t < Intervals[0]){ t = Intervals[0]; } // clamp the function at both ends
//	if (t > Intervals[size - 1]){ t = Intervals[size - 1]; }
//	int index = FindIndex(size,t); // find pointer to cubic spline coefficient using bisection method
//	t = (t - Intervals[index]); // calculate offset within interval
//	double &c0 = *(Coefficients + index * 4   ),
//		&c1 = *(Coefficients + index * 4 +1),
//		&c2 = *(Coefficients + index * 4 +2),
//		&c3 = *(Coefficients + index * 4 +3);
//	// spline value (c0 + t*(c1 + t*(c2 + c3*t))) == (c3*t^3 + c2*t^2 + c1*t + c0)
//
//	dfdy.resize(size);
//
//	Eigen::Block<Eigen::MatrixXd,-1,1,!Eigen::MatrixXd::IsRowMajor>
//		dc0dy = coeffDerivs.col(index * 4  ),
//		dc1dy = coeffDerivs.col(index * 4+1),
//		dc2dy = coeffDerivs.col(index * 4+2),
//		dc3dy = coeffDerivs.col(index * 4+3);
//	dfdy = dc3dy*(t*t*t) + dc2dy*(t*t) + dc1dy*t + dc0dy;
//}
//
//// use Eigen to auto-diff Coefficients wrt. input data -- based on vtkCardinalSpline::Compute() and vtkCardinalSpline::Fit1D(...)
//void DifferentiableSpline::Compute(){
//	//printf("\n%% DifferentiableSpline::Compute ... ");
//	if( Closed ){ printf("\n%% DifferentiableSpline::Compute not implemented for closed spines yet! "); return; }
//	
//	typedef Eigen::AutoDiffScalar<Eigen::VectorXd> adouble; //stores a scalar double value and a variable sized derivative vector
//	
//	adouble *work;
//	const int size = PiecewiseFunction->GetSize ();
//	int i;
//
//	// get the size of the independent variables
//	delete [] Intervals;
//	Intervals = new double[size];
//	double* ivals = Intervals;
//	double* ts = PiecewiseFunction->GetDataPointer ();
//	for (i = 0; i < size; i++)
//	{
//		ivals[i] = *(ts + 2*i);
//	}
//
//	// allocate memory for work arrays
//	work = new adouble[size];
//
//	// allocate memory for dependent variables
//	adouble* dependent = new adouble [size];
//
//	// get the dependent variable values
//	double* xs = PiecewiseFunction->GetDataPointer () + 1;
//	for (int j = 0; j < size; j++)
//	{
//		*(dependent + j) = *(xs + 2*j);
//
//		dependent[j].derivatives().resize( 4*size );
//		dependent[j].derivatives() = Eigen::VectorXd::Unit( 4*size, j );
//		work[j].derivatives().resize( 4*size );
//	}
//
//	//Fit1D (size, ivals, dependent,
//	//				work, (adouble (*)[4])coeffs,
//	//				LeftConstraint, LeftValue,
//	//				RightConstraint, RightValue);
//	adouble (*coeffs)[4] = (adouble (*)[4])(new adouble [4*size]);
//	adouble   b = 0.0;
//	adouble   xlk;
//	adouble   xlkp;
//	int      k;
//
//	// develop constraint at leftmost point.
//	switch (LeftConstraint)
//	{
//	case 0:
//		// desired slope at leftmost point is derivative from two points
//		coeffs[0][1] = 1.0;
//		coeffs[0][2] = 0.0;
//		work[0] = ComputeLeftDerivative();
//		break;
//	case 1:
//		// desired slope at leftmost point is LeftValue.
//		coeffs[0][1] = 1.0;
//		coeffs[0][2] = 0.0;
//		work[0] = LeftValue;
//		break;
//	case 2:
//		// desired second derivative at leftmost point is LeftValue.
//		coeffs[0][1] = 2.0;
//		coeffs[0][2] = 1.0;
//		work[0]= 3.0 * ((dependent[1] - dependent[0]) / (ivals[1] - ivals[0])) -
//		0.5 * (ivals[1] - ivals[0]) * LeftValue;
//		break;
//	case 3:
//		// desired second derivative at leftmost point is
//		// LeftValue times second derivative at first interior point.
//		coeffs[0][1] = 2.0;
//		coeffs[0][2] = 4.0 * ((0.5 + LeftValue) /
//									(2.0 + LeftValue));
//		work[0]= 6.0 * ((1.0 + LeftValue) / (2.0 + LeftValue)) *
//		((dependent[1] - dependent[0]) / (ivals[1]-ivals[0]));
//		break;
//	default:
//		assert("check: impossible case." && 0); // reaching this line is a bug.
//		break;
//	}
//
//	// develop body of band matrix.
//	for (k = 1; k < size - 1; k++)
//	{
//	xlk = ivals[k] - ivals[k-1];
//	xlkp = ivals[k+1] - ivals[k];
//	coeffs[k][0] = xlkp;
//	coeffs[k][1] = 2.0 * (xlkp + xlk);
//	coeffs[k][2] = xlk;
//	work[k] = 3.0 * (((xlkp * (dependent[k] - dependent[k-1])) / xlk) +
//						((xlk * (dependent[k+1] - dependent[k])) / xlkp));
//	}
//
//
//	// develop constraint at rightmost point.
//	switch (RightConstraint)
//	{
//	case 0:
//		// desired slope at leftmost point is derivative from two points
//		coeffs[size - 1][0] = 0.0;
//		coeffs[size - 1][1] = 1.0;
//		work[size - 1] = ComputeRightDerivative();
//		break;
//	case 1:
//		// desired slope at rightmost point is RightValue
//		coeffs[size - 1][0] = 0.0;
//		coeffs[size - 1][1] = 1.0;
//		work[size - 1] = RightValue;
//		break;
//	case 2:
//		// desired second derivative at rightmost point is RightValue.
//		coeffs[size-1][0] = 1.0;
//		coeffs[size-1][1] = 2.0;
//		work[size-1] = 3.0 * ((dependent[size-1] - dependent[size-2]) /
//							(ivals[size-1] - ivals[size-2])) +
//		0.5 * (ivals[size-1]-ivals[size-2]) * RightValue;
//		break;
//	case 3:
//		// desired second derivative at rightmost point is
//		// RightValue times second derivative at last interior point.
//		coeffs[size-1][0] = 4.0 * ((0.5 + RightValue) /
//										(2.0 + RightValue));
//		coeffs[size-1][1] = 2.0;
//		work[size-1] = 6.0 * ((1.0 + RightValue) / (2.0 + RightValue)) *
//		((dependent[size-1] - dependent[size-2]) /
//			(ivals[size-1] - ivals[size-2]));
//		break;
//	default:
//		assert("check: impossible case." && 0); // reaching this line is a bug.
//		break;
//	}
//
//	// solve resulting set of equations.
//	coeffs[0][2] = coeffs[0][2] / coeffs[0][1];
//	work[0] = work[0] / coeffs[0][1];
//	coeffs[size-1][2] = 0.0;
//
//	for (k = 1; k < size; k++)
//	{
//	coeffs[k][1] = coeffs[k][1] - (coeffs[k][0] *
//												coeffs[k-1][2]);
//	coeffs[k][2] = coeffs[k][2] / coeffs[k][1];
//	work[k]  = (work[k] - (coeffs[k][0] * work[k-1]))
//		/ coeffs[k][1];
//	}
//
//	for (k = size - 2; k >= 0; k--)
//	{
//	work[k] = work[k] - (coeffs[k][2] * work[k+1]);
//	}
//
//	// the column vector work now contains the first
//	// derivative of the spline function at each joint.
//	// compute the coeffs of the cubic between
//	// each pair of joints.
//	for (k = 0; k < size - 1; k++)
//	{
//	b = ivals[k+1] - ivals[k];
//	coeffs[k][0] = dependent[k];
//	coeffs[k][1] = work[k];
//	coeffs[k][2] = (3.0 * (dependent[k+1] - dependent[k])) / (b * b) -
//		(work[k+1] + 2.0 * work[k]) / b;
//	coeffs[k][3] = (2.0 * (dependent[k] - dependent[k+1])) / (b * b * b) +
//		(work[k+1] + work[k]) / (b * b);
//	}
//
//	// the coeffs of a fictitious nth cubic
//	// are evaluated.  This may simplify
//	// algorithms which include both end points.
//
//	coeffs[size-1][0] = dependent[size-1];
//	coeffs[size-1][1] = work[size-1];
//	coeffs[size-1][2] = coeffs[size-2][2] +
//	3.0 * coeffs[size-2][3] * b;
//	coeffs[size-1][3] = coeffs[size-2][3];
//
//	// free the work array and dependent variable storage
//	delete [] work;
//	delete [] dependent;
//
//	//cout << endl << "% coeff[0][0] derivs: " << coeffs[0][0].derivatives().transpose();
//	//cout << endl << "% coeff[0][1] derivs: " << coeffs[0][1].derivatives().transpose();
//	//cout << endl << "% coeff[0][2] derivs: " << coeffs[0][2].derivatives().transpose();
//	//cout << endl << "% coeff[0][3] derivs: " << coeffs[0][3].derivatives().transpose();
//	//cout << endl << "% coeff[1][0] derivs: " << coeffs[1][0].derivatives().transpose();
//	//cout << endl << "% coeff[1][1] derivs: " << coeffs[1][1].derivatives().transpose();
//	//cout << endl << "% coeff[1][2] derivs: " << coeffs[1][2].derivatives().transpose();
//	//cout << endl << "% coeff[1][3] derivs: " << coeffs[1][3].derivatives().transpose();
//	//cout << endl << "% coeff[2][0] derivs: " << coeffs[2][0].derivatives().transpose();
//	//cout << endl << "% coeff[2][1] derivs: " << coeffs[2][1].derivatives().transpose();
//	//cout << endl << "% coeff[2][2] derivs: " << coeffs[2][2].derivatives().transpose();
//	//cout << endl << "% coeff[2][3] derivs: " << coeffs[2][3].derivatives().transpose();
//	//exit(999);
//
//	// store data
//	delete [] Coefficients;
//	Coefficients = new double [4*size];
//	coeffDerivs.resize(size,4*size);
//	for(int i=0; i<size; ++i){ for(int j=0; j<4; ++j){
//		Coefficients[4*i+j]=coeffs[i][j].value();
//		coeffDerivs.col(4*i+j) = coeffs[i][j].derivatives();
//	}}
//
//
//	// update compute time
//	ComputeTime = GetMTime();
//}
