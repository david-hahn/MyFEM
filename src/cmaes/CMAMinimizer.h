#pragma once

#include <Eigen/Eigen>
#include "SensitivityAnalysis.h"
#include "LinearFEM.h"

typedef Eigen::VectorXd dVector;
class CMAObjectiveFunction{ public:
	MyFEM::SensitivityAnalysis& sa;
	Eigen::VectorXd unused;
	std::vector<double> fixedParamValues;
	std::vector<unsigned int> fixedParamIDs;
	CMAObjectiveFunction( MyFEM::SensitivityAnalysis& sa_ ) : sa(sa_) {
		fixedParamIDs.clear();
		fixedParamValues.clear();
	}
	CMAObjectiveFunction( MyFEM::SensitivityAnalysis& sa_, std::vector<double>& fixedValues, std::vector<unsigned int>& fixedIDs ) : sa(sa_) {
		fixedParamIDs = fixedIDs;
		fixedParamValues = fixedValues;
	}
	double evaluate(dVector& p){
		for(int i=0; i<fixedParamIDs.size(); ++i) p[fixedParamIDs[i]] = fixedParamValues[i];
		return sa(p,unused);
	}
};

/**
	Minimizes a given function using the Covariance Matrix Adaptation evolution strategy method (CMA-ES).
	The function can have any form and can be noisy, no gradients are required.

	NOTE: Call setBounds(...) before calling the minimize(...) function. The given lower and upper bounds will
	be used to scale the state vector p to assume the same distribution over all dimensions, which impacts the performance.
	Also, provide an initial guess for the values of p.
*/
class CMAMinimizer {
public:
	/**
		Verbosity level.
		0: No output (default).
		1: Only results are displayed.
		2: Output intermediate results at each iteration.
	*/
	int printLevel;

public:
	/**
		Constructor.
		- maxIterations: The maximum number of iterations taken by CMA. The maximum number of function iterations is maxIterations * populationSize.
		- populationSize: The number of sampling points taken at each CMA iteration. Choose a larger number to sample more densely within one generation.
		- initialStdDev: Initial value for standard deviation. CMA adjusts these values in the process. However, the initial value can impact performance. If you have confidence in your initial guess, choose a small value.
		- solveFunctionValue: The absolute objective function value at which CMA stops.
		- solveHistToleranceValue: Stop if the maximum function value difference of all iteration-best solutions of the last 10 + 30*N/lambda iterations become smaller than solveHistToleranceValue
	*/
	CMAMinimizer(int maxIterations = 5000, int populationSize = 16, double initialStdDev = 0.05, double solveFunctionValue = 0.001, double solveHistToleranceValue = 1e-13);
	virtual ~CMAMinimizer();
	
	/**
		Provides an interval [pMin, pMax] for parameter values p. p is scaled to [0,1] based on these value, which apparently increases CMA performance.
	*/
	void setBounds(const dVector &pMin, const dVector &pMax);

	/**
		Minimizes the objective function.
		NOTE: p should be initialized to a valid initial guess.
	*/
	virtual bool minimize(CMAObjectiveFunction *function, dVector &p, double &functionValue);
	virtual bool minimize(CMAObjectiveFunction *function, dVector &p, const dVector& initialStdDev, double stdDevModifier, double &functionValue);

protected:
	dVector pMin, pMax;
	int maxIterations;
	int populationSize;
	double solveFunctionValue;
	double solveHistToleranceValue;
	double initialStdDev;
};
