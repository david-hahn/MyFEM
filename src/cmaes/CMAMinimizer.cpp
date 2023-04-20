#include <cmaes/CMAMinimizer.h>

extern "C"
{
#include <cmaes/cmaes.h>
#include <cmaes/cmaes_interface.h>

	inline double linearlyInterpolate(double v1, double v2, double t1, double t2, double t){
		if (v1 == v2)
			return v2;
		return (t-t1)/(t2-t1) * v2 + (t2-t)/(t2-t1) * v1;
	}
}

CMAMinimizer::CMAMinimizer(int maxIterations, int populationSize, double initialStdDev, double solveFunctionValue, double solveHistToleranceValue)
	:	maxIterations(maxIterations),
		populationSize(populationSize),
		initialStdDev(initialStdDev),
		solveFunctionValue(solveFunctionValue),
		solveHistToleranceValue(solveHistToleranceValue),
		printLevel(0){
}

CMAMinimizer::~CMAMinimizer(){
}

void CMAMinimizer::setBounds(const dVector &pMin, const dVector &pMax){
	this->pMin = pMin;
	this->pMax = pMax;
}

bool CMAMinimizer::minimize(CMAObjectiveFunction *function, dVector &p, const dVector& startStdDev, double stdDevModifier, double &functionValue){
	// Make sure the bounds are set.
	if (pMin.size() != p.size() || pMax.size() != p.size()) {
		assert("Please call 'setBounds(...)' before calling the CMA minimizer function" == 0);
		return false;
	}

	// Initialize CMA with the current parameters, scaled to [0,1].
	//dVector pScaled((int)p.size(),0);
	dVector pScaled(p.size());
	pScaled.setZero();
	for (int i=0; i<p.size(); i++){
		//assert(pMin[i] < pMax[i]);
		//assert(pMin[i] <= p[i] + EPSILON);
		//assert(p[i] <= pMax[i] + EPSILON);

		pScaled[i] = (p[i] - pMin[i]) / (pMax[i] - pMin[i]);
	}

	// Map the initial parameters into the constraint manifold
	double initialFunctionValue = function->evaluate(p);

	if (printLevel >= 1) {
		//Logger::logPrint("   Initial function value: %.5lf\n", initialFunctionValue);
	}

	// Initialize the standard deviations
	dVector stdDev(startStdDev);
	stdDev *= stdDevModifier;

	// Initialize the CMA blackbox optimization
	cmaes_t evo;
	double *arFunVals = cmaes_init(&evo, (int)p.size(), &pScaled[0], &stdDev[0], 0, populationSize, "non");

	evo.sp.stopMaxFunEvals = 1e299;
	evo.sp.stStopFitness.flg = 1;
	evo.sp.stStopFitness.val = solveFunctionValue;
	evo.sp.stopMaxIter = maxIterations;
	evo.sp.stopTolFun = 1e-9;
	evo.sp.stopTolFunHist = solveHistToleranceValue;
	evo.sp.stopTolX = 1e-11;
	evo.sp.stopTolUpXFactor = 1e3;
	evo.sp.seed = 0;

	for (int i = 0;i < pScaled.size();i++)
		evo.rgxbestever[i] = pScaled[i];
	evo.rgxbestever[p.size()] = initialFunctionValue;
	evo.rgxbestever[p.size()+1] = 1;

	assert(IS_EQUAL(cmaes_Get(&evo, "fbestever"), initialFunctionValue));

	//dVector pi((int)p.size(), 0);
	dVector pi(p.size());
	pi.setZero();

	int iter = 0;
	for (; !cmaes_TestForTermination(&evo); iter++)
	{
		// Sample the parameter space
		double *const *pop = cmaes_SamplePopulation(&evo);
		int popSize = (int)cmaes_Get(&evo, "popsize");
		
		if (printLevel >= 2) {
			//Logger::logPrint("Starting iteration %d\n", iter);
		}

		for (int popIdx=0; popIdx<popSize; popIdx++){
			for (int i=0; i<p.size(); i++)
				pi[i] = (1 - pop[popIdx][i])*pMin[i] + (pop[popIdx][i])*pMax[i];
			
			// Evaluate the objective for each sampling point.
			arFunVals[popIdx] = function->evaluate(pi);
		}

		// Update the distribution
		cmaes_UpdateDistribution(&evo, arFunVals);

		// Print output
		if (printLevel >= 2)
		{
			dVector pTmp((int)p.size(),0);

			cmaes_GetInto(&evo, "xmean", &pTmp[0]);
			for (int i=0; i<p.size(); i++)
				pi[i] = (1 - pTmp[i])*pMin[i] + (pTmp[i])*pMax[i];
			double mean = function->evaluate(pi);

			cmaes_GetInto(&evo, "xbest", &pTmp[0]);
			for (int i=0; i<p.size(); i++)
				pi[i] = (1 - pTmp[i])*pMin[i] + (pTmp[i])*pMax[i];
			double best = function->evaluate(pi);

			cmaes_GetInto(&evo, "xbestever", &pTmp[0]);
			for (int i=0; i<p.size(); i++)
				pi[i] = (1 - pTmp[i])*pMin[i] + (pTmp[i])*pMax[i];
			double bestEver = function->evaluate(pi);

			//Logger::logPrint("       Mean function value: %.6lf\n", mean);
			//Logger::logPrint("       Best function value: %.6lf\n", best);
			//Logger::logPrint("   Bestever function value: %.6lf\n", bestEver);
		}
	}

	// Obtain the result and scale it back
	cmaes_GetInto(&evo, "xbestever", &pScaled[0]);
	for (int i=0; i<p.size(); i++)
		p[i] = linearlyInterpolate(pMin[i], pMax[i], 0, 1, pScaled[i]);
	functionValue = function->evaluate(p);

	if (printLevel >= 1) {
		//Logger::logPrint("CMA ended in %d/%d iterations\n", iter, maxIterations);
		//Logger::logPrint("   Function value improved from % .5lf\n", initialFunctionValue);
		//Logger::logPrint("                             to % .5lf\n", functionValue);

		//Logger::logPrint("CMA ended in %d/%d iterations\n", iter, maxIterations);

	}

	cmaes_exit(&evo);
	return iter < maxIterations;
}


bool CMAMinimizer::minimize(CMAObjectiveFunction *function, dVector &p, double &functionValue){
	if (printLevel >= 1) {
		//Logger::logPrint("Starting CMA minimization.\n");
		//Logger::logPrint("   Population size: %d, initial standard deviation: %.6lf\n", populationSize, initialStdDev);
	}
	
	dVector stdDev((int)p.size());
	stdDev.setConstant(initialStdDev);

	return minimize(function, p, stdDev, 1, functionValue);
}
