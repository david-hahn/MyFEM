#include "DirectSensitivity.h"
#include "LinearFEM.h"
#include "ContactFEM.h"
#include "Materials.h"
#include "BoundaryFieldObjectiveFunction.h"
#include "InitialConditionParameterHandler.h"

#include <vtkRenderWindow.h>
#include <omp.h> // for timing

using namespace MyFEM;

double SensitivityAnalysis::operator()(const Eigen::VectorXd& q, Eigen::VectorXd& dphi_dq){
	if( wnd!=NULL ) wnd->reset();
	double phi;
	if( evalCounter==0 ){
		t0 = omp_get_wtime();
		bestRunPhiValue = DBL_MAX;
		bestRunParameters.resizeLike( q );
		bestRunGradient.resizeLike( q );
	}

	fem.reset();
	qHdl.setNewParams(q, fem);
	phiQ = 0.0; // keep track of regularizer contribution to objective function in solver functions ...
	if( numberOfTimesteps>0 )
		phi = solveImplicitDynamicFEM(dphi_dq);
	else
		phi = solveStaticFEM(dphi_dq);


	if( fem.doPrintResiduals && numberOfTimesteps>0 ) printf("\n\n%% *************************\n");
	printf("\n phi = %10.4lg; t = %.2lf %% (%10.4lg), reg.fcn. %6.2lg, grad.norm %10.4lg ", phi, omp_get_wtime()-t0, evalCounter>0?(bestRunPhiValue-phi)/phi:1.0 ,phiQ, dphi_dq.norm() );
	if( q.size()<=20 ){
		std::cout << endl << "      q = [ " <<       q.transpose() << " ]; ";
		std::cout << endl << "dphi_dq = [ " << dphi_dq.transpose() << " ]; ";
	}else{
		std::cout << endl << "      q = [ " <<       q.segment<20>(0).transpose() << " ]; %% ... ";
		std::cout << endl << "dphi_dq = [ " << dphi_dq.segment<20>(0).transpose() << " ]; %% ... ";
	}
	if( numberOfTimesteps>0 ) printf("\n\n%% *************************\n");

	if( phi < bestRunPhiValue ){
		bestRunPhiValue = phi;
		bestRunParameters = q;
		bestRunGradient = dphi_dq;

		if( numberOfTimesteps==0 ){
			fem.setResetPoseFromCurrent(); // for static optimization: update reset pose to best yet found
			printf("\n%% Static reset pose updated ");
		}
	}
	++evalCounter;
	return phi;
}

double SensitivityAnalysis::finiteDifferenceTest(const Eigen::VectorXd& params, const double fd_h){
	printf("\n%% FD-check for %u params at step size %lg ... ", params.size(), fd_h);
	Eigen::VectorXd q = params, dphi_dq(params.size()), tmp(params.size());
	double maxErr=-1.0;
	unsigned int maxErrId=0;
	double test, phiVal = (*this)(params, dphi_dq);
	for(unsigned int id=0; id<params.size(); ++id){
		printf("\n%% FD-check %u / %u ...", id+1, params.size());
		q(id) = params(id)+fd_h;
		test = (*this)(q, tmp);
		q(id) = params(id);
		test = (test-phiVal)/fd_h;
		printf("%% ... FD grad %.4lg", test);
		test = std::abs(test-dphi_dq(id));
		printf(", SA grad %.4lg, abs.err. %.4lg, rel.err. %.4lg\n",dphi_dq(id),test,test/std::abs(dphi_dq(id)));
		if( test > maxErr){
			maxErr = test;
			maxErrId = id;
		}
	}
	printf("\n%% FD-check max err. is %.3lg (rel %.3lg) in parameter %u \n\n", maxErr, maxErr/std::abs(dphi_dq(maxErrId)), maxErrId);
	return maxErr;
}

void SensitivityAnalysis::setupDynamicSim( double dt, unsigned int tsteps, bool staticSolve, std::string fName, MeshPreviewWindow* wndPtr, double solverEps_ ){
	timestep = dt;
	numberOfTimesteps = tsteps;
	outFileName = fName;
	wnd = wndPtr;
	startWithStaticSolve = staticSolve;
	solverEps = solverEps_;
	bestRunPhiValue=DBL_MAX;
}

void SensitivityAnalysis::setupStaticPreregularizer(double regularize, unsigned int maxPreIters_, double preEps_){ preReg=regularize; maxPreIters=maxPreIters_; preEps=preEps_; }

double SensitivityAnalysis::solveImplicitDynamicFEM(Eigen::VectorXd& dphi_dq){
	std::stringstream fileName;
	if( solverEps<0.0 ) solverEps=LinearFEM::FORCE_BALANCE_EPS;

	double phiVal=0.0;
	dphi_dq.resize( qHdl.getNumberOfParams(fem) );
	dphi_dq.setZero();

	dynamicSimStart();

	// Run the forward sim
	fem.assembleMassAndExternalForce();

	if( startWithStaticSolve ){
		if( fem.doPrintResiduals ) printf("\n%% init: ");
		fem.updateAllBoundaryData();
		//fem.staticSolve(0.0);
		fem.staticSolvePreregularized(preEps,preReg,maxPreIters,solverEps);

		dynamicSimStaticInit();

		if( wnd!=NULL ) wnd->render();
	}

	for(int step=0; step<numberOfTimesteps; ++step){
		if( fem.doPrintResiduals ) printf("\n%% step %5d: ", step+1);
		if(!outFileName.empty()){
			fileName.str(""); fileName.clear();
			fileName << outFileName << "_" << std::setfill ('0') << std::setw(5) << step;
			fem.saveMeshToVTUFile( fileName.str(), step==0 );
		}

		fem.updateAllBoundaryData();
		if( dynamicImplicitTimeStep(step, timestep, phiVal, dphi_dq, solverEps) < 0 ){
			return VTK_DOUBLE_MAX;
		}

		if( wnd!=NULL ) wnd->render();
	}
	if(!outFileName.empty()){
		fileName.str(""); fileName.clear();
		fileName << outFileName << "_" << std::setfill ('0') << std::setw(5) << numberOfTimesteps;
		fem.saveMeshToVTUFile( fileName.str(), true);

		if( dynamic_cast<AverageBoundaryValueObjectiveFunction*>(&phi) != NULL ){
			fileName.str(""); fileName.clear();
			fileName << outFileName << "_trackedBCs_" << evalCounter;
			((AverageBoundaryValueObjectiveFunction&) phi).writeTrackedLocations(fileName.str(),numberOfTimesteps);
		}
		
	}

	dynamicSimFinalize(phiVal, dphi_dq);

	if( isnan(phiVal) || !dphi_dq.allFinite() ){
		return VTK_DOUBLE_MAX;
	}

	return phiVal;
}





double DirectSensitivity::solveStaticFEM(Eigen::VectorXd& dphi_dq){
	if( solverEps<0.0 ) solverEps=LinearFEM::FORCE_BALANCE_EPS;
	if( qHdl.useSparseDgDq() ){
		printf("\n%% SPARSE PARAMETER HANDLER NOT IMPLEMENTED FOR DirectSensitivity.\n"); return -1.0;
	}

	fem.assembleMassAndExternalForce();
	fem.updateAllBoundaryData();
	if( fem.doPrintResiduals ) printf("\n%% static solve: ");
	if( fem.staticSolvePreregularized(preEps,preReg,maxPreIters,solverEps) < 0 ) return DBL_MAX;

	// evaluate the objective function
	Eigen::VectorXd phi_x, phi_f, phi_q, unused, phiQ_q;
	double phiVal = phi.evaluate(fem, phi_x, unused, phi_f, phi_q);

	// evaluate constraint derivatives (~> force change wrt. params)
	Eigen::MatrixXd g_q;
	phiVal += qHdl.computeConstraintDerivatives(g_q, phiQ_q, fem);

	if( phi_q.size()       ==0 ) phi_q  = phiQ_q; // first is empty, overwrite
	else if( phiQ_q.size() !=0 ) phi_q += phiQ_q; // both are allocated

	if( phi_x.size()>0 ){ // objective function need not allocate unused components ...
		dx_dq.resize( qHdl.getNumberOfDOFs(fem),  qHdl.getNumberOfParams(fem) );
		// apply Dirichlet boundaries to g_q
		//ToDo: support parameter-dependent boundary conditions
		for(std::map<unsigned int,double>::iterator it=fem.diriData.begin(); it!=fem.diriData.end(); ++it){
			g_q.row(it->first).setZero();
		}
		//SparseSolver s(fem.S);
		//dx_dq = s.solve(g_q);
		dx_dq = fem.linearSolver.solve(g_q); // re-use stored solver
		dphi_dq = phi_x.transpose() * dx_dq;
	}else{
		dphi_dq.resize( qHdl.getNumberOfParams(fem) );
		dphi_dq.setZero();
	}
	//ToDo: account for phi_f ... test with force-dependent objective function ...
	//      same for phi_q although that should be correct already
		
	if( phi_q.size()>0 ) dphi_dq += phi_q; // objective function need not allocate unused components ...

	//ToDo: generalize second derivative objective functions to common base class or something ...
	if( dynamic_cast<BoundaryFieldObjectiveFunction*>(&phi)!=NULL ){
		BoundaryFieldObjectiveFunction& bfPhi = (BoundaryFieldObjectiveFunction&)phi;
		if( bfPhi.phi_xx.size()>0 ){
			H.resize( qHdl.getNumberOfParams(fem) , qHdl.getNumberOfParams(fem) );
			H.setZero();
			if( bfPhi.phi_xx.size()>0 ) H += dx_dq.transpose() * bfPhi.phi_xx.asDiagonal() * dx_dq;
		}
	}
	if( H.size()>0 && qHdl.phiQ_qq.size()==H.rows() ) H += qHdl.phiQ_qq.asDiagonal();

	return phiVal;
}

void DirectSensitivity::dynamicSimStart(){
	SensitivityAnalysis::dynamicSimStart();

	//ToDo: generalize second derivative objective functions to common base class or something ...
	if( dynamic_cast<BoundaryFieldObjectiveFunction*>(&phi)!=NULL ){
		H.resize( qHdl.getNumberOfParams(fem) , qHdl.getNumberOfParams(fem) );
		H.setZero();
	}else
		H.resize(0,0);

	dx_dq.resize( qHdl.getNumberOfDOFs(fem),  qHdl.getNumberOfParams(fem) ); dx_dq.setZero();
	dv_dq.resize( qHdl.getNumberOfDOFs(fem),  qHdl.getNumberOfParams(fem) ); dv_dq.setZero();

	InitialConditionParameterHandler* icQhdl=dynamic_cast<InitialConditionParameterHandler*>(&qHdl);
	if( icQhdl!=NULL ){
		icQhdl->applyInitialConditions(fem);
		icQhdl->computeInitialDerivatives(dx_dq, dv_dq, fem);
	}

	if( fem.useBDF2 ){
		//x_old = x - dt*v;
		//v_old = v; // assume zero acceleration for the start
		dx_dq_old=dx_dq - timestep*dv_dq;
		dv_dq_old=dv_dq;
		H_old=H;
	}
}

void DirectSensitivity::dynamicSimStaticInit(){
	SensitivityAnalysis::dynamicSimStaticInit();

	Eigen::MatrixXd static_g_q;
	Eigen::VectorXd unused; // we only need dx_dq here, the static solution is not directly part of the objective function evaluation
	static_g_q.resize( qHdl.getNumberOfDOFs(fem), qHdl.getNumberOfParams(fem) );
	qHdl.computeConstraintDerivatives(static_g_q, unused, fem);
	for(std::map<unsigned int,double>::iterator it=fem.diriData.begin(); it!=fem.diriData.end(); ++it){
		static_g_q.row(it->first).setZero();
	}
	dx_dq += fem.linearSolver.solve(static_g_q); // re-use stored solver

	//should we do this?? Probably not, as we don't evaluate the objective function on the initial state, so phi_xx and phi_vv will be formally zero here (probably empty in memory)
	//if( dynamic_cast<BoundaryFieldObjectiveFunction*>(&phi)!=NULL ){
	//	BoundaryFieldObjectiveFunction& bfPhi = (BoundaryFieldObjectiveFunction&)phi;
	//	if( bfPhi.phi_xx.size()>0 ) H += dx_dq.transpose() * bfPhi.phi_xx.asDiagonal() * dx_dq;
	//	if( bfPhi.phi_vv.size()>0 ) H += dv_dq.transpose() * bfPhi.phi_vv.asDiagonal() * dv_dq;
	//	if( qHdl.phiQ_qq.size()>0 ) H += qHdl.phiQ_qq.asDiagonal();
	//}

	if( fem.useBDF2 ){
		dx_dq_old = dx_dq;
		dv_dq_old = dv_dq;
		H_old     = H;
	}
}

#include <vtkDoubleArray.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPointData.h>

int DirectSensitivity::dynamicImplicitTimeStep(unsigned int step, double timestep, double& phiVal, Eigen::VectorXd& dphi_dq, double eps){
	if( qHdl.useSparseDgDq() ){
		printf("\n%% SPARSE PARAMETER HANDLER NOT IMPLEMENTED FOR DirectSensitivity.\n"); return -1.0;
	}

	int r = fem.dynamicImplicitTimestep(timestep,eps);
	if( r<0 ) return r;

	Eigen::VectorXd phiQ_q, phi_x, phi_v, phi_f, phi_q;
	Eigen::MatrixXd g_q( qHdl.getNumberOfDOFs(fem), qHdl.getNumberOfParams(fem) );

	double tmp = timestep* qHdl.computeConstraintDerivatives(g_q, phiQ_q, fem);
	phiQ += tmp;
	phiVal += tmp;
	phiVal += timestep* phi.evaluate(fem, phi_x, phi_v, phi_f, phi_q);

	if( phi_q.size()==0 ) phi_q  = phiQ_q; // first is empty, overwrite
	else if( phiQ_q.size() !=0 ) phi_q += phiQ_q; // both are allocated, sum up

	Eigen::MatrixXd rhsMatrix;
	if( fem.useBDF2 ){
			//g = (f+f_ext) - (M*(1.5*vi-1.5*v))/dt;
			//if( v_old.size()>0 ) g -= (M*(0.5*v_old-0.5*v))/dt; // assume v==v_old otherwise
		rhsMatrix =  g_q - fem.K * dx_dq + 1.5* fem.M * dv_dq / timestep;
		rhsMatrix += 0.5*fem.M * (dv_dq-dv_dq_old) / timestep - 1.0/3.0*fem.K * (dx_dq-dx_dq_old);

		// set dx_dq_old to dx_dq and dx_dq to 4/3 dx_dq - 1/3 dx_dq_old -- later we'll add the dv_dq term
		Eigen::MatrixXd tmp = 4.0/3.0*dx_dq - 1.0/3.0*dx_dq_old;
		dx_dq_old = dx_dq;
		dx_dq = tmp;
		dv_dq_old = dv_dq;
	}else
		rhsMatrix =  g_q - fem.K * dx_dq + fem.M * dv_dq / timestep;

	for(std::map<unsigned int,double>::iterator it=fem.diriData.begin(); it!=fem.diriData.end(); ++it){
		rhsMatrix.row(it->first).setZero();
	}

	if( dynamic_cast<ContactFEM*>(&fem)!=NULL ){ // ToDo: maybe move this part to a separate class for contact sensitivity?
		ContactFEM& cFEM = (ContactFEM&)fem;
		if( cFEM.method == cFEM.CONTACT_HYBRID ){
			// in CONTACT_HYBRID method we have hard constraints for stick DOFs that need to be treated here
			Eigen::MatrixXd rRhs = cFEM.R * rhsMatrix;
			// apply RHS stick constraints to RHS
			for(std::set<unsigned int>::iterator it=cFEM.stickDOFs.begin(); it!=cFEM.stickDOFs.end(); ++it){
				rRhs.row(*it).setZero();
			}

			dv_dq = cFEM.linearSolver.solve(rRhs); // linear solver contains R*S*R.transpose()
			dv_dq = (cFEM.R.transpose()*dv_dq).eval();

		}else{
			dv_dq = fem.linearSolver.solve(rhsMatrix); // default version
		}
	}else{
		dv_dq = fem.linearSolver.solve(rhsMatrix); // re-use stored solver - contains SparseMatrixD S = fem.K*timestep + fem.M/timestep; if( fem.D.size()>0 ) S += fem.D;
	}
	
	if( fem.useBDF2 ){
		dx_dq += 2.0/3.0*timestep * dv_dq;
	}else{
		dx_dq += timestep * dv_dq;
	}


	//ToDo: generalize second derivative objective functions to common base class or something ...
	if( dynamic_cast<BoundaryFieldObjectiveFunction*>(&phi)!=NULL ){
		BoundaryFieldObjectiveFunction& bfPhi = (BoundaryFieldObjectiveFunction&)phi;
		if( bfPhi.phi_xx.size()>0 || bfPhi.phi_vv.size()>0 ){
			if( fem.useBDF2 ){
				Eigen::MatrixXd Hi = H;
				if( bfPhi.phi_xx.size()>0 ) H += 2.0/3.0*timestep* dx_dq.transpose() * bfPhi.phi_xx.asDiagonal() * dx_dq;
				if( bfPhi.phi_vv.size()>0 ) H += 2.0/3.0*timestep* dv_dq.transpose() * bfPhi.phi_vv.asDiagonal() * dv_dq;
				if( qHdl.phiQ_qq.size()==H.rows() ) H += 2.0/3.0*timestep* qHdl.phiQ_qq.asDiagonal();
				if( H_old.size()>0 ) H += 1.0/3.0*(Hi-H_old);
				H_old = Hi;
			}else{
				if( bfPhi.phi_xx.size()>0 ) H += timestep* dx_dq.transpose() * bfPhi.phi_xx.asDiagonal() * dx_dq;
				if( bfPhi.phi_vv.size()>0 ) H += timestep* dv_dq.transpose() * bfPhi.phi_vv.asDiagonal() * dv_dq;
				if( qHdl.phiQ_qq.size()==H.rows() ) H += timestep* qHdl.phiQ_qq.asDiagonal();
			}
		}
	}

	//cout << endl << "phi_x size " << phi_x.rows() << " x " << phi_x.cols();
	//cout << endl << "dx_dq size " << dx_dq.rows() << " x " << dx_dq.cols();
	//cout << endl << "prod. size " << (phi_x.transpose() * dx_dq).rows() << " x " << (phi_x.transpose() * dx_dq).cols();

	// objective functions need not allocate storage for unused derivatives, add only the ones that exist ...
	// do not use BDF2 formula here even if FEM uses it, as the objective function value is still integrated with a plain rectangular rule
	if( phi_q.size() > 0) dphi_dq += timestep* phi_q;
	if( phi_x.size() > 0) dphi_dq += timestep* phi_x.transpose() * dx_dq;
	if( phi_v.size() > 0) dphi_dq += timestep* phi_v.transpose() * dv_dq;
	if( phi_f.size() > 0) dphi_dq += timestep* phi_f.transpose() * g_q;
	//cout << endl << "% dphi_dq tstep = " << dphi_dq.transpose();

	unsigned int qidx=0;
	vtkSmartPointer<vtkDoubleArray> vtkDx = vtkDoubleArray::SafeDownCast( fem.mesh->GetPointData()->GetAbstractArray("dx_dqi") );
	if( vtkDx==NULL ){
		vtkDx = vtkDoubleArray::New();
		vtkDx->SetName("dx_dqi");
		vtkDx->SetNumberOfComponents(3);
		vtkDx->SetNumberOfTuples(fem.getNumberOfNodes());
		fem.mesh->GetPointData()->AddArray(vtkDx);
	}	
	for(unsigned int i=0; i<fem.getNumberOfNodes(); ++i){
		vtkDx->SetTuple3(i, dx_dq( fem.getNodalDof(i,fem.X_DOF), qidx ), dx_dq( fem.getNodalDof(i,fem.Y_DOF), qidx ), dx_dq( fem.getNodalDof(i,fem.Z_DOF), qidx ) );
		//vtkDx->SetTuple3(i, // special for paper intro animation
		//	dx_dq( fem.getNodalDof(i,fem.X_DOF), qidx )*0.0 +0.8*dx_dq( fem.getNodalDof(i,fem.X_DOF), qidx+1 ) -0.3*dx_dq( fem.getNodalDof(i,fem.X_DOF), qidx+2 ), 
		//	dx_dq( fem.getNodalDof(i,fem.Y_DOF), qidx )*0.0 +0.8*dx_dq( fem.getNodalDof(i,fem.Y_DOF), qidx+1 ) -0.3*dx_dq( fem.getNodalDof(i,fem.Y_DOF), qidx+2 ), 
		//	dx_dq( fem.getNodalDof(i,fem.Z_DOF), qidx )*0.0 +0.8*dx_dq( fem.getNodalDof(i,fem.Z_DOF), qidx+1 ) -0.3*dx_dq( fem.getNodalDof(i,fem.Z_DOF), qidx+2 ) );
	}


	return r;
}
