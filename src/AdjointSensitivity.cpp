#include "AdjointSensitivity.h"
#include "BoundaryFieldObjectiveFunction.h"
#include "InitialConditionParameterHandler.h"
#include "Materials.h"
#include "LinearFEM.h"
#include "ContactFEM.h"

#include <vtkRenderWindow.h>

using namespace MyFEM;

double AdjointSensitivity::solveStaticFEM(Eigen::VectorXd& dphi_dq){
	if( solverEps<0.0 ) solverEps=LinearFEM::FORCE_BALANCE_EPS;
	fem.assembleMassAndExternalForce();
	fem.updateAllBoundaryData();
	if( fem.doPrintResiduals ) printf("\n%% static solve: ");
	//if( fem.staticSolve(0.0 /*0.0 == do not regularize the stiffness*/) < 0 ) return DBL_MAX;
	if( fem.staticSolvePreregularized(preEps,preReg,maxPreIters,solverEps) < 0 ) return DBL_MAX;

	// evaluate the objective function
	Eigen::VectorXd phi_x, phi_f, phi_q, unused, phiQ_q;
	double phiVal = phi.evaluate(fem, phi_x, unused, phi_f, phi_q);

	// evaluate constraint derivatives (~> force change wrt. params)
	Eigen::MatrixXd g_q; SparseMatrixD spg_q;
	if( qHdl.useSparseDgDq() ){
		phiVal += qHdl.computeConstraintDerivatives(spg_q, phiQ_q, fem);
	}else{
		phiVal += qHdl.computeConstraintDerivatives(g_q, phiQ_q, fem);
	}
	if( phi_q.size()       ==0 ) phi_q  = phiQ_q; // first is empty, overwrite
	else if( phiQ_q.size() !=0 ) phi_q += phiQ_q; // both are allocated

	if( phi_x.size()>0 ){ // objective function need not allocate unused components ...
		// compute adjoint state
		lambdaX.resize( qHdl.getNumberOfDOFs(fem) );
		// -lambdaX = (K^T)^-1 phi_x
		// apply Dirichlet boundaries to phi_x
		for(std::map<unsigned int,double>::iterator it=fem.diriData.begin(); it!=fem.diriData.end(); ++it){
			phi_x(it->first)=0.0;
		}
		if( assumeSymmetricMatrices ){
			lambdaX = fem.linearSolver.solve( phi_x ); // re-use stored solver
		}else{
			SparseSolver s( fem.S.transpose() );
			lambdaX = s.solve( phi_x );
		}
		// compute objective function sensitivity on parameters -lambdaX^T (dg/dq)
		if( qHdl.useSparseDgDq() ){
			dphi_dq = lambdaX.transpose()*spg_q;
		}else{
			dphi_dq = lambdaX.transpose()*g_q; // .transpose() here should not be a performance issue as it is a vector anyway (used as row-vector) - no change to memory access (?)
		}
	}else{
		dphi_dq.resize( qHdl.getNumberOfParams(fem) );
		dphi_dq.setZero();
	}
	//ToDo: account for phi_f ... test with force-dependent objective function ...
	//      same for phi_q although that should be correct already
		
	if( phi_q.size()>0 ) dphi_dq += phi_q; // objective function need not allocate unused components ...

	return phiVal;
}

void AdjointSensitivity::dynamicSimStart(){
	InitialConditionParameterHandler* icQhdl=dynamic_cast<InitialConditionParameterHandler*>(&qHdl);
	if( icQhdl!=NULL ){	//printf("\n%% AdjointSensitivity::dynamicSimStart -- applying initial conditions ");
		icQhdl->applyInitialConditions(fem);
		dx0_dq.resize( qHdl.getNumberOfDOFs(fem),  qHdl.getNumberOfParams(fem) ); dx0_dq.setZero();
		dv0_dq.resize( qHdl.getNumberOfDOFs(fem),  qHdl.getNumberOfParams(fem) ); dv0_dq.setZero();
		icQhdl->computeInitialDerivatives(dx0_dq, dv0_dq, fem);
	}
}

void AdjointSensitivity::dynamicSimStaticInit(){
	SensitivityAnalysis::dynamicSimStaticInit();

	staticK = fem.getStiffnessMatrix();
	Eigen::VectorXd unused;
	if( qHdl.useSparseDgDq() ){
		static_spg_q.resize( qHdl.getNumberOfDOFs(fem), qHdl.getNumberOfParams(fem) );
		qHdl.computeConstraintDerivatives(static_spg_q, unused, fem);
	}else{
		static_g_q.resize( qHdl.getNumberOfDOFs(fem), qHdl.getNumberOfParams(fem) );
		qHdl.computeConstraintDerivatives(static_g_q, unused, fem);
	}
}

void AdjointSensitivity::dynamicSimFinalize(double& phiVal, Eigen::VectorXd& dphi_dq){
	SensitivityAnalysis::dynamicSimFinalize(phiVal, dphi_dq);

	// Run the adjoint sim ...
	lambdaX.resize( qHdl.getNumberOfDOFs(fem) ); lambdaX.setZero();
	lambdaV.resize( qHdl.getNumberOfDOFs(fem) ); lambdaV.setZero();

	if( fem.useBDF2 ){
		lambdaX_old = lambdaX; 
		lambdaV_old = lambdaV;
	}

	for(int step=(numberOfTimesteps-1); step>=0; --step){
		dynamicImplicitAdjointStep(step, phiVal,dphi_dq);
	}

	if( startWithStaticSolve ){
		fem.applyStaticBoundaryConditions(staticK,lambdaX,lambdaX,0.0);
		SparseSolver s( staticK );
		if( qHdl.useSparseDgDq() ){
			dphi_dq += s.solve(lambdaX).transpose() * static_spg_q;
		}else{
			dphi_dq += s.solve(lambdaX).transpose() * static_g_q;
		}
	}


	InitialConditionParameterHandler* icQhdl=dynamic_cast<InitialConditionParameterHandler*>(&qHdl);
	if( icQhdl!=NULL ){
		dphi_dq += lambdaX.transpose()*dx0_dq;
		dphi_dq +=(lambdaV.transpose()*Ms[0])*dv0_dq;
		if( fem.useBDF2 ){
			dphi_dq += 0.5*((lambdaX-lambdaX_old).transpose()*dx0_dq); // fairly sure this is correct
			dphi_dq += ((lambdaV-lambdaV_old).transpose()*Ms[0])*dv0_dq; // not so sure about this one -- probably ok though (certainly not 0.5*, 2/3*, 4/3*, or 1.5*)
		}
	}
}


int AdjointSensitivity::dynamicImplicitTimeStep(unsigned int step, double timestep, double& phiVal, Eigen::VectorXd& dphi_dq, double eps){
	int r = fem.dynamicImplicitTimestep(timestep,eps);
	if( r<0 ) return r;

	dts[step] = timestep;

	//ToDo: reduce storage if either matrix has not changed (linear elasticity -> K const., same mesh -> M const.)
	if( assumeSymmetricMatrices ){
		Ks[step] = fem.getStiffnessMatrix();
		Ms[step] = fem.getMassMatrix();
		Ds[step] = fem.getDampingMatrix();
	}else{
		Ks[step] = fem.getStiffnessMatrix().transpose();
		Ms[step] = fem.getMassMatrix().transpose();
		Ds[step] = fem.getDampingMatrix().transpose();
		if( dynamic_cast<ContactFEM*>(&fem)!=NULL ){ // ToDo: maybe move this part to a separate class for contact sensitivity?
			ContactFEM& cFEM = (ContactFEM&)fem;
			if( cFEM.method == cFEM.CONTACT_HYBRID ){
				Rs[step] = cFEM.R;
				stickDOFs[step] = cFEM.stickDOFs;
			}
		}
	}

	Eigen::VectorXd phiQ_q;
	//ToDo: for local (such as per-element) parameters q, g_q will be sparse -- consider using a sparse storage format (can we support both?)

	if( qHdl.useSparseDgDq() ){
		spg_qs[step] = SparseMatrixD( qHdl.getNumberOfDOFs(fem), qHdl.getNumberOfParams(fem) );
		phiVals[step] = qHdl.computeConstraintDerivatives(spg_qs[step], phiQ_q, fem);
	}else{
		g_qs[step] = Eigen::MatrixXd( qHdl.getNumberOfDOFs(fem), qHdl.getNumberOfParams(fem) );
		phiVals[step] = qHdl.computeConstraintDerivatives(g_qs[step], phiQ_q, fem);
	}

	// evaluate the objective function
	phi_xs[step] = Eigen::VectorXd();
	phi_vs[step] = Eigen::VectorXd();
	phi_fs[step] = Eigen::VectorXd();
	phi_qs[step] = Eigen::VectorXd();

	phiQ += dts[step]*phiVals[step];
	phiVals[step] += phi.evaluate(fem, phi_xs[step], phi_vs[step], phi_fs[step], phi_qs[step] );
	if( phi_qs[step].size()==0 ) phi_qs[step]  = phiQ_q; // first is empty, overwrite
	else if( phiQ_q.size() !=0 ) phi_qs[step] += phiQ_q; // both are allocated
	return r;
}

void AdjointSensitivity::dynamicImplicitAdjointStep(unsigned int step, double& phiVal, Eigen::VectorXd& dphi_dq){
	// integrate the adjoint states in time ...
	//MatLab version (FD checked) ... we'll again ignore all transpositions
    ////rv(freeDofs) = (M'/opt.dt + opt.dt*Kt{ti}) \ ( M'/opt.dt*rv(freeDofs) + rx(freeDofs) + opt.dt*dFdcc(freeDofs) );
    ////rx(freeDofs) = rx(freeDofs) + opt.dt*dFdcc(freeDofs) - opt.dt*Kt{ti}*rv(freeDofs);

	//ToDo: store linear solver object instead of rebuilding the matrix from scratch (more memory, less computation)
	SparseMatrixD S;
	if( fem.useBDF2 ){
		S = (1.5/dts[step])*Ms[step] + (2.0/3.0*dts[step])*Ks[step];
	}else
		S = (1.0/dts[step])*Ms[step] + dts[step]*Ks[step];
	if( Ds[step].size()>0 ) S+=Ds[step];

	Eigen::VectorXd g;

	//if( normalContactDOFs[step].size()>0 ){
	//	Eigen::VectorXd rl = Rs[step]*lambdaX;
	//	for(std::set<unsigned int>::iterator it=normalContactDOFs[step].begin(); it!=normalContactDOFs[step].end(); ++it){
	//		rl(*it)=0.0;
	//	}
	//	lambdaX=Rs[step].transpose()*rl;
	//}

	if( fem.useBDF2 ){
		g = (1.5/dts[step])*Ms[step]*lambdaV + lambdaX;
		g += (0.5/dts[step])*(Ms[step]*(lambdaV-lambdaV_old)) + (1.0/3.0)*(lambdaX-lambdaX_old);

		// update lambdaX with all terms except lambdaV, store old values
		Eigen::MatrixXd tmp = (4.0/3.0)*lambdaX-(1.0/3.0)*lambdaX_old;
		if( phi_xs[step].size()>0 ){
			g +=   (2.0/3.0*dts[step])*phi_xs[step]; // rhs done -- objective function need not allocate unused components ...
			tmp += (2.0/3.0*dts[step])*phi_xs[step];
		}

		lambdaX_old = lambdaX;
		lambdaX = tmp;

		lambdaV_old = lambdaV;
	}else{
		g = (1.0/dts[step])*Ms[step]*lambdaV + lambdaX;
		if( phi_xs[step].size()>0 ){
			g +=       dts[step]*phi_xs[step]; // objective function need not allocate unused components ...
			lambdaX += dts[step]*phi_xs[step];
		}
	}
	if( phi_vs[step].size()>0 ) g += phi_vs[step]; //ToDo: test if this is still correct if fem.useBDF2==true
	//ToDo: add phi_f ...

	//Dirichlet boundaries ...
	fem.applyStaticBoundaryConditions(S,g,g,0.0);
	//ToDo: should we allow boundary condition type or location to change during the sim???
	//      also, if we remesh, that will happen (we'd also need to store how to interpolate the adjoint states between meshes ...)

	if( dynamic_cast<ContactFEM*>(&fem)!=NULL ){ // ToDo: maybe move this part to a separate class for contact sensitivity?
		ContactFEM& cFEM = (ContactFEM&)fem;
		if( cFEM.method == cFEM.CONTACT_HYBRID ){
			// in CONTACT_HYBRID method we have hard constraints for stick DOFs that need to be treated here
			Eigen::VectorXd rr = Rs[step] * g;
			// apply RHS stick constraints to rr
			for(std::set<unsigned int>::iterator it=stickDOFs[step].begin(); it!=stickDOFs[step].end(); ++it){
				rr(*it)=0.0;
			}
			SparseMatrixD Sr = Rs[step]*S*Rs[step].transpose();
			// apply stick constraints to Sr
			for(Eigen::Index k=0; k < Sr.outerSize(); ++k){
				for(SparseMatrixD::InnerIterator it(Sr,k); it; ++it){
					if( stickDOFs[step].count(it.row()) ){
						it.valueRef() = (it.row()==it.col()) ? 1.0 : 0.0;
					}
				}
			}

			SparseSolver s(Sr);
			lambdaV = s.solve(rr); // linear solver contains R*S*R.transpose()
			lambdaV = (Rs[step].transpose()*lambdaV).eval();

		}else{
			SparseSolver s(S);
			lambdaV  = s.solve( g );
		}
	}else{
		SparseSolver s(S);
		lambdaV  = s.solve( g );
	}


	if( fem.useBDF2 ){
		lambdaX -= (2.0/3.0*dts[step])*Ks[step]*lambdaV;
	}else{
		lambdaX -= dts[step]*Ks[step]*lambdaV;
	}

	////lambdaX==0 on normalContactDOFs?
	//if( normalContactDOFs[step].size()>0 ){
	//	Eigen::VectorXd test = Rs[step]*lambdaX;
	//	printf("\n laX==0? ");
	//	for(std::set<unsigned int>::iterator it=normalContactDOFs[step].begin(); it!=normalContactDOFs[step].end(); ++it){
	//		printf(" %.2lg " , test(*it));
	//	}
	//}

	// integrate the objective function and gradient in time ...
	phiVal  += dts[step]*phiVals[step];
	if( qHdl.useSparseDgDq() ){
		dphi_dq += dts[step]*lambdaV.transpose()*spg_qs[step];
	}else{
		dphi_dq += dts[step]*lambdaV.transpose()*g_qs[step];
	}
	if( phi_qs[step].size()>0 ) dphi_dq += dts[step]*phi_qs[step]; // objective function need not allocate unused components ...
}

double ObjectiveFunction::evaluate( LinearFEM& fem,
	Eigen::VectorXd& phi_x, Eigen::VectorXd& phi_v, Eigen::VectorXd& phi_f, Eigen::VectorXd& phi_q
){
	// only allocate the storage we'll actually use ...
	phi_x.resize( fem.getRestCoords().size() );
	phi_x.setZero();
	// note if phi depends on internal forces f, there are two derivative chains to deal with: in addition to computing partial derivs phi_f, we also need to account for the contribution to phi_x += -K*phi_f (where K=f_x) - right??
	double val=0.0;

	// First start with a basic objective on the deformation: phi(x) = 0.5*sum (x-x0-p(x0))^2 to aim for target displacement uTarget ...
	Eigen::Vector3d u;
	for(unsigned int i=0; i<fem.getNumberOfNodes(); ++i){

		u = fem.getDeformedCoord(i) - fem.getRestCoord(i) -uTarget;

		for(int dof=0; dof<LinearFEM::N_DOFS; ++dof){
			unsigned int idof=fem.getNodalDof(i,dof);

			val += 0.5*u(dof)*u(dof); // phi += 1/2*(x-x0-p)^2
			phi_x[idof] += u(dof); // d(phi)/dx += (x-x0-p) (= 1/2*1*(x-x0-p)+1/2*(x-x0-p)*1 )
		}
	}

	return val;
}

 unsigned int ParameterHandler::getNumberOfDOFs(const LinearFEM& fem){
	return fem.getRestCoords().size();
}