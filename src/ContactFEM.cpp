#include "ContactFEM.h"
#include "fieldNames.h"
#include <vtkUnstructuredGrid.h>
#include <vtkPoints.h>
#include <vtkDoubleArray.h>
#include <vtkIntArray.h>
#include <vtkPointData.h>

#include <set>

#ifdef  QPSOLVE_USE_MATLAB
#include <MatlabEngine.hpp>
#include <MatlabDataArray.hpp>
#elif QPSOLVE_USE_MOSEK
#include <mosek.h>
#elif _USE_OSQP
#include <OsqpEigen\OsqpEigen.h>
#endif


using namespace MyFEM;

// static
double ContactFEM::FORCE_BALANCE_DELTA = -1.0;
bool ContactFEM::printConvergenceDebugInfo=false;

ContactFEM::ContactFEM() : LinearFEM(){ epsD=0.0; method = CONTACT_CLAMP_PENALTY; }

ContactFEM::~ContactFEM(){}


int ContactFEM::dynamicImplicitTimestep(double dt, double eps){
	QPSolver qpSolver; //ToDo: store in member variable
	double qpEps = 1e-10 /*/ 1e-5 /**/; //MOSEK can't do much better than around 1e-5 ... not sure why

	switch( method ){
	case CONTACT_TANH_PENALTY:
		return dynamicImplicitTanhPenaltyContactTimestep(dt,eps);
	case CONTACT_CLAMP_PENALTY:
		return dynamicImplicitClampedLinearPenaltyContactTimestep(dt,eps);
	case CONTACT_HYBRID:
		return dynamicImplicitHybridPenaltyContactTimestep(dt,eps);
	case CONTACT_QP:
		return dynamicImplicitQPContactTimestep(qpSolver, dt, std::max(eps,qpEps) );
	case CONTACT_IGNORE:
	default:
		return LinearFEM::dynamicImplicitTimestep(dt,eps);
	}
}


int ContactFEM::dynamicImplicitTanhPenaltyContactTimestep(double dt, double eps){
	unsigned int max_iters = LinearFEM::MAX_SOLVER_ITERS;
	unsigned int max_linesearch = 20;

	if( useBDF2 && (x_old.size()==0 || v_old.size()==0) ){ // x_old or v_old for BDF2 integration need initializing ...
		x_old = x - dt*v;
		v_old = v; // assume zero acceleration for the start
	}

	double firstResidual=-1.0 ,rn;
	bool done=false; int iter;
	Eigen::VectorXd vi( v.size() );
	Eigen::VectorXd x0 = x, r;

	vi.setZero(); // x == x0 here
	if( useBDF2 ){
		//x = x0 + 2.0/3.0*dt*vi + 1.0/3.0*(x0-x_old);
		vi = 0.5*(x_old-x0)/dt;
	}


	assembleForceAndStiffness(); // overwrites f vector (does setZero() first), also writes K = -df/dx
	assembleViscousForceAndDamping( vi ); // adds to f vector - call assembleForceAndStiffness first! - also writes D = -df_visc/dv
	assembleTanhPenaltyForceAndStiffness(vi, useBDF2?2.0/3.0*dt:dt ); //adds to f and K and D - call assembleForceAndStiffness and assembleViscousForceAndDamping first!

	if( useBDF2 ){
		S =  2.0/3.0*K*dt + 1.5*M/dt;
		r = (f+f_ext) - (1.5/dt)*(M*(vi-v)) + 0.5*(M*(v-v_old))/dt;
	}else{
		S =  K*dt + M/dt;
		r = (f+f_ext) - (M*(vi-v))/dt;
	}
	//if( D.size()>0 ) S += D;
	applyDynamicBoundaryConditions(S,r,x0,vi,dt);
	linearSolver.compute(S);
			
	rn = freeDOFnorm(r);  //printf(" %8.3lg ", rn);
	firstResidual = rn;
	if( printConvergenceDebugInfo ) printf("\n ri = [ ");
	for(iter=0; iter<max_iters && !done; ++iter){
		if( printConvergenceDebugInfo ) printf("%.4lg ", rn);

		Eigen::VectorXd dv = linearSolver.solve( r );  //printf(" (s) ");
		if( !isfinite( dv.squaredNorm() )){ printf("\n%% LINEAR SOLVER FAILED! (search direction magnitude non-finite) \n"); return -2;}

		//line-search along dv ...
		Eigen::VectorXd vs( v.size() );
		double stepsize = 1.0, rs; int lstep=0;
		do{	
			vs = vi + stepsize*dv;

			if( useBDF2 )
				x = x0 + 2.0/3.0*dt*vs + 1.0/3.0*(x0-x_old);
			else
				x = x0+dt*vs;

			assembleForceAndStiffness( SKIP ); // overwrites f vector (does setZero() first), skip stiffness assembly
			assembleViscousForceAndDamping( vs, SKIP ); // adds to f vector - keep order of assembly fcn. calls! - skip damping matrix assembly
			assembleTanhPenaltyForceAndStiffness( vs, useBDF2?2.0/3.0*dt:dt, SKIP ); //adds to fem.f and fem.K - keep order of assembly fcn. calls!

			if( useBDF2 )
				r = (f+f_ext) - (1.5/dt)*(M*(vs-v)) + 0.5*(M*(v-v_old))/dt;
			else
				r = (f+f_ext) - (M*(vs-v))/dt;

			rs = freeDOFnorm(r); //printf(" [[%8.3lg]] ", rs);
			stepsize *= 0.5; ++lstep;
		}while( ( (!isfinite(rs)) || rs > rn ) && lstep < max_linesearch );

		vi = vs;

		// residual is already up to date, but we need to assemble the new matrices for the next iteration
		assembleForceAndStiffness(); // overwrites f vector (does setZero() first), also writes K = -df/dx
		assembleViscousForceAndDamping( vi ); // adds to f vector - call assembleForceAndStiffness first! - also writes D = -df_visc/dv
		assembleTanhPenaltyForceAndStiffness(vi, useBDF2?2.0/3.0*dt:dt ); //adds to f and K - call assembleForceAndStiffness first!
		if( useBDF2 ){
			S =  2.0/3.0*K*dt + 1.5*M/dt;
		}else{
			S =  K*dt + M/dt;
		}
		if( D.size()>0 ) S += D;
		applyDynamicBoundaryConditions(S,r,x0,vi,dt);
		linearSolver.compute(S); // even in the final iteration we need to update the linear solver because we'll re-use it in the sensitivity/adjoint integration

		if( rs < eps /**/|| std::abs(rn-rs) < FORCE_BALANCE_DELTA/**/ ){
			done = true;
		}
		rn = rs;  //printf(" %8.3lg ", rn);
	}
	if( printConvergenceDebugInfo ) printf("%.4lg ]; residuals(end+1,1:length(ri))=ri; %%", rn);

	if( useBDF2 )
		currentAcceleration = (1.5/dt)*(vi-v) - (0.5/dt)*(v-v_old);
	else
		currentAcceleration = (vi-v)/dt;
	if( useBDF2 ){ x_old=x0; v_old=v; }

	v=vi; // x is already set to end-of-timestep positions: x = x0+dt*v;
	simTime += dt;
	if( doPrintResiduals ) printf(" (%3d) %8.3lg (r%8.3lg) ", iter, rn, rn/firstResidual);
	mesh->GetPoints()->Modified(); // mark VTK structure as modified so rendering etc. can update properly
	if( !done && (rn > firstResidual) ) {if( doPrintResiduals ) printf(" !!! "); return -1;} // return -1 if not properly converged
	return iter;
}


int ContactFEM::dynamicImplicitClampedLinearPenaltyContactTimestep(double dt, double eps){
	unsigned int max_iters = LinearFEM::MAX_SOLVER_ITERS;
	unsigned int max_linesearch = 20;
	unsigned int ls_after_iter = 0;//10;//max_iters/2;// 

	if( useBDF2 && (x_old.size()==0 || v_old.size()==0) ){ // x_old or v_old for BDF2 integration need initializing ...
		x_old = x - dt*v;
		v_old = v; // assume zero acceleration for the start
	}

	double firstResidual=-1.0 ,rn, rs;
	bool done=false, flag=false; int iter;
	Eigen::VectorXd vi( v.size() ), fc( v.size() );
	Eigen::VectorXd x0 = x, r;

	vi.setZero(); // x == x0 here
	if( useBDF2 ){
		//x = x0 + 2.0/3.0*dt*vi + 1.0/3.0*(x0-x_old);
		vi = 0.5*(x_old-x0)/dt;
	}


	for(unsigned int oid=0; oid<rigidObstacles.size(); ++oid){
		contactStateByObstacle[oid].assign( getNumberOfNodes(), CONTACT_STATE::CONTACT_NONE ); // clear contact states at start of time step (set by assembleClampedLinearPenaltyForceAndStiffness)
	}

	assembleForceAndStiffness(); // overwrites f vector (does setZero() first), also writes K = -df/dx
	assembleViscousForceAndDamping( vi ); // adds to f vector - call assembleForceAndStiffness first! - also writes D = -df_visc/dv
	assembleClampedLinearPenaltyForceAndStiffness(fc, vi, 0, useBDF2?2.0/3.0*dt:dt ); //adds to f and K - call assembleForceAndStiffness first!

	if( useBDF2 ){
		S =  2.0/3.0*K*dt + 1.5*M/dt;
		r = (f+f_ext+fc) - (1.5/dt)*(M*(vi-v)) + 0.5*(M*(v-v_old))/dt;
	}else{
		S =  K*dt + M/dt;
		r = (f+f_ext+fc) - (M*(vi-v))/dt;
	}
	applyDynamicBoundaryConditions(S,r,x0,vi,dt);
	linearSolver.compute(S);

	rn = freeDOFnorm(r);  //printf(" %8.3lg ", rn);
	firstResidual = rn; rs = rn;
	if( printConvergenceDebugInfo ) printf("\n ri = [ ");
	for(iter=0; iter<max_iters && !done; ++iter){
		if( printConvergenceDebugInfo ) printf("%.4lg ", rs);

		Eigen::VectorXd dv = linearSolver.solve( r );  //printf(" (s) ");
		if( !isfinite( dv.squaredNorm() )){ printf("\n%% LINEAR SOLVER FAILED! (search direction magnitude non-finite) \n"); return -2;}

		//line-search along dv ...
		Eigen::VectorXd vs( v.size() );
		double stepsize = 1.0; int lstep=0;
		do{	
			vs = vi + stepsize*dv;

			if( useBDF2 )
				x = x0 + 2.0/3.0*dt*vs + 1.0/3.0*(x0-x_old);
			else
				x = x0+dt*vs;

			assembleForceAndStiffness( SKIP ); // overwrites f vector (does setZero() first), skip stiffness assembly
			assembleViscousForceAndDamping( vs, SKIP ); // adds to f vector - keep order of assembly fcn. calls! - skip damping matrix assembly
			flag |= assembleClampedLinearPenaltyForceAndStiffness(fc, vs, iter, useBDF2?2.0/3.0*dt:dt, SKIP );

			if( useBDF2 )
				r = (f+f_ext+fc) - (1.5/dt)*(M*(vs-v)) + 0.5*(M*(v-v_old))/dt;
			else
				r = (f+f_ext+fc) - (M*(vs-v))/dt;

			rs = freeDOFnorm(r); //printf(" [[%8.3lg]] ", rs);
			stepsize *= 0.5; ++lstep;
		}while( (iter>=ls_after_iter||rs>1e1*firstResidual) && rs > rn && lstep < max_linesearch );

		vi = vs;

		// residual is already up to date, but we need to assemble the new matrices for the next iteration
		assembleForceAndStiffness(); // overwrites f vector (does setZero() first), also writes K = -df/dx
		assembleViscousForceAndDamping( vi ); // adds to f vector - call assembleForceAndStiffness first! - also writes D = -df_visc/dv
		flag |= assembleClampedLinearPenaltyForceAndStiffness(fc, vi, iter, useBDF2?2.0/3.0*dt:dt );

		if( useBDF2 ){
			S =  2.0/3.0*K*dt + 1.5*M/dt;
		}else{
			S =  K*dt + M/dt;
		}
		if( D.size()>0 ) S += D;
		applyDynamicBoundaryConditions(S,r,x0,vi,dt);
		linearSolver.compute(S); // even in the final iteration we need to update the linear solver because we'll re-use it in the sensitivity/adjoint integration

		if( flag ) rn = firstResidual;
		if( !flag ){
			if( rs < eps || (iter>=(ls_after_iter+1) && std::abs(rn-rs) < FORCE_BALANCE_DELTA) ) done = true;
			rn = rs;
		}
		flag = false;
	}
	if( printConvergenceDebugInfo ) printf("%.4lg ]; residuals(end+1,1:length(ri))=ri; %%", rn);

	if( useBDF2 )
		currentAcceleration = (1.5/dt)*(vi-v) - (0.5/dt)*(v-v_old);
	else
		currentAcceleration = (vi-v)/dt;
	if( useBDF2 ){ x_old=x0; v_old=v; }

	// add contact forces to VTK output for debug
	vtkSmartPointer<vtkDoubleArray> vtkFc = vtkDoubleArray::SafeDownCast( mesh->GetPointData()->GetAbstractArray( fieldNames[CONTACT_FORCE_NAME].c_str() ) );
	if( vtkFc!=NULL ) for(unsigned int i=0; i<getNumberOfNodes(); ++i){
		unsigned int ixdof = getNodalDof(i,LinearFEM::X_DOF);
		vtkFc->SetTuple3(i,
			fc(ixdof  ),
			fc(ixdof+1),
			fc(ixdof+2)
		);
	}

	v=vi; // x is already set to end-of-timestep positions
	simTime += dt;
	if( doPrintResiduals ) printf(" (%3d) %8.3lg (r%8.3lg) ", iter, rn, rn/firstResidual);
	mesh->GetPoints()->Modified(); // mark VTK structure as modified so rendering etc. can update properly
	if( !done && (rn > firstResidual) ) {if( doPrintResiduals ) printf(" !!! "); return -1;} // return -1 if not properly converged
	return iter;
}


int ContactFEM::dynamicImplicitClassificationLinearPenaltyContactTimestep(double dt, double eps){
	unsigned int max_iters = LinearFEM::MAX_SOLVER_ITERS;

	if( useBDF2 && (x_old.size()==0 || v_old.size()==0) ){ // x_old or v_old for BDF2 integration need initializing ...
		x_old = x - dt*v;
		v_old = v; // assume zero acceleration for the start
	}

	double firstResidual=-1.0 ,rn;
	bool done=false; int iter; bool flag=false;
	Eigen::VectorXd vi( v.size() );
	Eigen::VectorXd x0 = x, fc( v.size() ), r; fc.setZero();

	std::vector<CONTACT_STATE> contactState; contactState.assign(getNumberOfNodes(), CONTACT_NONE); //ToDo: reduce allocated space to pre-counted number of boundary nodes
	std::vector<unsigned int> contactObstacle; contactObstacle.assign(getNumberOfNodes(), INT_MAX); //ToDo: reduce allocated space to pre-counted number of boundary nodes
	//ToDo: keep contactState across time steps for better initialization?

	vi.setZero(); // x == x0 here
	if( useBDF2 ){
		//x = x0 + 2.0/3.0*dt*vi + 1.0/3.0*(x0-x_old);
		vi = 0.5*(x_old-x0)/dt;
	}


	assembleForceAndStiffness(); // overwrites f vector (does setZero() first), also writes K = -df/dx
	assembleViscousForceAndDamping( vi ); // adds to f vector - call assembleForceAndStiffness first! - also writes D = -df_visc/dv
	assembleLinearPenaltyForceAndStiffness(fc, vi, contactState, contactObstacle, useBDF2?2.0/3.0*dt:dt); //overwrites fc and adds to K - call assembleForceAndStiffness first!

	if( useBDF2 ){
		S =  2.0/3.0*K*dt + 1.5*M/dt;
		r = (f+f_ext+fc) - (1.5/dt)*(M*(vi-v)) + 0.5*(M*(v-v_old))/dt;
	}else{
		S =  K*dt + M/dt;
		r = (f+f_ext+fc) - (M*(vi-v))/dt;
	}
	if( D.size()>0 ) S += D;
	applyDynamicBoundaryConditions(S,r,x0,vi,dt);
	linearSolver.compute(S);

	firstResidual = freeDOFnorm(r);  //printf(" %8.3lg ", rn);
	for(iter=0; iter<max_iters && !done; ++iter){

		vi += linearSolver.solve( r );  //printf(" (s) ");

		// enforce Coulomb limit
		flag = false;
		if( iter < (max_iters - 5) ){
			std::vector<bool> contactDone; contactDone.assign(getNumberOfNodes(), false); //ToDo: reduce allocated space to pre-counted number of boundary nodes
			for(unsigned int k = 0; k < getNumberOfBndryElems(); ++k) for(int j = 0; j<3; ++j){ //ToDo: keep a list of boundary nodes to iterate directly?
				unsigned int i = getBoundaryElement(k)(j);
				if( !contactDone[i] ){
					unsigned int prevContactObstacle = contactObstacle[i];
					unsigned int contactCount=0;
					double maxFs = -1.0, fTnorm;
					for(unsigned int oid=0; oid<rigidObstacles.size(); ++oid){
						Eigen::Vector3d n, fT; double fN;
						double g = rigidObstacles[oid]->eval(n,getRestCoord(i),getDeformedCoord(i),simTime); //ToDo: could store contact normals instead ...
						fT = fc.block<3,1>(getNodalDof(i,0),0);
						fN = n.dot(fT);
						fT -= n*fN; // now fT is the tangential part of fc and fN the normal part
						if( g < 0.0 || fN > 0.0){
							++contactCount;

							if( contactState[i] == CONTACT_SLIP && prevContactObstacle==oid ){ // check for back-slip
								if( fT.dot(  vi.block<3,1>(getNodalDof(i,0),0) )>DBL_EPSILON ){
									//if(i==10) printf("\n%% node %d backslip velocity %.2le", i,fT.dot(  vi.block<3,1>(getNodalDof(i,0),0) ) );
									contactState[i] = CONTACT_STICK; //printf("x");
									flag = true;
									fT.setZero();
								}
							}else
							if( contactState[i] == CONTACT_NONE ){
								contactState[i] = CONTACT_STICK;
							}
							// ... classify, set contactObstacle[i] to oid where node is most likely to stick (max fn*cf) ...
							//if( frictionCoefficient[oid]*fN > maxFs ){
							//	maxFs=frictionCoefficient[oid]*fN;
							if( frictionCoefficient[oid]*(-g*normalPenaltyFactor) > maxFs ){
								maxFs=frictionCoefficient[oid]*(-g*normalPenaltyFactor);
								fTnorm = fT.norm();
								contactObstacle[i]=oid;
							}
							if( iter<=3 ) flag=true;
						}
					}
					//if( contactCount == 0) contactState[i] = CONTACT_NONE;
					//else
					if( iter>3 && maxFs>0.0 && fTnorm > (maxFs) && contactState[i] != CONTACT_SLIP){
						contactState[i] = CONTACT_SLIP;
						flag = true; //printf("s");
						//if(i==10) printf("\n%% node %d slip limit %.2le", i,maxFs);
					}
					contactDone[i]=true;
				}
			}
		}

		if( useBDF2 )
			x = x0 + 2.0/3.0*dt*vi + 1.0/3.0*(x0-x_old);
		else
			x = x0+dt*vi;

		assembleForceAndStiffness(); // overwrites f vector (does setZero() first), also writes K = -df/dx
		assembleViscousForceAndDamping( vi ); // adds to f vector - call assembleForceAndStiffness first! - also writes D = -df_visc/dv
		assembleLinearPenaltyForceAndStiffness(fc, vi, contactState, contactObstacle, useBDF2?2.0/3.0*dt:dt ); //overwrites fc and adds to K - call assembleForceAndStiffness first!

		if( useBDF2 ){
			S =  2.0/3.0*K*dt + 1.5*M/dt;
			r = (f+f_ext+fc) - (1.5/dt)*(M*(vi-v)) + 0.5*(M*(v-v_old))/dt;
		}else{
			S =  K*dt + M/dt;
			r = (f+f_ext+fc) - (M*(vi-v))/dt;
		}
		if( D.size()>0 ) S += D;
		applyDynamicBoundaryConditions(S,r,x0,vi,dt);
		linearSolver.compute(S);

		rn = freeDOFnorm(r); //printf(" (%.2le %c) ", rn, flag?'F':' ');
		if( rn < eps && !flag ) done = true;

	}
	if( useBDF2 )
		currentAcceleration = (1.5/dt)*(vi-v) - (0.5/dt)*(v-v_old);
	else
		currentAcceleration = (vi-v)/dt;
	if( useBDF2 ){ x_old=x0; v_old=v; }

	// add contact forces to VTK output for debug
	vtkSmartPointer<vtkDoubleArray> vtkFc = vtkDoubleArray::SafeDownCast( mesh->GetPointData()->GetAbstractArray("contactForce") );
	if( vtkFc!=NULL ) for(unsigned int i=0; i<getNumberOfNodes(); ++i){
		unsigned int ixdof = getNodalDof(i,LinearFEM::X_DOF);
		vtkFc->SetTuple3(i,
			fc(ixdof  ),
			fc(ixdof+1),
			fc(ixdof+2)
		);
	}
	vtkSmartPointer<vtkIntArray> vtkFst = vtkIntArray::SafeDownCast( mesh->GetPointData()->GetAbstractArray(fieldNames[CONTACT_STATE_NAME].c_str()) );
	if( vtkFst!=NULL ) for(unsigned int i=0; i<getNumberOfNodes(); ++i){
		vtkFst->SetValue(i, ((int) contactState[i]) ); //printf("\n%% cs %d %d ", i,  contactState[i]);
	}//else{printf("notfound "); }

	v=vi; // x is already set to end-of-timestep positions
	simTime += dt;
	if( doPrintResiduals ) printf(" (%3d) %8.3lg (r%8.3lg) ", iter, rn, rn/firstResidual);
	mesh->GetPoints()->Modified(); // mark VTK structure as modified so rendering etc. can update properly
	return iter;
}


int ContactFEM::dynamicImplicitHybridPenaltyContactTimestep(double dt, double eps){
	unsigned int max_iters = LinearFEM::MAX_SOLVER_ITERS;
	unsigned int max_linesearch=20, ls_after_iter = 0;//10;//max_iters/2;// 

	if( useBDF2 && (x_old.size()==0 || v_old.size()==0) ){ // x_old or v_old for BDF2 integration need initializing ...
		x_old = x - dt*v;
		v_old = v; // assume zero acceleration for the start
	}

	double firstResidual=-1.0 ,rn, rs;
	bool done=false; int iter; bool flag=false, fsDirReset=false;
	Eigen::VectorXd vi( v.size() );
	Eigen::VectorXd x0 = x, fc( v.size() ), fs( v.size() ), fsDir( v.size() ), r;
	fc.setZero(); fs.setZero(); fsDir.setZero();

	std::vector<CONTACT_STATE> contactState; contactState.assign(getNumberOfNodes(), CONTACT_NONE); //ToDo: reduce allocated space to pre-counted number of boundary nodes
	std::vector<unsigned int> contactObstacle; contactObstacle.assign(getNumberOfNodes(), INT_MAX); //ToDo: reduce allocated space to pre-counted number of boundary nodes
	std::vector<unsigned int> contactFailCount; contactFailCount.assign(getNumberOfNodes(), 0);

	vi.setZero(); // x == x0 here
	if( useBDF2 ){
		//x = x0 + 2.0/3.0*dt*vi + 1.0/3.0*(x0-x_old);
		vi = 0.5*(x_old-x0)/dt;
	}

	// first phase: linear penalty approximate solution
	double fullEps=eps, linPenResidual=-1.0; unsigned int linPenIters=0;
	eps = sqrt(fullEps);
	{
		for(unsigned int oid=0; oid<rigidObstacles.size(); ++oid){
			contactStateByObstacle[oid].assign( getNumberOfNodes(), CONTACT_STATE::CONTACT_NONE ); // clear contact states at start of time step (set by assembleClampedLinearPenaltyForceAndStiffness)
		}

		assembleForceAndStiffness(); // overwrites f vector (does setZero() first), also writes K = -df/dx
		assembleViscousForceAndDamping( vi ); // adds to f vector - call assembleForceAndStiffness first! - also writes D = -df_visc/dv
		assembleClampedLinearPenaltyForceAndStiffness(fc, vi, 0, useBDF2?2.0/3.0*dt:dt ); //adds to f and K - call assembleForceAndStiffness first!

		if( useBDF2 ){
			S =  2.0/3.0*K*dt + 1.5*M/dt;
			r = (f+f_ext+fc) - (1.5/dt)*(M*(vi-v)) + 0.5*(M*(v-v_old))/dt;
		}else{
			S =  K*dt + M/dt;
			r = (f+f_ext+fc) - (M*(vi-v))/dt;
		}
		applyDynamicBoundaryConditions(S,r,x0,vi,dt);
		linearSolver.compute(S);

		rn = freeDOFnorm(r);  //printf(" %8.3lg ", rn);
		firstResidual = rn; rs = rn;
		if( printConvergenceDebugInfo ) printf("\n ri = [ ");
		for(iter=0; iter<max_iters && !done; ++iter){
			if( printConvergenceDebugInfo ) printf("%.4lg ", rs);

			Eigen::VectorXd dv = linearSolver.solve( r );  //printf(" (s) ");
			if( !isfinite( dv.squaredNorm() )){ printf("\n%% LINEAR SOLVER FAILED! (search direction magnitude non-finite) \n"); return -2;}

			//line-search along dv ...
			Eigen::VectorXd vs( v.size() );
			double stepsize = 1.0; int lstep=0;
			do{	
				vs = vi + stepsize*dv;

				if( useBDF2 )
					x = x0 + 2.0/3.0*dt*vs + 1.0/3.0*(x0-x_old);
				else
					x = x0+dt*vs;

				assembleForceAndStiffness( SKIP ); // overwrites f vector (does setZero() first), skip stiffness assembly
				assembleViscousForceAndDamping( vs, SKIP ); // adds to f vector - keep order of assembly fcn. calls! - skip damping matrix assembly
				flag |= assembleClampedLinearPenaltyForceAndStiffness(fc, vs, iter, useBDF2?2.0/3.0*dt:dt, SKIP );

				if( useBDF2 )
					r = (f+f_ext+fc) - (1.5/dt)*(M*(vs-v)) + 0.5*(M*(v-v_old))/dt;
				else
					r = (f+f_ext+fc) - (M*(vs-v))/dt;

				rs = freeDOFnorm(r); //printf(" [[%8.3lg]] ", rs);
				stepsize *= 0.5; ++lstep;
			}while( (iter>=ls_after_iter||rs>1e1*firstResidual) && rs > rn && lstep < max_linesearch );

			vi = vs;

			// residual is already up to date, but we need to assemble the new matrices for the next iteration
			assembleForceAndStiffness(); // overwrites f vector (does setZero() first), also writes K = -df/dx
			assembleViscousForceAndDamping( vi ); // adds to f vector - call assembleForceAndStiffness first! - also writes D = -df_visc/dv
			flag |= assembleClampedLinearPenaltyForceAndStiffness(fc, vi, iter, useBDF2?2.0/3.0*dt:dt );

			if( useBDF2 ){
				S =  2.0/3.0*K*dt + 1.5*M/dt;
			}else{
				S =  K*dt + M/dt;
			}
			if( D.size()>0 ) S += D;
			applyDynamicBoundaryConditions(S,r,x0,vi,dt);
			linearSolver.compute(S); // even in the final iteration we need to update the linear solver because we'll re-use it in the sensitivity/adjoint integration

			if( flag ) rn = firstResidual;
			if( !flag ){
				if( rs < eps || (iter>=(ls_after_iter+1) && std::abs(rn-rs) < FORCE_BALANCE_DELTA) ) done = true;
				rn = rs;
			}
			flag = false;
		}
		linPenResidual = rn;
		linPenIters = iter;
	}
	eps = fullEps;

	// assign contact states of penalty solution to local state maps
	{	//ToDO: use only one member-variable storage for contact states
		std::vector<bool> done; done.assign(getNumberOfNodes(), false); //ToDo: reduce allocated space to pre-counted number of boundary nodes
		for(unsigned int k = 0; k < getNumberOfBndryElems(); ++k) for(int j = 0; j<3; ++j){ //ToDo: keep a list of boundary nodes to iterate directly?
			unsigned int i = getBoundaryElement(k)(j);
			if(!done[i]){
				for(unsigned int oid=0; oid<rigidObstacles.size(); ++oid){
					if( contactStateByObstacle[oid][i] == CONTACT_STICK ){
						contactState[i] = contactStateByObstacle[oid][i];
						contactObstacle[i] = oid;
					}else
					if( contactStateByObstacle[oid][i] == CONTACT_SLIP && contactState[i] != CONTACT_STICK){
						contactState[i] = contactStateByObstacle[oid][i];
						contactObstacle[i] = oid;
					}
				}
			}
		}
	}

	assembleForceAndStiffness(); // overwrites f vector (does setZero() first), also writes K = -df/dx
	assembleViscousForceAndDamping( vi ); // adds to f vector - call assembleForceAndStiffness first! - also writes D = -df_visc/dv
	assembleClampedLinearPenaltyForceAndStiffness(fc, vi, linPenIters, useBDF2?2.0/3.0*dt:dt );

	if( useBDF2 )
		r = (f+f_ext) - (1.5/dt)*(M*(vi-v)) + 0.5*(M*(v-v_old))/dt;
	else
		r = (f+f_ext) - (M*(vi-v))/dt;
			
	bool allowReleaseConstraints=true;
	rn = freeDOFnorm(r); done=false;
	//if( printConvergenceDebugInfo ) printf("\n ri = [ ");
	for(iter=0; iter<max_iters && !done; ++iter){
		if( printConvergenceDebugInfo ) printf("%.4lg ", rn);

		if( useBDF2 )
			S =  2.0/3.0*K*dt + 1.5*M/dt;
		else
			S =  K*dt + M/dt;
		if( D.size()>0 ) S += D;

		Eigen::VectorXd dv( vi.size() ); dv.setZero();
		Eigen::VectorXd rr;
		// build stick constraints
		{
			std::vector<Eigen::Triplet<double> > triplets;
			stickDOFs.clear();
			std::vector<bool> constraintsDone; constraintsDone.assign(getNumberOfNodes(), false); //ToDo: reduce allocated space to pre-counted number of boundary nodes
			for(unsigned int k = 0; k < getNumberOfBndryElems(); ++k) for(int j = 0; j<3; ++j){ //ToDo: keep a list of boundary nodes to iterate directly?
				unsigned int i = getBoundaryElement(k)(j);
				if( !constraintsDone[i] ){
					if( contactState[i] == CONTACT_STICK && contactObstacle[i]<rigidObstacles.size() ){
						Eigen::Vector3d t1,t2,n;
						double g = rigidObstacles[contactObstacle[i]]->eval(n,getRestCoord(i),getDeformedCoord(i),simTime); //ToDo: could store contact normals instead ...
						// add TWO constraints in the plane orthogonal to n, i.e. (I-nn')*vi=0
						Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigN( Eigen::Matrix3d::Identity()-(n*n.transpose()) , Eigen::ComputeEigenvectors );
						bool firstTangent=true;
						for(int k=0; k<3; ++k) if( eigN.eigenvalues()(k)>0.5 ){ // one eigenvalue should be (almost) zero, the other two (close to) 1
							if( firstTangent ) t1 = eigN.eigenvectors().col(k);
							else               t2 = eigN.eigenvectors().col(k);
							firstTangent = false;
						}
						// add triplets for local coordinate transform matrix [ t1' ; t2' ; n' ] where t1,t2 are tangent vectors orthogonal to the contact normal n -- each vector in a row!
						unsigned int gjxdof = getNodalDof(i,X_DOF); // global dof index
						triplets.push_back( Eigen::Triplet<double>( gjxdof  , gjxdof  , t1(0) ));
						triplets.push_back( Eigen::Triplet<double>( gjxdof  , gjxdof+1, t1(1) ));
						triplets.push_back( Eigen::Triplet<double>( gjxdof  , gjxdof+2, t1(2) ));
						triplets.push_back( Eigen::Triplet<double>( gjxdof+1, gjxdof  , t2(0) ));
						triplets.push_back( Eigen::Triplet<double>( gjxdof+1, gjxdof+1, t2(1) ));
						triplets.push_back( Eigen::Triplet<double>( gjxdof+1, gjxdof+2, t2(2) ));
						triplets.push_back( Eigen::Triplet<double>( gjxdof+2, gjxdof  ,  n(0) ));
						triplets.push_back( Eigen::Triplet<double>( gjxdof+2, gjxdof+1,  n(1) ));
						triplets.push_back( Eigen::Triplet<double>( gjxdof+2, gjxdof+2,  n(2) ));
						stickDOFs.insert(gjxdof  );
						stickDOFs.insert(gjxdof+1);
						dv(gjxdof  ) = -(t1.dot( vi.block<3,1>(gjxdof,0) ));
						dv(gjxdof+1) = -(t2.dot( vi.block<3,1>(gjxdof,0) ));
					}
					constraintsDone[i] = true;
				}
			}
			for(unsigned int i = 0; i < getNumberOfNodes(); ++i){ // add triplets such that R=I for all unconstrained nodes
				if(!(constraintsDone[i] && contactState[i] == CONTACT_STICK && contactObstacle[i]<rigidObstacles.size()) ){
					unsigned int gjxdof = getNodalDof(i,X_DOF); // global dof index
					triplets.push_back( Eigen::Triplet<double>( gjxdof  , gjxdof  , 1.0 ));
					triplets.push_back( Eigen::Triplet<double>( gjxdof+1, gjxdof+1, 1.0 ));
					triplets.push_back( Eigen::Triplet<double>( gjxdof+2, gjxdof+2, 1.0 ));
				}
			}
			R.resize( vi.size(), vi.size() );
			if( stickDOFs.size()>0 ){
				R.setFromTriplets(triplets.begin(), triplets.end());
			}else{
				R.setIdentity();
			}
			Sr = R*S*R.transpose();
			applyDynamicBoundaryConditions(Sr,r,x0,vi,dt); // this will likely fail if a Dirichlet node also has contact conditions applied - which probably should never happen but is not explicitly prevented here
			rr = R*r;
			// apply stick constraints to Sr
			for(Eigen::Index k=0; k < Sr.outerSize(); ++k){
				for(SparseMatrixD::InnerIterator it(Sr,k); it; ++it){
					if( stickDOFs.count(it.row()) ){
						it.valueRef() = (it.row()==it.col()) ? 1.0 : 0.0;
					}
				}
			}
			// apply RHS stick constraints to rr (copy from dv)
			for(std::set<unsigned int>::iterator it=stickDOFs.begin(); it!=stickDOFs.end(); ++it){
				rr(*it) = dv(*it);
			}
		}
		linearSolver.compute(Sr);
		dv = linearSolver.solve( rr ); // velocity change in rotated coordinate system
		if( !isfinite( dv.squaredNorm() )){ printf("\n%% LINEAR SOLVER FAILED! (search direction magnitude non-finite) \n"); return -2;}

		dv = (R.transpose()*dv).eval();

		//line-search along dv ...
		Eigen::VectorXd vs( v.size() );
		double stepsize = 1.0; int lstep=0;
		do{
			vs = vi + stepsize*dv;
			if( useBDF2 ){
				x = x0 + 2.0/3.0*dt*vs + 1.0/3.0*(x0-x_old);
			}else{
				x = x0+dt*vs;
			}
			assembleForceAndStiffness( SKIP ); // overwrites f vector (does setZero() first), also writes K = -df/dx
			assembleViscousForceAndDamping( vs, SKIP ); // adds to f vector - call assembleForceAndStiffness first! - also writes D = -df_visc/dv
			//assembleNormalAndSlipPenaltyForceAndStiffness(fc, fsDir, vs, contactState, contactObstacle, useBDF2?2.0/3.0*dt:dt, SKIP ); //overwrites fc and adds to K - call assembleForceAndStiffness first!
			assembleClampedLinearPenaltyForceAndStiffness(fc, vs, iter+linPenIters, useBDF2?2.0/3.0*dt:dt, SKIP );

			if( useBDF2 ){
				r = (f+fc+f_ext) - (1.5/dt)*(M*(vi-v)) + 0.5*(M*(v-v_old))/dt;
			}else{
				r = (f+fc+f_ext) - (M*(vi-v))/dt;
			}
			if( stickDOFs.size()>0 ){
				fs = R*r; rr.setZero();
				for(std::set<unsigned int>::iterator it=stickDOFs.begin(); it!=stickDOFs.end(); ++it){
					rr(*it) = -fs(*it); // write fs on constrained DOFs
				}
				fs = R.transpose()*rr; // fs now in standard coordinate system

				rr = r+fs;
				rs = freeDOFnorm(rr);
				
			}else{
				fs.setZero();
				rs = freeDOFnorm(r);
			}
			//printf("\n%% L %4d/%2d [[%8.3lg>%8.3lg]]{%d} ", iter,lstep, rs,rn, stickDOFs.size());
			stepsize *= 0.5; ++lstep;
		}while( (iter>=ls_after_iter||rs>1e1*firstResidual) && rs > rn && lstep < max_linesearch );
		//printf(" ls%d [[%.2le]] ", lstep, rs);
		vi = vs;
		if( lstep > 10 ) fsDirReset=true; else fsDirReset=false;

		// enforce Coulomb limit
		flag = false; fsDir.setZero();
		if( 0 /*old version*/){
			std::vector<bool> contactDone; contactDone.assign(getNumberOfNodes(), false); //ToDo: reduce allocated space to pre-counted number of boundary nodes
			for(unsigned int k = 0; k < getNumberOfBndryElems(); ++k) for(int j = 0; j<3; ++j){ //ToDo: keep a list of boundary nodes to iterate directly?
				unsigned int i = getBoundaryElement(k)(j);
				if( !contactDone[i] ){
					unsigned int prevContactObstacle = contactObstacle[i];
					unsigned int contactCount=0;
					double maxFs = -1.0, fTnorm=0.0; bool bslipFlag=false;
					for(unsigned int oid=0; oid<rigidObstacles.size(); ++oid){
						Eigen::Vector3d n, fT; double fN;
						Eigen::Vector3d vt = ( Eigen::Matrix3d::Identity()-(n*n.transpose()) )*(vi.block<3,1>(getNodalDof(i,0),0));
						double g = rigidObstacles[oid]->eval(n,getRestCoord(i),getDeformedCoord(i),simTime); //ToDo: could store contact normals instead ...
						fT = fc.block<3,1>(getNodalDof(i,0),0);
						fN = n.dot(fT); // fN the normal contact force (from fc -- penalty based)
						fT -= n*fN; // now fT is the tangential part of fc (sliding force)
						fT += ( Eigen::Matrix3d::Identity()-(n*n.transpose()) )*fs.block<3,1>(getNodalDof(i,0),0);
						if( g < 0.0 ){
							++contactCount;
							if( frictionCoefficient[oid]*fN > maxFs && !bslipFlag ){
								maxFs = frictionCoefficient[oid]*fN;
								fTnorm = fT.norm();
								contactObstacle[i]=oid;
							}
							//if( iter>5 && contactState[i] == CONTACT_SLIP && prevContactObstacle==oid && frictionCoefficient[oid]>0.0 ){
							//	if( vt.norm()<(100*epsD) ){
							//		// slip too slow, transition to sticking
							//		contactState[i] = CONTACT_STICK;
							//	}
							//}
							if( contactState[i] == CONTACT_SLIP && prevContactObstacle==oid && frictionCoefficient[oid]>0.0 ){ // check for back-slip
								if( fT.squaredNorm()<DBL_EPSILON || vt.squaredNorm()<DBL_EPSILON || (  fT.dot( -vt )/vt.norm()/fT.norm()  )<0.1 ){
									contactState[i] = CONTACT_STICK; // slip to stick transition due to back slip (applying force at Coulomb limit reversed velocity)
									contactObstacle[i]=oid;
									fTnorm = 0.0; bslipFlag=true;
									++contactFailCount[i];
									flag = true; printf("x"); //printf("(x%d-%.2lg)",contactFailCount[i], fT.norm() );					
								}
							}
						}
					}
					// we now have the contact count, contactObstacle[i] is the ID where the node encounters the highest Coulomb threshold, fTnorm and maxFs are tangential force and Coulomb limit respectively
					if( contactCount > 0 && contactState[i] == CONTACT_NONE ){
						contactState[i] = CONTACT_STICK; // assume stick on first contact (do not revert to CONTACT_NONE in later iterations)
					}else
					if( iter>3 && contactState[i] == CONTACT_STICK && contactFailCount[i]<5 && (( fTnorm > maxFs && iter < (max_iters-5)) || contactCount==0) ){
						contactState[i] = CONTACT_SLIP; // stick to slip transition
						fsDir.block<3,1>(getNodalDof(i,0),0) = fs.block<3,1>(getNodalDof(i,0),0).normalized();
						fs.block<3,1>(getNodalDof(i,0),0).setZero();
						flag=true; printf("h");
						if( contactCount==0 ){
							fsDir.block<3,1>(getNodalDof(i,0),0).setZero();
							++contactFailCount[i];
						}
					}else
					if( contactState[i] == CONTACT_SLIP && fsDirReset ){
						if( fs.block<3,1>(getNodalDof(i,0),0).squaredNorm()>DBL_EPSILON ){
							fsDir.block<3,1>(getNodalDof(i,0),0) = fs.block<3,1>(getNodalDof(i,0),0).normalized();
						}else if( frictionCoefficient[contactObstacle[i]]>0.0 ){
							contactState[i] = CONTACT_STICK; flag=true; // slip to stick transition due to almost-zero tangential force
						}
					}

					if( contactCount > 0 && iter<=3 ) flag=true;

					contactDone[i]=true;
				}
			}
		}
		else if( 1&& allowReleaseConstraints && rs<sqrt(eps) /*new version: only release contraints*/){
			//allowReleaseConstraints = false; // just once
			std::vector<bool> contactDone; contactDone.assign(getNumberOfNodes(), false); //ToDo: reduce allocated space to pre-counted number of boundary nodes
			for(unsigned int k = 0; k < getNumberOfBndryElems(); ++k) for(int j = 0; j<3; ++j){ //ToDo: keep a list of boundary nodes to iterate directly?
				unsigned int i = getBoundaryElement(k)(j);
				if( !contactDone[i] && contactObstacle[i]<rigidObstacles.size() ){
					contactDone[i] = true;
					unsigned int oid = contactObstacle[i];
					double maxFs = -1.0, fTnorm=0.0;
					Eigen::Vector3d n, fContact; double fN;
					double g = rigidObstacles[oid]->eval(n,getRestCoord(i),getDeformedCoord(i),simTime); //ToDo: could store contact normals instead ...
					fContact = fc.block<3,1>(getNodalDof(i,0),0)+fs.block<3,1>(getNodalDof(i,0),0);
					fN = n.dot(fContact); // fN the normal contact force
					if( g<0.0 && frictionCoefficient[oid]>0.0 ) maxFs = frictionCoefficient[oid] * fN; 
					fTnorm = (( Eigen::Matrix3d::Identity()-(n*n.transpose()) )*fContact).norm();
					if( contactState[i] == CONTACT_STICK && (( fTnorm > maxFs ) || iter > (max_iters-10) ) ){
						contactState[i] = CONTACT_SLIP; // stick to slip transition
						flag=true; //printf("(h%u)",iter);
					}
				}
			}
		}

		if( !flag && (rs < eps /**/|| (iter>=(ls_after_iter+1) && std::abs(rn-rs) < FORCE_BALANCE_DELTA))/**/ ){
			rn=rs; //for debug output
			done = true;
		}else{
			if( useBDF2 )
				x = x0 + 2.0/3.0*dt*vi + 1.0/3.0*(x0-x_old);
			else
				x = x0+dt*vi;

			assembleForceAndStiffness(); // overwrites f vector (does setZero() first), also writes K = -df/dx
			assembleViscousForceAndDamping( vi ); // adds to f vector - call assembleForceAndStiffness first! - also writes D = -df_visc/dv
			//assembleNormalAndSlipPenaltyForceAndStiffness(fc, fsDir, vi, contactState, contactObstacle, useBDF2?2.0/3.0*dt:dt ); //overwrites fc and adds to K - call assembleForceAndStiffness first!
			assembleClampedLinearPenaltyForceAndStiffness(fc, vi, iter+linPenIters, useBDF2?2.0/3.0*dt:dt );

			if( useBDF2 )
				r = (f+fc+f_ext) - (1.5/dt)*(M*(vi-v)) + 0.5*(M*(v-v_old))/dt;
			else
				r = (f+fc+f_ext) - (M*(vi-v))/dt;

			//Eigen::VectorXd fsr = fs+r; // note that fs is not part of r for the next solve!
			//rn = freeDOFnorm(fsr); //printf(" (%.2le %c) \n", rn, flag?'F':' ');
			rn = rs;
			if( rn < eps && !flag ) done = true;
			if( flag ) rn=freeDOFnorm(r);//firstResidual; // if we've changed stick/slip classification, allow the residual to increase in the next iteration
		}
	}
	if( printConvergenceDebugInfo ) printf("%.4lg ]; residuals(end+1,1:length(ri))=ri; %%", rn);

	for(unsigned int oid=0; oid<rigidObstacles.size(); ++oid){
		contactStateByObstacle[oid].assign( getNumberOfNodes(), CONTACT_STATE::CONTACT_NONE );
	}
	for(unsigned int i=0; i<getNumberOfNodes(); ++i){
		if( contactState[i]!=CONTACT_NONE ) contactStateByObstacle[ contactObstacle[i] ][i] = contactState[i];
		//note that here even nodes that are not touching any obstacle at the end of the time step can be in CONTACT_SLIP state if they have come in contact during any earlier iteration of the solve -- maybe fix this in the future?
	}


	if( useBDF2 )
		currentAcceleration = (1.5/dt)*(vi-v) - (0.5/dt)*(v-v_old);
	else
		currentAcceleration = (vi-v)/dt;
	if( useBDF2 ){
		x_old=x0; v_old=v;
	}//else printf("BDF1");
	
	// count number of contacts in this time step for stats output
	if( doPrintResiduals ) {
		unsigned int numberOfContactNodes=0, numberOfStickNodes=0;
		for(unsigned int i=0; i<contactState.size(); ++i){
			if( contactState[i]!=CONTACT_NONE  ) ++numberOfContactNodes;
			if( contactState[i]==CONTACT_STICK ) ++numberOfStickNodes;
		}	printf(" [nc %u//%u]", numberOfContactNodes,numberOfStickNodes);
	}
	// add contact forces to VTK output for debug
	vtkSmartPointer<vtkDoubleArray> vtkFc = vtkDoubleArray::SafeDownCast( mesh->GetPointData()->GetAbstractArray(fieldNames[CONTACT_FORCE_NAME].c_str()) );
	if( vtkFc!=NULL ) for(unsigned int i=0; i<getNumberOfNodes(); ++i){
		unsigned int ixdof = getNodalDof(i,LinearFEM::X_DOF);
		vtkFc->SetTuple3(i,
			1.0*fc(ixdof  )+fs(ixdof  ),
			1.0*fc(ixdof+1)+fs(ixdof+1),
			1.0*fc(ixdof+2)+fs(ixdof+2)
		);
	}
	vtkSmartPointer<vtkIntArray> vtkFst = vtkIntArray::SafeDownCast( mesh->GetPointData()->GetAbstractArray(fieldNames[CONTACT_STATE_NAME].c_str()) );
	if( vtkFst!=NULL ) for(unsigned int i=0; i<getNumberOfNodes(); ++i){
		vtkFst->SetValue(i, ((int) contactState[i]) ); //printf("\n%% cs %d %d ", i,  contactState[i]);
	}//else{printf("notfound "); }

	v=vi; // x is already set to end-of-timestep positions
	simTime += dt;
	iter+=linPenIters;
	if( doPrintResiduals ) printf(" (%3d) %8.3lg (r%8.3lg) ", iter, rn, rn/firstResidual);
	if( doPrintResiduals ) printf("(lp[%u]%8.3lg) ", linPenIters, linPenResidual);
	mesh->GetPoints()->Modified(); // mark VTK structure as modified so rendering etc. can update properly
	if( !done && (rn > firstResidual) ) {if( doPrintResiduals ) printf(" !!! "); return -1;} // return -1 if not properly converged
	return iter;
}


//int ContactFEM::dynamicImplicitHybridPenaltyContactTimestep__OLD(double dt, double eps){
//	unsigned int max_iters = LinearFEM::MAX_SOLVER_ITERS;
//	unsigned int max_linesearch=20, ls_after_iter = 0;//10;//max_iters/2;// 
//
//	//bool useBDF2 = this->useBDF2 && (x_old.size()!=0 || v_old.size()!=0); // start with BDF1 step
//	if( this->useBDF2 && (x_old.size()==0 || v_old.size()==0) ){ // x_old or v_old for BDF2 integration need initializing ...
//		x_old = x - dt*v;
//		v_old = v; // assume zero acceleration for the start
//	}
//
//	double firstResidual=-1.0 ,rn, rs;
//	bool done=false; int iter; bool flag=false, fsDirReset=false;
//	Eigen::VectorXd vi( v.size() );
//	Eigen::VectorXd x0 = x, fc( v.size() ), fs( v.size() ), fsDir( v.size() ), r;
//	fc.setZero(); fs.setZero(); fsDir.setZero();
//
//	std::vector<CONTACT_STATE> contactState; contactState.assign(getNumberOfNodes(), CONTACT_NONE); //ToDo: reduce allocated space to pre-counted number of boundary nodes
//	std::vector<unsigned int> contactObstacle; contactObstacle.assign(getNumberOfNodes(), INT_MAX); //ToDo: reduce allocated space to pre-counted number of boundary nodes
//	std::vector<unsigned int> contactFailCount; contactFailCount.assign(getNumberOfNodes(), 0);
//
//	vi.setZero();
//	if( useBDF2 ){
//		//vi = 4.0/3.0*v - 1.0/3.0*v_old;
//		x = x0 + 2.0/3.0*dt*vi + 1.0/3.0*(x0-x_old);
//	}else{
//		//vi = v;
//		x = x0+dt*vi;
//	}
//
//	assembleForceAndStiffness(); // overwrites f vector (does setZero() first), also writes K = -df/dx
//	assembleViscousForceAndDamping( vi ); // adds to f vector - call assembleForceAndStiffness first! - also writes D = -df_visc/dv
//	assembleNormalAndSlipPenaltyForceAndStiffness(fc, fsDir, vi, contactState, contactObstacle, useBDF2?2.0/3.0*dt:dt); //overwrites fc and adds to K - call assembleForceAndStiffness first!
//
//	if( useBDF2 )
//		r = (f+f_ext) - (1.5/dt)*(M*(vi-v)) + 0.5*(M*(v-v_old))/dt;
//	else
//		r = (f+f_ext) - (M*(vi-v))/dt;
//			
//	firstResidual = freeDOFnorm(r);
//	rn = firstResidual; //printf(" %8.3lg ", rn);
//	if( printConvergenceDebugInfo ) printf("\n ri = [ ");
//	for(iter=0; iter<max_iters && !done; ++iter){
//		if( printConvergenceDebugInfo ) printf("%.4lg ", rn);
//
//		if( useBDF2 )
//			S =  2.0/3.0*K*dt + 1.5*M/dt;
//		else
//			S =  K*dt + M/dt;
//		if( D.size()>0 ) S += D;
//
//		//ToDo: re-arrange code to encapsulate local rotation and constraints
//		//      line-search such that we assemble and store the matrix corresponding to the final state
//		//      store additional constraint data for sensitivity/adjoint integration ...
//
//		// build stick constraints
//		std::vector<Eigen::Triplet<double> > triplets;
//		Eigen::VectorXd dv( vi.size() ); dv.setZero();
//		stickDOFs.clear();
//		std::vector<bool> constraintsDone; constraintsDone.assign(getNumberOfNodes(), false); //ToDo: reduce allocated space to pre-counted number of boundary nodes
//		for(unsigned int k = 0; k < getNumberOfBndryElems(); ++k) for(int j = 0; j<3; ++j){ //ToDo: keep a list of boundary nodes to iterate directly?
//			unsigned int i = getBoundaryElement(k)(j);
//			if( !constraintsDone[i] ){
//				if( contactState[i] == CONTACT_STICK && contactObstacle[i]<rigidObstacles.size() ){
//					Eigen::Vector3d t1,t2,n;
//					double g = rigidObstacles[contactObstacle[i]]->eval(n,getRestCoord(i),getDeformedCoord(i),simTime); //ToDo: could store contact normals instead ...
//					// add TWO constraints in the plane orthogonal to n, i.e. (I-nn')*vi=0
//					Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigN( Eigen::Matrix3d::Identity()-(n*n.transpose()) , Eigen::ComputeEigenvectors );
//					bool firstTangent=true;
//					for(int k=0; k<3; ++k) if( eigN.eigenvalues()(k)>0.5 ){ // one eigenvalue should be (almost) zero, the other two (close to) 1
//						if( firstTangent ) t1 = eigN.eigenvectors().col(k);
//						else               t2 = eigN.eigenvectors().col(k);
//						firstTangent = false;
//					}
//					// add triplets for local coordinate transform matrix [ t1' ; t2' ; n' ] where t1,t2 are tangent vectors orthogonal to the contact normal n -- each vector in a row!
//					unsigned int gjxdof = getNodalDof(i,X_DOF); // global dof index
//					triplets.push_back( Eigen::Triplet<double>( gjxdof  , gjxdof  , t1(0) ));
//					triplets.push_back( Eigen::Triplet<double>( gjxdof  , gjxdof+1, t1(1) ));
//					triplets.push_back( Eigen::Triplet<double>( gjxdof  , gjxdof+2, t1(2) ));
//					triplets.push_back( Eigen::Triplet<double>( gjxdof+1, gjxdof  , t2(0) ));
//					triplets.push_back( Eigen::Triplet<double>( gjxdof+1, gjxdof+1, t2(1) ));
//					triplets.push_back( Eigen::Triplet<double>( gjxdof+1, gjxdof+2, t2(2) ));
//					triplets.push_back( Eigen::Triplet<double>( gjxdof+2, gjxdof  ,  n(0) ));
//					triplets.push_back( Eigen::Triplet<double>( gjxdof+2, gjxdof+1,  n(1) ));
//					triplets.push_back( Eigen::Triplet<double>( gjxdof+2, gjxdof+2,  n(2) ));
//					stickDOFs.insert(gjxdof  );
//					stickDOFs.insert(gjxdof+1);
//					dv(gjxdof  ) = -(t1.dot( vi.block<3,1>(gjxdof,0) ));
//					dv(gjxdof+1) = -(t2.dot( vi.block<3,1>(gjxdof,0) ));
//				}
//				constraintsDone[i] = true;
//			}
//		}
//		for(unsigned int i = 0; i < getNumberOfNodes(); ++i){ // add triplets such that R=I for all unconstrained nodes
//			if(!(constraintsDone[i] && contactState[i] == CONTACT_STICK && contactObstacle[i]<rigidObstacles.size()) ){
//				unsigned int gjxdof = getNodalDof(i,X_DOF); // global dof index
//				triplets.push_back( Eigen::Triplet<double>( gjxdof  , gjxdof  , 1.0 ));
//				triplets.push_back( Eigen::Triplet<double>( gjxdof+1, gjxdof+1, 1.0 ));
//				triplets.push_back( Eigen::Triplet<double>( gjxdof+2, gjxdof+2, 1.0 ));
//			}
//		}
//		Eigen::VectorXd rr;
//		R.resize( vi.size(), vi.size() );
//		if( stickDOFs.size()>0 ){
//			R.setFromTriplets(triplets.begin(), triplets.end());
//		}else{
//			R.setIdentity();
//		}
//		Sr = R*S*R.transpose();
//		applyDynamicBoundaryConditions(Sr,r,x0,vi,dt); // this will likely fail if a Dirichlet node also has contact conditions applied - which probably should never happen but is not explicitly prevented here
//		rr = R*r;
//		// apply stick constraints to Sr
//		for(Eigen::Index k=0; k < Sr.outerSize(); ++k){
//			for(SparseMatrixD::InnerIterator it(Sr,k); it; ++it){
//				if( stickDOFs.count(it.row()) ){
//					it.valueRef() = (it.row()==it.col()) ? 1.0 : 0.0;
//				}
//			}
//		}
//		// apply RHS stick constraints to rr (copy from dv)
//		for(std::set<unsigned int>::iterator it=stickDOFs.begin(); it!=stickDOFs.end(); ++it){
//			rr(*it) = dv(*it);
//		}
//		linearSolver.compute(Sr);
//		dv = linearSolver.solve( rr ); // velocity change in rotated coordinate system
//		if( !isfinite( dv.squaredNorm() )){ printf("\n%% LINEAR SOLVER FAILED! (search direction magnitude non-finite) \n"); return -2;}
//
//		dv = (R.transpose()*dv).eval();
//		//}else{
//		//	applyDynamicBoundaryConditions(S,r,x0,vi,dt);
//		//	linearSolver.compute(S);
//		//	dv = linearSolver.solve( r );
//		//	fs.setZero();
//		//}
//
//		//line-search along dv ...
//		Eigen::VectorXd vs( v.size() );
//		double stepsize = 1.0; int lstep=0;
//		do{
//			vs = vi + stepsize*dv;
//			if( useBDF2 ){
//				x = x0 + 2.0/3.0*dt*vs + 1.0/3.0*(x0-x_old);
//			}else{
//				x = x0+dt*vs;
//			}
//			assembleForceAndStiffness( SKIP ); // overwrites f vector (does setZero() first), also writes K = -df/dx
//			assembleViscousForceAndDamping( vs, SKIP ); // adds to f vector - call assembleForceAndStiffness first! - also writes D = -df_visc/dv
//			assembleNormalAndSlipPenaltyForceAndStiffness(fc, fsDir, vs, contactState, contactObstacle, useBDF2?2.0/3.0*dt:dt, SKIP ); //overwrites fc and adds to K - call assembleForceAndStiffness first!
//			if( useBDF2 ){
//				r = (f+fc+f_ext) - (1.5/dt)*(M*(vi-v)) + 0.5*(M*(v-v_old))/dt;
//			}else{
//				r = (f+fc+f_ext) - (M*(vi-v))/dt;
//			}
//			if( stickDOFs.size()>0 ){
//				fs = R*r; rr.setZero();
//				for(std::set<unsigned int>::iterator it=stickDOFs.begin(); it!=stickDOFs.end(); ++it){
//					rr(*it) = -fs(*it); // write fs on constrained DOFs
//				}
//				fs = R.transpose()*rr; // fs now in standard coordinate system
//
//			rr = r+fs;
//			rs = freeDOFnorm(rr);
//				
//			}else{
//				fs.setZero();
//				rs = freeDOFnorm(r);
//			}
//			//printf("\n%% L %4d/%2d [[%8.3lg>%8.3lg]]{%d} ", iter,lstep, rs,rn, stickDOFs.size());
//			stepsize *= 0.5; ++lstep;
//		}while( (iter>=ls_after_iter||rs>1e1*firstResidual) && rs > rn && lstep < max_linesearch );
//		//printf(" ls%d [[%.2le]] ", lstep, rs);
//		vi = vs;
//		if( lstep > 10 ) fsDirReset=true; else fsDirReset=false;
//
//		// enforce Coulomb limit
//		flag = false; fsDir.setZero();
//		std::vector<bool> contactDone; contactDone.assign(getNumberOfNodes(), false); //ToDo: reduce allocated space to pre-counted number of boundary nodes
//		for(unsigned int k = 0; k < getNumberOfBndryElems(); ++k) for(int j = 0; j<3; ++j){ //ToDo: keep a list of boundary nodes to iterate directly?
//			unsigned int i = getBoundaryElement(k)(j);
//			if( !contactDone[i] ){
//				unsigned int prevContactObstacle = contactObstacle[i];
//				unsigned int contactCount=0;
//				double maxFs = -1.0, fTnorm=0.0; bool bslipFlag=false;
//				for(unsigned int oid=0; oid<rigidObstacles.size(); ++oid){
//					Eigen::Vector3d n, fT; double fN;
//					Eigen::Vector3d vt = ( Eigen::Matrix3d::Identity()-(n*n.transpose()) )*(vi.block<3,1>(getNodalDof(i,0),0));
//					double g = rigidObstacles[oid]->eval(n,getRestCoord(i),getDeformedCoord(i),simTime); //ToDo: could store contact normals instead ...
//					fT = fc.block<3,1>(getNodalDof(i,0),0);
//					fN = n.dot(fT); // fN the normal contact force (from fc -- penalty based)
//					fT -= n*fN; // now fT is the tangential part of fc (sliding force)
//					fT += ( Eigen::Matrix3d::Identity()-(n*n.transpose()) )*fs.block<3,1>(getNodalDof(i,0),0);
//					if( g < 0.0 ){
//						++contactCount;
//						if( frictionCoefficient[oid]*fN > maxFs && !bslipFlag ){
//							maxFs = frictionCoefficient[oid]*fN;
//							fTnorm = fT.norm();
//							contactObstacle[i]=oid;
//						}
//						//if( iter>5 && contactState[i] == CONTACT_SLIP && prevContactObstacle==oid && frictionCoefficient[oid]>0.0 ){
//						//	if( vt.norm()<(100*epsD) ){
//						//		// slip too slow, transition to sticking
//						//		contactState[i] = CONTACT_STICK;
//						//	}
//						//}
//						if( contactState[i] == CONTACT_SLIP && prevContactObstacle==oid && frictionCoefficient[oid]>0.0 ){ // check for back-slip
//							if( fT.squaredNorm()<DBL_EPSILON || vt.squaredNorm()<DBL_EPSILON || (  fT.dot( -vt )/vt.norm()/fT.norm()  )<0.1 ){
//								contactState[i] = CONTACT_STICK; // slip to stick transition due to back slip (applying force at Coulomb limit reversed velocity)
//								contactObstacle[i]=oid;
//								fTnorm = 0.0; bslipFlag=true;
//								++contactFailCount[i];
//								flag = true; //printf("(x%d-%.2lg)",contactFailCount[i], fT.norm() );					
//							}
//						}
//					}
//				}
//				// we now have the contact count, contactObstacle[i] is the ID where the node encounters the highest Coulomb threshold, fTnorm and maxFs are tangential force and Coulomb limit respectively
//				if( contactCount > 0 && contactState[i] == CONTACT_NONE ){
//					contactState[i] = CONTACT_STICK; // assume stick on first contact (do not revert to CONTACT_NONE in later iterations)
//				}else
//				if( iter>3 && contactState[i] == CONTACT_STICK && contactFailCount[i]<5 && (( fTnorm > maxFs && iter < (max_iters-5)) || contactCount==0) ){
//					contactState[i] = CONTACT_SLIP; // stick to slip transition
//					fsDir.block<3,1>(getNodalDof(i,0),0) = fs.block<3,1>(getNodalDof(i,0),0).normalized();
//					fs.block<3,1>(getNodalDof(i,0),0).setZero();
//					flag=true; //printf("h");
//					if( contactCount==0 ){
//						fsDir.block<3,1>(getNodalDof(i,0),0).setZero();
//						++contactFailCount[i];
//					}
//				}else
//				if( contactState[i] == CONTACT_SLIP && fsDirReset ){
//					if( fs.block<3,1>(getNodalDof(i,0),0).squaredNorm()>DBL_EPSILON ){
//						fsDir.block<3,1>(getNodalDof(i,0),0) = fs.block<3,1>(getNodalDof(i,0),0).normalized();
//					}else if( frictionCoefficient[contactObstacle[i]]>0.0 ){
//						contactState[i] = CONTACT_STICK; flag=true; // slip to stick transition due to almost-zero tangential force
//					}
//				}
//
//				if( contactCount > 0 && iter<=3 ) flag=true;
//
//				contactDone[i]=true;
//			}
//		}
//
//		if( !flag && rs < eps /**/|| (iter>=(ls_after_iter+1) && std::abs(rn-rs) < FORCE_BALANCE_DELTA)/**/ ){
//			rn=rs; //for debug output
//			done = true;
//		}else{
//			if( useBDF2 )
//				x = x0 + 2.0/3.0*dt*vi + 1.0/3.0*(x0-x_old);
//			else
//				x = x0+dt*vi;
//
//			assembleForceAndStiffness(); // overwrites f vector (does setZero() first), also writes K = -df/dx
//			assembleViscousForceAndDamping( vi ); // adds to f vector - call assembleForceAndStiffness first! - also writes D = -df_visc/dv
//			assembleNormalAndSlipPenaltyForceAndStiffness(fc, fsDir, vi, contactState, contactObstacle, useBDF2?2.0/3.0*dt:dt ); //overwrites fc and adds to K - call assembleForceAndStiffness first!
//
//			if( useBDF2 )
//				r = (f+fc+f_ext) - (1.5/dt)*(M*(vi-v)) + 0.5*(M*(v-v_old))/dt;
//			else
//				r = (f+fc+f_ext) - (M*(vi-v))/dt;
//
//			//Eigen::VectorXd fsr = fs+r; // note that fs is not part of r for the next solve!
//			//rn = freeDOFnorm(fsr); //printf(" (%.2le %c) \n", rn, flag?'F':' ');
//			rn = rs;
//			if( rn < eps && !flag ) done = true;
//			if( flag ) rn=firstResidual; // if we've changed stick/slip classification, allow the residual to increase in the next iteration
//		}
//	}
//	if( printConvergenceDebugInfo ) printf("%.4lg ]; residuals(end+1,1:length(ri))=ri; %%", rn);
//
//	for(unsigned int oid=0; oid<rigidObstacles.size(); ++oid){
//		contactStateByObstacle[oid].assign( getNumberOfNodes(), CONTACT_STATE::CONTACT_NONE );
//	}
//	for(unsigned int i=0; i<getNumberOfNodes(); ++i){
//		if( contactState[i]!=CONTACT_NONE ) contactStateByObstacle[ contactObstacle[i] ][i] = contactState[i];
//		//note that here even nodes that are not touching any obstacle at the end of the time step can be in CONTACT_SLIP state if they have come in contact during any earlier iteration of the solve -- maybe fix this in the future?
//	}
//
//
//	if( useBDF2 )
//		currentAcceleration = (1.5/dt)*(vi-v) - (0.5/dt)*(v-v_old);
//	else
//		currentAcceleration = (vi-v)/dt;
//	if( useBDF2 ){
//		x_old=x0; v_old=v;
//	}//else printf("BDF1");
//	
//	//// count number of contacts in this time step for stats output
//	//unsigned int numberOfContactNodes=0, numberOfStickNodes=0;
//	//for(unsigned int i=0; i<contactState.size(); ++i){
//	//	if( contactState[i]==CONTACT_SLIP || contactState[i]==CONTACT_STICK ) ++numberOfContactNodes;
//	//	if( contactState[i]==CONTACT_STICK ) ++numberOfStickNodes;
//	//}	printf(" [nc %u//%u]", numberOfContactNodes,numberOfStickNodes);
//
//	// add contact forces to VTK output for debug
//	vtkSmartPointer<vtkDoubleArray> vtkFc = vtkDoubleArray::SafeDownCast( mesh->GetPointData()->GetAbstractArray(fieldNames[CONTACT_FORCE_NAME].c_str()) );
//	if( vtkFc!=NULL ) for(unsigned int i=0; i<getNumberOfNodes(); ++i){
//		unsigned int ixdof = getNodalDof(i,LinearFEM::X_DOF);
//		vtkFc->SetTuple3(i,
//			1.0*fc(ixdof  )+fs(ixdof  ),
//			1.0*fc(ixdof+1)+fs(ixdof+1),
//			1.0*fc(ixdof+2)+fs(ixdof+2)
//		);
//	}
//	vtkSmartPointer<vtkIntArray> vtkFst = vtkIntArray::SafeDownCast( mesh->GetPointData()->GetAbstractArray(fieldNames[CONTACT_STATE_NAME].c_str()) );
//	if( vtkFst!=NULL ) for(unsigned int i=0; i<getNumberOfNodes(); ++i){
//		vtkFst->SetValue(i, ((int) contactState[i]) ); //printf("\n%% cs %d %d ", i,  contactState[i]);
//	}//else{printf("notfound "); }
//
//	v=vi; // x is already set to end-of-timestep positions
//	simTime += dt;
//	if( doPrintResiduals ) printf(" (%3d) %8.3lg (r%8.3lg) ", iter, rn, rn/firstResidual);
//	mesh->GetPoints()->Modified(); // mark VTK structure as modified so rendering etc. can update properly
//	if( !done && (rn > firstResidual) ) {if( doPrintResiduals ) printf(" !!! "); return -1;} // return -1 if not properly converged
//	return iter;
//}


int ContactFEM::dynamicImplicitQPContactTimestep(QPSolver& qpSolver, double dt, double eps){
	unsigned int max_iters = LinearFEM::MAX_SOLVER_ITERS;
	bool flag;

	if( useBDF2 && (x_old.size()==0 || v_old.size()==0) ){ // x_old or v_old for BDF2 integration need initializing ...
		x_old = x - dt*v;
		v_old = v; // assume zero acceleration for the start
	}

	double firstResidual=-1.0, rn,rp;
	bool done=false; int iter;
	Eigen::VectorXd vi( v.size() );
	Eigen::VectorXd x0 = x, r( v.size() ), fc( v.size() ), fs( v.size() );
	Eigen::VectorXd lambda;
	
	std::vector<CONTACT_STATE> contactState; contactState.assign(getNumberOfNodes(), CONTACT_NONE); //ToDo: reduce allocated space to pre-counted number of boundary nodes
	std::vector<unsigned int> contactObstacle; contactObstacle.assign(getNumberOfNodes(), INT_MAX); //ToDo: reduce allocated space to pre-counted number of boundary nodes
	//ToDo: keep contactState across time steps for better initialization? also allow for multiple obstacles

	vi.setZero(); // x == x0 here
	if( useBDF2 ){
		//x = x0 + 2.0/3.0*dt*vi + 1.0/3.0*(x0-x_old);
		vi = 0.5*(x_old-x0)/dt;
	}


	fc.setZero(); fs.setZero();
	assembleForceAndStiffness(); // overwrites f vector (does setZero() first), also writes K = -df/dx
	assembleViscousForceAndDamping( vi ); // adds to f vector - call assembleForceAndStiffness first! - also writes D = -df_visc/dv
	if( useBDF2 )
		r = (f+f_ext) - (1.5/dt)*(M*(vi-v)) + 0.5*(M*(v-v_old))/dt;
	else
		r = (f+f_ext) - (M*(vi-v))/dt;
	rn = r.norm();  //printf(" %8.3lg ", rn);
	rp = rn+1.0;

	firstResidual = rn;
	if( printConvergenceDebugInfo ) printf("\n ri = [ ");
	for(iter=0; iter<max_iters && !done; ++iter){
		if( printConvergenceDebugInfo ) printf("%.4lg ", rn);
		flag=false;

		if( useBDF2 ){
			S =  2.0/3.0*K*dt + 1.5*M/dt;
			r = -(1.5/dt)*(M*v) - 0.5*(M*(v-v_old))/dt - (2.0/3.0*dt)*(K*vi) - (f+f_ext+fs);
		}else{
			S =  K*dt + M/dt;
			r = (-1.0/dt)*(M*v) - dt*(K*vi) - (f+f_ext+fs);   //r = -(M*v) - dt*dt*(K*vi) - dt*(f+f_ext+fs);
		}
		if( D.size()>0 ){
			S += D;   //S =  M + dt*dt*K; //if( D.size()>0 ) S += dt*D;
			r -= (D*vi);
		}

		SparseMatrixD C; // constraint matrix
		Eigen::VectorXd cLower, cUpper; // lower and upper bounds, such that   cLower <= C*x <= cUpper

		unsigned int m=0; // number of variables and constraints respectively
		std::vector<bool> contactConstraintDone; contactConstraintDone.assign(getNumberOfNodes(), false); //ToDo: reduce allocated space to pre-counted number of boundary nodes
		cLower.resize( diriData.size() + 3*3*getNumberOfBndryElems()); cUpper.resize( diriData.size() + 3*3*getNumberOfBndryElems() ); //ToDo: reduce allocated space to pre-counted number of boundary nodes (near collisions?)
		std::vector<Eigen::Triplet<double> > triplets;
		// Diri BCs
		double weight = 1.0; //printf(" M diag avg per dt %.4lg \n", M.diagonal().sum() / M.rows() / dt);
		for(std::map<unsigned int,double>::iterator it=diriData.begin(); it!=diriData.end(); ++it){
			triplets.push_back( Eigen::Triplet<double>( m, it->first, weight ));
			cLower(m) = cUpper(m) = weight*(it->second - x0(it->first))/dt;
			++m;
		}
		{	// contact constraints
			Eigen::Vector3d n;
			for(unsigned int k = 0; k < getNumberOfBndryElems(); ++k) for(int j = 0; j<3; ++j){ //ToDo: keep a list of boundary nodes to iterate directly?
				unsigned int i = getBoundaryElement(k)(j);
				if( !contactConstraintDone[i] && !( diriData.count(getNodalDof(i,LinearFEM::X_DOF))>0 || diriData.count(getNodalDof(i,LinearFEM::Y_DOF))>0 || diriData.count(getNodalDof(i,LinearFEM::Z_DOF))>0 ) ){ // don't mix Diri BCs with contact constraints
					for(unsigned int oid=0; oid<rigidObstacles.size(); ++oid){
						double g = rigidObstacles[oid]->eval(n,getRestCoord(i),getDeformedCoord(i),simTime);
						double nv = n.dot( vi.block<3,1>(getNodalDof(i,0),0) );
						// add ONE constraint dt*n'*vi >= dt*nv-g --> nv-N*vi <= g/dt ie. move at most to the edge of the obstacle
						if( contactState[i]!=CONTACT_STICK || contactObstacle[i]==oid ){ // if there is a sticky contact all DOFs will be constrained - do not add additional constraints
							for(int ljdof=0; ljdof<LinearFEM::N_DOFS; ++ljdof){
								unsigned int gjdof = getNodalDof(i,ljdof); // global dof index
								triplets.push_back( Eigen::Triplet<double>( m, gjdof, dt*n(ljdof) ));
							}
							cLower(m) = nv*dt - g;
							cUpper(m) = std::numeric_limits<double>::infinity();
							++m;
						}
						if( contactState[i]==CONTACT_STICK && contactObstacle[i]==oid ){
							//tangential stick conditions ...
							// add TWO constraints in the plane orthogonal to n, i.e. (I-nn')*vi=0
							Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigN( Eigen::Matrix3d::Identity()-(n*n.transpose()) , Eigen::ComputeEigenvectors );
							for(int k=0; k<3; ++k) if( eigN.eigenvalues()(k)>0.5 ){ // one eigenvalue should be (almost) zero, the other two (close to) 1
								for(int ljdof=0; ljdof<LinearFEM::N_DOFS; ++ljdof){
									unsigned int gjdof = getNodalDof(i,ljdof); // global dof index
									triplets.push_back( Eigen::Triplet<double>( m, gjdof, dt*eigN.eigenvectors().col(k)(ljdof) ));
								}
								cLower(m) = cUpper(m) = 0.0;
								++m;
							}
						}
					}
					contactConstraintDone[i]=true; // constraint is active, state to be determined
				}
			}
		}
		cLower.conservativeResize(m); cUpper.conservativeResize(m);
		C.resize(m,vi.size()); C.setFromTriplets(triplets.begin(), triplets.end());

#ifdef QPSOLVE_USE_MATLAB
		if( matlabPtr==NULL ) matlabPtr = matlab::engine::startMATLAB(); //ToDo: encapsulate if we ever use it for more than just debugging ...
		matlab::data::ArrayFactory factory;
		//matlab::data::TypedArray<double> const argArray =  factory.createArray({ 1,4 }, { 4.0, 2.0, 6.0, 8.0 });
		//matlab::data::TypedArray<double> const results = matlabPtr->feval(u"sqrt", argArray);
		//cout << endl << "MATLAB test [";
		//for(unsigned int i=0; i<results.getNumberOfElements(); ++i){
		//	cout << results[i] << "  ";
		//}	cout << "];" << endl;

		// S
		unsigned int nnz = S.nonZeros();
		matlab::data::TypedArray<double> ml_si = factory.createArray<double>({nnz,1});
		matlab::data::TypedArray<double> ml_sj = factory.createArray<double>({nnz,1});
		matlab::data::TypedArray<double> ml_sv = factory.createArray<double>({nnz,1});
		nnz=0;
		for(Eigen::Index k=0; k < S.outerSize(); ++k){ for(SparseMatrixD::InnerIterator it(S,k); it; ++it){
				ml_si[nnz] = it.row(); ml_sj[nnz] = it.col(); ml_sv[nnz] = it.value(); ++nnz;
		}}
		matlabPtr->setVariable(u"si",ml_si); matlabPtr->setVariable(u"sj",ml_sj); matlabPtr->setVariable(u"sv",ml_sv); //matlabPtr->eval(u"disp([size(si) max(abs(si))])");
		matlabPtr->setVariable(u"n",factory.createScalar<unsigned int>(vi.size()));
		matlabPtr->eval(u"S = sparse(si+1,sj+1,sv,double(n),double(n));");
			
		// r
		matlab::data::TypedArray<double> ml_r = factory.createArray<double>({(unsigned int)r.size(),1}, r.data(), r.data()+r.size() ); matlabPtr->setVariable(u"r",ml_r);

		// C
		nnz = C.nonZeros();
		matlab::data::TypedArray<double> ml_ceqi = factory.createArray<double>({nnz,1});
		matlab::data::TypedArray<double> ml_ceqj = factory.createArray<double>({nnz,1});
		matlab::data::TypedArray<double> ml_ceqv = factory.createArray<double>({nnz,1});
		matlab::data::TypedArray<double> ml_ceqb = factory.createArray<double>({nnz,1});
		matlab::data::TypedArray<double> ml_cini = factory.createArray<double>({nnz,1});
		matlab::data::TypedArray<double> ml_cinj = factory.createArray<double>({nnz,1});
		matlab::data::TypedArray<double> ml_cinv = factory.createArray<double>({nnz,1});
		matlab::data::TypedArray<double> ml_cinb = factory.createArray<double>({nnz,1});
		unsigned int m_eq=0, m_in=0, nextEqRow=0, nextIneqRow=0;
		id_map eqRows, ineqRows;
		for(Eigen::Index k=0; k < C.outerSize(); ++k){ for(SparseMatrixD::InnerIterator it(C,k); it; ++it){
			if( cUpper(it.row()) == std::numeric_limits<double>::infinity() ){ // inequality with lower bound
				if( ineqRows.count(it.row())==0 ){ ineqRows[it.row()]=nextIneqRow++; }
				ml_cini[m_in] = ineqRows[it.row()]; ml_cinj[m_in] = it.col();
				ml_cinv[m_in] = -it.value(); ml_cinb[ineqRows[it.row()]] = -cLower(it.row());
				++m_in;
			}else{ // equality constraint
				if( eqRows.count(it.row())==0 ){ eqRows[it.row()]=nextEqRow++; }
				ml_ceqi[m_eq] = eqRows[it.row()]; ml_ceqj[m_eq] = it.col();
				ml_ceqv[m_eq] = -it.value(); ml_ceqb[eqRows[it.row()]] = -cLower(it.row());
				++m_eq;
			}
		}}
		matlabPtr->setVariable(u"cini",ml_cini); matlabPtr->setVariable(u"cinj",ml_cinj); matlabPtr->setVariable(u"cinv",ml_cinv); matlabPtr->setVariable(u"cinb",ml_cinb);
		matlabPtr->setVariable(u"m_in",factory.createScalar<unsigned int>(m_in)); matlabPtr->setVariable(u"m_inRows",factory.createScalar<unsigned int>(nextIneqRow)); 
		matlabPtr->eval(u"Cin = sparse(cini(1:m_in)+1,cinj(1:m_in)+1,cinv(1:m_in),double(m_inRows),double(n)); cinb = cinb(1:m_inRows);");
		matlabPtr->setVariable(u"ceqi",ml_ceqi); matlabPtr->setVariable(u"ceqj",ml_ceqj); matlabPtr->setVariable(u"ceqv",ml_ceqv); matlabPtr->setVariable(u"ceqb",ml_ceqb);
		matlabPtr->setVariable(u"m_eq",factory.createScalar<unsigned int>(m_eq)); matlabPtr->setVariable(u"m_eqRows",factory.createScalar<unsigned int>(nextEqRow));
		matlabPtr->eval(u"Ceq = sparse(ceqi(1:m_eq)+1,ceqj(1:m_eq)+1,ceqv(1:m_eq),double(m_eqRows),double(n)); ceqb = ceqb(1:m_eqRows);");



		matlabPtr->eval(u"oopt = optimoptions(@quadprog,'Display','off','OptimalityTolerance',1e-14,'ConstraintTolerance', 1e-14, 'StepTolerance', 1e-14);");
		matlabPtr->eval(u"[vi,~,~,~,qpLa] = quadprog( 0.5*(S+S'), r, Cin,cinb,Ceq,ceqb,[],[],[],oopt);");
		//matlabPtr->eval(u"disp( [ size(Cin) size(cinb) size(qpLa.ineqlin) 0 size(Ceq) size(ceqb) size(qpLa.eqlin) ]  );");
		//matlabPtr->eval(u"disp(  norm( S*vi+r +Cin'*qpLa.ineqlin+Ceq'*qpLa.eqlin )  );");
		matlabPtr->eval(u"fc = -Cin'*qpLa.ineqlin  -Ceq'*qpLa.eqlin;");
		matlab::data::TypedArray<double> ml_vi = matlabPtr->getVariable(u"vi");
		matlab::data::TypedArray<double> ml_fc = matlabPtr->getVariable(u"fc");
		for(unsigned int i=0; i<vi.size(); ++i) vi[i]=ml_vi[i];
		for(unsigned int i=0; i<fc.size(); ++i) fc[i]=ml_fc[i];
#else
		if( qpSolver.solve(vi, lambda, eps, m, S, r, C, cLower, cUpper) !=0 ){
			vi.setZero(); x=x0; printf("\n%% Error: the QP solver failed.\n"); return -1;
		}

		// compute contact forces (includes Diri BCs - skip?)
		if( lambda.size()==0 ) fc.setZero();
		else fc = C.transpose()*lambda;

		//printf(" [%.3le] ", (S*vi+r-fc).norm() );
#endif

		// enforce Coulomb friction limits
		if( iter < (max_iters - 5) ){
			flag = classifyStickSlipNodes(contactState, contactObstacle, fs, contactConstraintDone, vi, fc, iter, eps);
		}else flag = false;

		if( useBDF2 )
			x = x0 + 2.0/3.0*dt*vi + 1.0/3.0*(x0-x_old);
		else
			x = x0+dt*vi;

		assembleForceAndStiffness(); // overwrites f vector (does setZero() first), also writes K = -df/dx
		assembleViscousForceAndDamping( vi ); // adds to f vector - call assembleForceAndStiffness first! - also writes D = -df_visc/dv
		if( useBDF2 )
			r = (f+f_ext+fc+fs) - (1.5/dt)*(M*(vi-v)) + 0.5*(M*(v-v_old))/dt;
		else
			r = (f+f_ext+fc+fs) - (M*(vi-v))/dt;   //r = dt*(f+f_ext+fc+fs) - M*(vi-v);
		rn = r.norm(); //printf(" %8.3le (%c) ", rn, flag?'F':' ');

		if( rn < eps /**/|| std::abs(rn-rp) < FORCE_BALANCE_DELTA/**/ ){ done = true; }
		rp = rn;
		if( flag ) {rp = firstResidual+1.0; done = false; }
	}
	if( printConvergenceDebugInfo ) printf("%.4lg ]; residuals(end+1,1:length(ri))=ri; %%", rn);

	//ToDo: for sensitivity analysis prepare matrices for QP derivatives ... probably need a separate QP sensitivity analysis class (adjoint form unknown)


	//cout << endl << "fc" << endl << fc << endl << "fs" << endl << fs << endl;
	if( useBDF2 )
		currentAcceleration = (1.5/dt)*(vi-v) - (0.5/dt)*(v-v_old);
	else
		currentAcceleration = (vi-v)/dt;
	if( useBDF2 ){ x_old=x0; v_old=v; }
	v=vi; // x is already set to end-of-timestep positions
	simTime += dt;
	if( doPrintResiduals ) printf(" (%3d) %8.3le (r%8.3le) ", iter, rn, rn/firstResidual);
	mesh->GetPoints()->Modified();

	// add contact forces to VTK output for debug
	vtkSmartPointer<vtkDoubleArray> vtkFc = vtkDoubleArray::SafeDownCast( mesh->GetPointData()->GetAbstractArray(fieldNames[CONTACT_FORCE_NAME].c_str()) );
	if( vtkFc!=NULL ) for(unsigned int i=0; i<getNumberOfNodes(); ++i){
		unsigned int ixdof = getNodalDof(i,LinearFEM::X_DOF);
		vtkFc->SetTuple3(i,
			1.0*fc(ixdof  ) + 1.0*fs(ixdof  ),
			1.0*fc(ixdof+1) + 1.0*fs(ixdof+1),
			1.0*fc(ixdof+2) + 1.0*fs(ixdof+2)
		);
	}//else{printf("notfound "); }
	vtkSmartPointer<vtkIntArray> vtkFst = vtkIntArray::SafeDownCast( mesh->GetPointData()->GetAbstractArray(fieldNames[CONTACT_STATE_NAME].c_str()) );
	if( vtkFst!=NULL ) for(unsigned int i=0; i<getNumberOfNodes(); ++i){
		vtkFst->SetValue(i, ((double) contactState[i]) ); //printf("\n%% cs %d %d ", i,  contactState[i]);
	}//else{printf("notfound "); }

	return iter;
}


bool ContactFEM::classifyStickSlipNodes(std::vector<ContactFEM::CONTACT_STATE>& contactState, std::vector<unsigned int>& contactObstacle, Eigen::VectorXd& fs, const std::vector<bool>& contactConstraintDone, const Eigen::VectorXd& vi, const Eigen::VectorXd& fc, unsigned int iter, double eps){
	bool flag=false;
	for(unsigned int k = 0; k < getNumberOfBndryElems(); ++k) for(int j = 0; j<3; ++j){ //ToDo: keep a list of boundary nodes to iterate directly?
		unsigned int i = getBoundaryElement(k)(j);
		unsigned int ixdof = getNodalDof(i,0);
		if( contactConstraintDone[i] ){ // constraint is set, evaluate contact force and determine stick or slip state
			for(unsigned int oid=0; oid<rigidObstacles.size(); ++oid){
				Eigen::Vector3d n, fT; double fN;
				double g = rigidObstacles[oid]->eval(n,getRestCoord(i),getDeformedCoord(i),simTime); //ToDo: could store contact normals instead ...
				fT = fc.block<3,1>(ixdof,0);
				fN = n.dot(fT);
				fT -= n*fN; // now fT is the tangential part of fc and fN the normal part
				Eigen::Block<Eigen::VectorXd,3,1> fS = fs.block<3,1>(ixdof,0);
				if( fN < eps ){
					fN=0.0;
					if( contactState[i] == CONTACT_STICK  && contactObstacle[i]==oid ){ contactState[i] = CONTACT_SLIP; fS.setZero(); } // make sure we don't apply stick conditions if the node is about to lift off (no normal force)
				}else{
					if( contactState[i]==CONTACT_NONE ){
						contactState[i] =CONTACT_STICK;
						contactObstacle[i]=oid;
						fS.setZero();
					}
					if( iter<=3 ) flag=true; // if we have a stick contact in the first iterations, do at least enough iters to check for slipping correctly
				}
				
				if( iter>3 && contactState[i]!=CONTACT_NONE && contactObstacle[i]==oid && (fT.norm() > (frictionCoefficient[oid]*fN) || contactState[i]==CONTACT_SLIP )){ // first iter -> detect contact; then -> solve stick forces; later -> allow slipping
					if( contactState[i]==CONTACT_STICK ){
						contactState[i] =CONTACT_SLIP;
						fS = fT*(frictionCoefficient[oid]*fN/fT.norm()); // set direction of fS from fT if we change from stick to slip
						flag=true; //printf("*");
					}else if( fS.dot( vi.block<3,1>(ixdof,0) ) > DBL_EPSILON ){
						contactState[i]=CONTACT_STICK;
						flag=true; //printf("x");
						fS.setZero();
					}else if( fS.squaredNorm() < DBL_EPSILON ){ // already slipping with low force magnitude, set direction from velocity instead
						if( vi.block<3,1>(ixdof,0).squaredNorm() > DBL_EPSILON ){
							fS = -(frictionCoefficient[oid]*fN)*vi.block<3,1>(ixdof,0).normalized();
						}else{ fS.setZero(); }
					}else{
						fS *= (frictionCoefficient[oid]*fN/fS.eval().norm());
					}
				}
			}
		}
	}
	return flag;
}


void ContactFEM::assembleTanhPenaltyForceAndStiffness(const Eigen::VectorXd& vi, const double dt, UPDATE_MODE mode){
	if( mode==REBUILD ){ printf("\n%% WARNING: assembleTanhPenaltyForceAndStiffness is not intended to be called with mode==REBUILD \n"); mode=UPDATE; }
	if( K.size()==0 ){ printf("\n%% WARNING: standard stiffness matrix should be assembled before calling assembleTanhPenaltyForceAndStiffness \n"); }
	if( D.size()==0 ){ D=K; // set D to zero with sparsity pattern of K
		for(unsigned int k=0; k<D.data().size(); ++k)  D.data().value(k)=0.0;
	}
	std::vector<bool> done; done.assign(getNumberOfNodes(), false); //ToDo: reduce allocated space to pre-counted number of boundary nodes
	//ToDo: add penalty forces per bndry elem so that we take 1-ring area into account (penalize densely meshed areas less)
					
	Eigen::Vector3d n,t, f_n, f_t; Eigen::Matrix3d N,T,Kn,Kt,Dt;
	f_n.setZero(); f_t.setZero(); Kn.setZero(); Kt.setZero(); Dt.setZero();
	for(unsigned int k = 0; k < getNumberOfBndryElems(); ++k) for(int j = 0; j<3; ++j){ //ToDo: keep a list of boundary nodes to iterate directly?
		unsigned int i = getBoundaryElement(k)(j);
		if(!done[i]){
			for(unsigned int oid=0; oid<rigidObstacles.size(); ++oid){
				double g = rigidObstacles[oid]->eval(n,getRestCoord(i),getDeformedCoord(i),simTime);
				
				nodalTanhPenaltyForceAndStiffness(f_n,Kn, f_t,Kt,Dt, normalPenaltyFactor,tangentialPenaltyFactor,frictionCoefficient[oid], g,n, vi.block<3,1>(getNodalDof(i,0),0),dt, mode);

				for(int lidof=0; lidof<LinearFEM::N_DOFS; ++lidof){
					unsigned int gidof = getNodalDof(i,lidof); // global dof index
					f(gidof) += (f_n(lidof) + f_t(lidof));
					if( mode==UPDATE ) for(int ljdof=0; ljdof<LinearFEM::N_DOFS; ++ljdof){
						unsigned int gjdof = getNodalDof(i,ljdof); // global dof index
						K.coeffRef(gjdof,gidof) += (Kn(ljdof,lidof)+Kt(ljdof,lidof));
						D.coeffRef(gjdof,gidof) += (Dt(ljdof,lidof)*dt);
					}
				}
			}

			done[i]=true;
		}
	}
}


bool ContactFEM::assembleClampedLinearPenaltyForceAndStiffness(Eigen::VectorXd& fc, const Eigen::VectorXd& vi, const unsigned int iter, const double dt, UPDATE_MODE mode){
	bool flag=false;
	fc.setZero();
	if( mode==REBUILD ){ printf("\n%% WARNING: assembleClampedLinearPenaltyForceAndStiffness is not intended to be called with mode==REBUILD \n"); mode=UPDATE; }
	if( K.size()==0 ){ printf("\n%% WARNING: standard stiffness matrix should be assembled before calling assembleClampedLinearPenaltyForceAndStiffness \n"); }
	if( D.size()==0 ){ D=K; // set D to zero with sparsity pattern of K
		for(unsigned int k=0; k<D.data().size(); ++k)  D.data().value(k)=0.0;
	}
	std::vector<bool> done; done.assign(getNumberOfNodes(), false); //ToDo: reduce allocated space to pre-counted number of boundary nodes
	//ToDo: add penalty forces per bndry elem so that we take 1-ring area into account (penalize densely meshed areas less)
					
	Eigen::Vector3d n, f_n, f_t; Eigen::Matrix3d N,T,Kn,Kt,Dt;
	f_n.setZero(); f_t.setZero(); Kn.setZero(); Kt.setZero(); Dt.setZero();
	for(unsigned int k = 0; k < getNumberOfBndryElems(); ++k) for(int j = 0; j<3; ++j){ //ToDo: keep a list of boundary nodes to iterate directly?
		unsigned int i = getBoundaryElement(k)(j);
		if(!done[i]){
			for(unsigned int oid=0; oid<rigidObstacles.size(); ++oid){
				double g = rigidObstacles[oid]->eval(n,getRestCoord(i),getDeformedCoord(i),simTime);

				CONTACT_STATE state = nodalClampedLinearPenaltyForceAndStiffness(f_n,Kn, f_t,Kt,Dt, normalPenaltyFactor,tangentialPenaltyFactor,
					frictionCoefficient[oid] ,   g,n,(/**/iter<=3/*/0/**/)?true:false ,vi.segment<3>(getNodalDof(i,0)), dt, mode);
				if( g<0.0 && iter<=3) flag=true;

				contactStateByObstacle[oid][i] = state;

				for(int lidof=0; lidof<LinearFEM::N_DOFS; ++lidof){
					unsigned int gidof = getNodalDof(i,lidof); // global dof index
					fc(gidof) += (f_n(lidof) + f_t(lidof));
					if( mode==UPDATE ) for(int ljdof=0; ljdof<LinearFEM::N_DOFS; ++ljdof){
						unsigned int gjdof = getNodalDof(i,ljdof); // global dof index
						K.coeffRef(gjdof,gidof) += (Kn(ljdof,lidof)+Kt(ljdof,lidof));
						D.coeffRef(gjdof,gidof) += (Dt(ljdof,lidof)*dt);
					}
				}
			}

			done[i]=true;
		}
	}
	return flag;
}


void ContactFEM::assembleLinearPenaltyForceAndStiffness(Eigen::VectorXd& fc, const Eigen::VectorXd& vi, const std::vector<CONTACT_STATE>& contactState, const std::vector<unsigned int>& contactObstacle, const double dt, UPDATE_MODE mode){
	fc.setZero();
	if( mode==REBUILD ){ printf("\n%% WARNING: assembleLinearPenaltyForceAndStiffness is not intended to be called with mode==REBUILD \n"); mode=UPDATE; }
	if( K.size()==0 ){ printf("\n%% WARNING: standard stiffness matrix should be assembled before calling assembleLinearPenaltyForceAndStiffness \n"); }
	if( D.size()==0 ){ D=K; // set D to zero with sparsity pattern of K
		for(unsigned int k=0; k<D.data().size(); ++k)  D.data().value(k)=0.0;
	}
	std::vector<bool> done; done.assign(getNumberOfNodes(), false); //ToDo: reduce allocated space to pre-counted number of boundary nodes
	//ToDo: add penalty forces per bndry elem so that we take 1-ring area into account (penalize densely meshed areas less)
					
	Eigen::Vector3d n,t, f_n, f_t; Eigen::Matrix3d N,T,Kn,Kt,Dt;
	f_n.setZero(); f_t.setZero(); Kn.setZero(); Kt.setZero(); Dt.setZero();
	for(unsigned int k = 0; k < getNumberOfBndryElems(); ++k) for(int j = 0; j<3; ++j){ //ToDo: keep a list of boundary nodes to iterate directly?
		unsigned int i = getBoundaryElement(k)(j);
		if(!done[i]){
			for(unsigned int oid=0; oid<rigidObstacles.size(); ++oid){
				double g = rigidObstacles[oid]->eval(n,getRestCoord(i),getDeformedCoord(i),simTime);

				nodalLinearPenaltyForceAndStiffness(f_n,Kn, f_t,Kt,Dt, normalPenaltyFactor,tangentialPenaltyFactor,frictionCoefficient[oid], g,n, vi.block<3,1>(getNodalDof(i,0),0),
					contactObstacle[i]==oid ? contactState[i] : CONTACT_NONE, dt, mode); // only add tangential stick or slip penalty for the designated contactObstacle

				//if(i==10 && contactState[i]==CONTACT_SLIP){ printf("\n%% node %d slip force is %.2le  ", i,f_t.norm());  cout << " f=( " << f_t.transpose() << " ) ";}

				for(int lidof=0; lidof<LinearFEM::N_DOFS; ++lidof){
					unsigned int gidof = getNodalDof(i,lidof); // global dof index
					fc(gidof) += (f_n(lidof) + f_t(lidof));
					if( mode==UPDATE ) for(int ljdof=0; ljdof<LinearFEM::N_DOFS; ++ljdof){
						unsigned int gjdof = getNodalDof(i,ljdof); // global dof index
						K.coeffRef(gjdof,gidof) += (Kn(ljdof,lidof)+Kt(ljdof,lidof));
						//K.coeffRef(gjdof,gidof) += (Dt(ljdof,lidof));
						D.coeffRef(gjdof,gidof) += (Dt(ljdof,lidof)*dt);
					}
				}
			}

			done[i]=true;
		}
	}
}


void ContactFEM::assembleNormalAndSlipPenaltyForceAndStiffness(Eigen::VectorXd& fc, const Eigen::VectorXd& fsDir, const Eigen::VectorXd& vi, const std::vector<CONTACT_STATE>& contactState, const std::vector<unsigned int>& contactObstacle, const double dt, UPDATE_MODE mode){
	fc.setZero();
	if( mode==REBUILD ){ printf("\n%% WARNING: assembleLinearPenaltyForceAndStiffness is not intended to be called with mode==REBUILD \n"); mode=UPDATE; }
	if( K.size()==0 ){ printf("\n%% WARNING: standard stiffness matrix should be assembled before calling assembleLinearPenaltyForceAndStiffness \n"); }
	if( D.size()==0 ){ D=K; // set D to zero with sparsity pattern of K
		for(unsigned int k=0; k<D.data().size(); ++k)  D.data().value(k)=0.0;
	}
	std::vector<bool> done; done.assign(getNumberOfNodes(), false); //ToDo: reduce allocated space to pre-counted number of boundary nodes
	//ToDo: add penalty forces per bndry elem so that we take 1-ring area into account (penalize densely meshed areas less)
					
	Eigen::Vector3d n,t, f_n, f_t; Eigen::Matrix3d N,T,Kn,Kt,Dt;
	f_n.setZero(); f_t.setZero(); Kn.setZero(); Kt.setZero(); Dt.setZero();
	for(unsigned int k = 0; k < getNumberOfBndryElems(); ++k) for(int j = 0; j<3; ++j){ //ToDo: keep a list of boundary nodes to iterate directly?
		unsigned int i = getBoundaryElement(k)(j);
		if(!done[i]){
			for(unsigned int oid=0; oid<rigidObstacles.size(); ++oid){
				double g = rigidObstacles[oid]->eval(n,getRestCoord(i),getDeformedCoord(i),simTime);

				nodalNormalAndSlipPenaltyForceAndStiffness(f_n,Kn, f_t,Kt,Dt, normalPenaltyFactor,frictionCoefficient[oid], g,n, vi.block<3,1>(getNodalDof(i,0),0),
					fsDir.block<3,1>(getNodalDof(i,0),0), contactObstacle[i]==oid ? contactState[i] : CONTACT_NONE, dt, mode); // only add tangential stick or slip penalty for the designated contactObstacle



				for(int lidof=0; lidof<LinearFEM::N_DOFS; ++lidof){
					unsigned int gidof = getNodalDof(i,lidof); // global dof index
					fc(gidof) += (f_n(lidof) + f_t(lidof));
					if( mode==UPDATE ) for(int ljdof=0; ljdof<LinearFEM::N_DOFS; ++ljdof){
						unsigned int gjdof = getNodalDof(i,ljdof); // global dof index
						K.coeffRef(gjdof,gidof) += (Kn(ljdof,lidof)+Kt(ljdof,lidof));
						D.coeffRef(gjdof,gidof) += (Dt(ljdof,lidof)*dt);
					}
				}
			}
			done[i]=true;
		}
	}
}


void ContactFEM::nodalNormalAndSlipPenaltyForceAndStiffness(
	Eigen::Vector3d& f_n, Eigen::Matrix3d& Kn, Eigen::Vector3d& f_t, Eigen::Matrix3d& Kt, Eigen::Matrix3d& Dt,
	const double pF, const double cF, const double g, const Eigen::Vector3d& n,
	const Eigen::Vector3d& v, const Eigen::Vector3d& d, CONTACT_STATE contactState, const double dt, UPDATE_MODE mode
){
	//const double epsD=1e-6; // regularization of direction normalization (could need a bit more here than in the clamp penalty version, as the stick case is stricter)
	f_n.setZero(); Kn.setZero();  f_t.setZero(); Kt.setZero(); Dt.setZero();
	if( g<0.0 ){
		Eigen::Matrix3d N = n*n.transpose();
		Eigen::Matrix3d T = Eigen::Matrix3d::Identity() - N;

		f_n = (-pF*g)*n; // normal penalty force
		if( mode==UPDATE ) Kn =  pF*N;

		if( contactState == CONTACT_SLIP && cF >0.0 ){ // slip force
			double fMax = (-cF*pF*g); // -pF*g >0 is the normal force magnitude -- (-cF*pF*g) is the Coulomb limit			
			if( d.squaredNorm()>0.2 ){ // prescribed direction of tangential force (applied if stick condition has just been released and consequently T*v==0 for current v
				f_t = fMax*d;
				if( mode==UPDATE ){
					Kt = cF*(d*(n.transpose()*Kn));
				}
			}else{
				Eigen::Vector3d t = dt*T*v;
				double tu = t.norm();
				t /= (tu+epsD); // now t is the direction of tangential motion
				f_t = -fMax*t;
				if( mode==UPDATE ){
					// ... Kt = -df_t/dx = df_t/dfMax dfMax/dx + df_t/dt dt/dx ...
					{
						const double
							&PT_1_1=T(0,0), &PT_1_2=T(0,1), &PT_1_3=T(0,2),
							&PT_2_1=T(1,0), &PT_2_2=T(1,1), &PT_2_3=T(1,2),
							&PT_3_1=T(2,0), &PT_3_2=T(2,1), &PT_3_3=T(2,2),
							vi1=dt*v(0),vi2=dt*v(1),vi3=dt*v(2);
						const double t5 = PT_1_1*vi1;
						const double t6 = PT_1_2*vi2;
						const double t7 = PT_1_3*vi3;
						const double t2 = t5+t6+t7;
						const double t9 = PT_2_1*vi1;
						const double t10 = PT_2_2*vi2;
						const double t11 = PT_2_3*vi3;
						const double t3 = t9+t10+t11;
						const double t13 = PT_3_1*vi1;
						const double t14 = PT_3_2*vi2;
						const double t15 = PT_3_3*vi3;
						const double t4 = t13+t14+t15;
						const double t8 = t2*t2;
						const double t12 = t3*t3;
						const double t16 = t4*t4;
						const double t17 = t8+t12+t16;
						const double t18 = sqrt(t17);
						const double t19 = epsD+t18;
						const double t20 = 1.0/t19;
						const double t21 = 1.0/(t19*t19);
						const double t22 = 1.0/sqrt(t17);
						const double t23 = PT_1_1*t2*2.0;
						const double t24 = PT_2_1*t3*2.0;
						const double t25 = PT_3_1*t4*2.0;
						const double t26 = t23+t24+t25;
						const double t27 = PT_1_2*t2*2.0;
						const double t28 = PT_2_2*t3*2.0;
						const double t29 = PT_3_2*t4*2.0;
						const double t30 = t27+t28+t29;
						const double t31 = PT_1_3*t2*2.0;
						const double t32 = PT_2_3*t3*2.0;
						const double t33 = PT_3_3*t4*2.0;
						const double t34 = t31+t32+t33;
						Dt(0,0) = PT_1_1*t20-t2*t21*t22*t26*(1.0/2.0);
						Dt(0,1) = PT_1_2*t20-t2*t21*t22*t30*(1.0/2.0);
						Dt(0,2) = PT_1_3*t20-t2*t21*t22*t34*(1.0/2.0);
						Dt(1,0) = PT_2_1*t20-t3*t21*t22*t26*(1.0/2.0);
						Dt(1,1) = PT_2_2*t20-t3*t21*t22*t30*(1.0/2.0);
						Dt(1,2) = PT_2_3*t20-t3*t21*t22*t34*(1.0/2.0);
						Dt(2,0) = PT_3_1*t20-t4*t21*t22*t26*(1.0/2.0);
						Dt(2,1) = PT_3_2*t20-t4*t21*t22*t30*(1.0/2.0);
						Dt(2,2) = PT_3_3*t20-t4*t21*t22*t34*(1.0/2.0);
					}
					Dt *= fMax; // df_t/dt dt/dx
					Kt  = -cF*(t*(n.transpose()*Kn)); // df_t/dfMax dfMax/dx
				}
			}
		}
	}
}


void ContactFEM::nodalTanhPenaltyForceAndStiffness(
	Eigen::Vector3d& f_n, Eigen::Matrix3d& Kn, Eigen::Vector3d& f_t, Eigen::Matrix3d& Kt, Eigen::Matrix3d& Dt,
	const double pF, const double pFt, const double cF, const double g, const Eigen::Vector3d& n,
	const Eigen::Vector3d& v, const double dt, UPDATE_MODE mode
){
	f_n.setZero(); Kn.setZero();  f_t.setZero(); Kt.setZero(); Dt.setZero();
	if( g<0.0 ){
		Eigen::Matrix3d N = n*n.transpose();
		Eigen::Matrix3d T = Eigen::Matrix3d::Identity() - N;


		f_n = (-pF*g)*n; // normal penalty force
		if( mode==UPDATE ) Kn =  pF*N;

		const double pf=pFt*dt;
		Eigen::Vector3d t = T*v;
		double fth, tu = t.norm(), fMax = (-cF*pF*g); // -pF*g >0 is the normal force magnitude -- (-cF*pF*g) is the Coulomb limit
		if( tu > DBL_EPSILON && fMax > DBL_EPSILON ){
			t /= tu; //n.normalize(); // now t is the direction of tangential motion

			fth = std::tanh(pf/fMax*tu);
			f_t = (-fMax*fth)*t; 
			if( mode==UPDATE ){
				{	// first part: derivative of f_t wrt. v, where both tu and t are functions of v (assuming fMax const.)
					const double &nx=n[0],&ny=n[1],&nz=n[2], &vx=v[0], &vy=v[1], &vz=v[2];
					double t5 = nx*nx;
					double t6 = t5*vx;
					double t7 = nx*ny*vy;
					double t8 = nx*nz*vz;
					double t2 = t6+t7+t8-vx;
					double t10 = ny*ny;
					double t11 = t10*vy;
					double t12 = nx*ny*vx;
					double t13 = ny*nz*vz;
					double t3 = t11+t12+t13-vy;
					double t15 = nz*nz;
					double t16 = t15*vz;
					double t17 = nx*nz*vx;
					double t18 = ny*nz*vy;
					double t4 = t16+t17+t18-vz;
					double t9 = t2*t2;
					double t14 = t3*t3;
					double t19 = t4*t4;
					double t20 = t9+t14+t19;
					double t21 = 1.0/fMax;
					double t22 = sqrt(t20);
					double t23 = pf*t21*t22;
					double t24 = tanh(t23);
					double t25 = t5-1.0;
					double t26 = t2*t25*2.0;
					double t27 = nx*ny*t3*2.0;
					double t28 = nx*nz*t4*2.0;
					double t29 = t26+t27+t28;
					double t30 = 1.0/dt;
					double t31 = 1.0/sqrt(t20);
					double t32 = 1.0/sqrt(t20*t20*t20); //1.0/pow(t20,3.0/2.0);
					double t33 = t24*t24;
					double t34 = t33-1.0;
					double t35 = t10-1.0;
					double t36 = t3*t35*2.0;
					double t37 = nx*ny*t2*2.0;
					double t38 = ny*nz*t4*2.0;
					double t39 = t36+t37+t38;
					double t40 = 1.0/t20;
					double t41 = t15-1.0;
					double t42 = t4*t41*2.0;
					double t43 = nx*nz*t2*2.0;
					double t44 = ny*nz*t3*2.0;
					double t45 = t42+t43+t44;
					Dt(0,0) = t30*(-fMax*t24*t25*t31+fMax*t2*t24*t29*t32*(1.0/2.0)+pf*t2*t29*t34*t40*(1.0/2.0));
					Dt(0,1) = t30*(-fMax*nx*ny*t24*t31+fMax*t2*t24*t32*t39*(1.0/2.0)+pf*t2*t34*t39*t40*(1.0/2.0));
					Dt(0,2) = t30*(-fMax*nx*nz*t24*t31+fMax*t2*t24*t32*t45*(1.0/2.0)+pf*t2*t34*t40*t45*(1.0/2.0));
					Dt(1,0) = t30*(-fMax*nx*ny*t24*t31+fMax*t3*t24*t29*t32*(1.0/2.0)+pf*t3*t29*t34*t40*(1.0/2.0));
					Dt(1,1) = t30*(-fMax*t24*t31*t35+fMax*t3*t24*t32*t39*(1.0/2.0)+pf*t3*t34*t39*t40*(1.0/2.0));
					Dt(1,2) = t30*(-fMax*ny*nz*t24*t31+fMax*t3*t24*t32*t45*(1.0/2.0)+pf*t3*t34*t40*t45*(1.0/2.0));
					Dt(2,0) = t30*(-fMax*nx*nz*t24*t31+fMax*t4*t24*t29*t32*(1.0/2.0)+pf*t4*t29*t34*t40*(1.0/2.0));
					Dt(2,1) = t30*(-fMax*ny*nz*t24*t31+fMax*t4*t24*t32*t39*(1.0/2.0)+pf*t4*t34*t39*t40*(1.0/2.0));
					Dt(2,2) = t30*(-fMax*t24*t31*t41+fMax*t4*t24*t32*t45*(1.0/2.0)+pf*t4*t34*t40*t45*(1.0/2.0));
				}
				// second part: derivative of f_t wrt. x, where fMax is a function of g, which is a function of x (assuming t const.)
				Kt = (t*n.transpose()*Kn)*( pf/pF*(fth*fth-1.0)*tu/g - cF*fth );
			}
		}
	}
}


ContactFEM::CONTACT_STATE ContactFEM::nodalClampedLinearPenaltyForceAndStiffness(
	Eigen::Vector3d& f_n, Eigen::Matrix3d& Kn, Eigen::Vector3d& f_t, Eigen::Matrix3d& Kt, Eigen::Matrix3d& Dt,
	const double pF, const double pFt, const double cF, const double g, const Eigen::Vector3d& n, bool forceStick,
	const Eigen::Vector3d& v, const double dt, UPDATE_MODE mode
){
	//const double epsD=1e-8; // regularization of direction normalization
	f_n.setZero(); Kn.setZero();  f_t.setZero(); Kt.setZero(); Dt.setZero();
	CONTACT_STATE state = CONTACT_STATE::CONTACT_NONE;
	if( g<0.0 ){
		Eigen::Matrix3d N = n*n.transpose();
		Eigen::Matrix3d T = Eigen::Matrix3d::Identity() - N;
		double fn = (-pF*g);

		f_n = fn*n; // normal penalty force
		if( mode==UPDATE ) Kn = pF*N;

		Eigen::Vector3d t = dt*T*v;
		double tu = t.norm(), fMax = cF*fn;
		if( cF>0.0 ||forceStick){ // disable any friction if cF is zero or negative
			if(!forceStick && pFt*tu > fMax ){ // slip force
				state = CONTACT_STATE::CONTACT_SLIP;

				t /= (tu+epsD); // now t is the direction of tangential motion
				f_t = -fMax*t;
				if( mode==UPDATE ){
					// ... Kt = -df_t/dx = df_t/dfMax dfMax/dx + df_t/dt dt/dx ...
					{
						const double
							&PT_1_1=T(0,0), &PT_1_2=T(0,1), &PT_1_3=T(0,2),
							&PT_2_1=T(1,0), &PT_2_2=T(1,1), &PT_2_3=T(1,2),
							&PT_3_1=T(2,0), &PT_3_2=T(2,1), &PT_3_3=T(2,2),
							vi1=dt*v(0),vi2=dt*v(1),vi3=dt*v(2);
						const double t5 = PT_1_1*vi1;
						const double t6 = PT_1_2*vi2;
						const double t7 = PT_1_3*vi3;
						const double t2 = t5+t6+t7;
						const double t9 = PT_2_1*vi1;
						const double t10 = PT_2_2*vi2;
						const double t11 = PT_2_3*vi3;
						const double t3 = t9+t10+t11;
						const double t13 = PT_3_1*vi1;
						const double t14 = PT_3_2*vi2;
						const double t15 = PT_3_3*vi3;
						const double t4 = t13+t14+t15;
						const double t8 = t2*t2;
						const double t12 = t3*t3;
						const double t16 = t4*t4;
						const double t17 = t8+t12+t16;
						const double t18 = sqrt(t17);
						const double t19 = epsD+t18;
						const double t20 = 1.0/t19;
						const double t21 = 1.0/(t19*t19);
						const double t22 = 1.0/sqrt(t17);
						const double t23 = PT_1_1*t2*2.0;
						const double t24 = PT_2_1*t3*2.0;
						const double t25 = PT_3_1*t4*2.0;
						const double t26 = t23+t24+t25;
						const double t27 = PT_1_2*t2*2.0;
						const double t28 = PT_2_2*t3*2.0;
						const double t29 = PT_3_2*t4*2.0;
						const double t30 = t27+t28+t29;
						const double t31 = PT_1_3*t2*2.0;
						const double t32 = PT_2_3*t3*2.0;
						const double t33 = PT_3_3*t4*2.0;
						const double t34 = t31+t32+t33;
						Dt(0,0) = PT_1_1*t20-t2*t21*t22*t26*(1.0/2.0);
						Dt(0,1) = PT_1_2*t20-t2*t21*t22*t30*(1.0/2.0);
						Dt(0,2) = PT_1_3*t20-t2*t21*t22*t34*(1.0/2.0);
						Dt(1,0) = PT_2_1*t20-t3*t21*t22*t26*(1.0/2.0);
						Dt(1,1) = PT_2_2*t20-t3*t21*t22*t30*(1.0/2.0);
						Dt(1,2) = PT_2_3*t20-t3*t21*t22*t34*(1.0/2.0);
						Dt(2,0) = PT_3_1*t20-t4*t21*t22*t26*(1.0/2.0);
						Dt(2,1) = PT_3_2*t20-t4*t21*t22*t30*(1.0/2.0);
						Dt(2,2) = PT_3_3*t20-t4*t21*t22*t34*(1.0/2.0);
					}
					Dt *= fMax; // df_t/dt dt/dx
					Kt  = -cF*(t*(n.transpose()*Kn)); // df_t/dfMax dfMax/dx
				}
			}else{ // stick penalty
				state = CONTACT_STATE::CONTACT_STICK;
				f_t = -pFt*t;
				if( mode==UPDATE ) Dt = pFt*T;
			}
		}
	}
	return state;
}


void ContactFEM::nodalLinearPenaltyForceAndStiffness(
	Eigen::Vector3d& f_n, Eigen::Matrix3d& Kn, Eigen::Vector3d& f_t, Eigen::Matrix3d& Kt, Eigen::Matrix3d& Dt,
	const double pF, const double pFt, const double cF, const double g, const Eigen::Vector3d& n,
	const Eigen::Vector3d& v, CONTACT_STATE contactState, const double dt, UPDATE_MODE mode
){
	//const double epsD=1e-8; // regularization of direction normalization
	f_n.setZero(); Kn.setZero();  f_t.setZero(); Kt.setZero(); Dt.setZero();
	if( g<0.0 ){
		Eigen::Matrix3d N = n*n.transpose();
		Eigen::Matrix3d T = Eigen::Matrix3d::Identity() - N;


		f_n = (-pF*g)*n; // normal penalty force
		if( mode==UPDATE ) Kn =  pF*N;

		Eigen::Vector3d t = dt*T*v;
		if( contactState == CONTACT_STICK ){ // stick penalty
			f_t = -pFt*t;
			if( mode==UPDATE ) Dt = pFt*T;
		}else if( contactState == CONTACT_SLIP ){ // slip force
			double tu = t.norm(), fMax = (-cF*pF*g); // -pF*g >0 is the normal force magnitude -- (-cF*pF*g) is the Coulomb limit
			t /= (tu+epsD); // now t is the direction of tangential motion
			f_t = -fMax*t;
			if( mode==UPDATE ){
				// ... Kt = -df_t/dx = df_t/dfMax dfMax/dx + df_t/dt dt/dx ...
				{
					const double
						&PT_1_1=T(0,0), &PT_1_2=T(0,1), &PT_1_3=T(0,2),
						&PT_2_1=T(1,0), &PT_2_2=T(1,1), &PT_2_3=T(1,2),
						&PT_3_1=T(2,0), &PT_3_2=T(2,1), &PT_3_3=T(2,2),
						vi1=dt*v(0),vi2=dt*v(1),vi3=dt*v(2);
					const double t5 = PT_1_1*vi1;
					const double t6 = PT_1_2*vi2;
					const double t7 = PT_1_3*vi3;
					const double t2 = t5+t6+t7;
					const double t9 = PT_2_1*vi1;
					const double t10 = PT_2_2*vi2;
					const double t11 = PT_2_3*vi3;
					const double t3 = t9+t10+t11;
					const double t13 = PT_3_1*vi1;
					const double t14 = PT_3_2*vi2;
					const double t15 = PT_3_3*vi3;
					const double t4 = t13+t14+t15;
					const double t8 = t2*t2;
					const double t12 = t3*t3;
					const double t16 = t4*t4;
					const double t17 = t8+t12+t16;
					const double t18 = sqrt(t17);
					const double t19 = epsD+t18;
					const double t20 = 1.0/t19;
					const double t21 = 1.0/(t19*t19);
					const double t22 = 1.0/sqrt(t17);
					const double t23 = PT_1_1*t2*2.0;
					const double t24 = PT_2_1*t3*2.0;
					const double t25 = PT_3_1*t4*2.0;
					const double t26 = t23+t24+t25;
					const double t27 = PT_1_2*t2*2.0;
					const double t28 = PT_2_2*t3*2.0;
					const double t29 = PT_3_2*t4*2.0;
					const double t30 = t27+t28+t29;
					const double t31 = PT_1_3*t2*2.0;
					const double t32 = PT_2_3*t3*2.0;
					const double t33 = PT_3_3*t4*2.0;
					const double t34 = t31+t32+t33;
					Dt(0,0) = PT_1_1*t20-t2*t21*t22*t26*(1.0/2.0);
					Dt(0,1) = PT_1_2*t20-t2*t21*t22*t30*(1.0/2.0);
					Dt(0,2) = PT_1_3*t20-t2*t21*t22*t34*(1.0/2.0);
					Dt(1,0) = PT_2_1*t20-t3*t21*t22*t26*(1.0/2.0);
					Dt(1,1) = PT_2_2*t20-t3*t21*t22*t30*(1.0/2.0);
					Dt(1,2) = PT_2_3*t20-t3*t21*t22*t34*(1.0/2.0);
					Dt(2,0) = PT_3_1*t20-t4*t21*t22*t26*(1.0/2.0);
					Dt(2,1) = PT_3_2*t20-t4*t21*t22*t30*(1.0/2.0);
					Dt(2,2) = PT_3_3*t20-t4*t21*t22*t34*(1.0/2.0);
				}
				Dt *= fMax; // df_t/dt dt/dx
				Kt  = -cF*(t*(n.transpose()*Kn)); // df_t/dfMax dfMax/dx
			}
		}
	}
}


int ContactFEM::addRigidObstacle(const DiffableScalarField& g, double frictionCoeff){
	int nextID=rigidObstacles.size();
	rigidObstacles.push_back(&g);
	frictionCoefficient.push_back(frictionCoeff);
	return nextID;
}






QPSolver::QPSolver(){
#ifdef QPSOLVE_USE_MOSEK
	if( MSK_makeenv(&mosekEnv, NULL) != MSK_RES_OK) throw "MSK_makeenv";
#endif
}


QPSolver::~QPSolver(){
#ifdef QPSOLVE_USE_MOSEK
	if( mosekEnv != NULL){ MSK_deleteenv(&mosekEnv); }
#endif
}


int QPSolver::solve(
	Eigen::VectorXd& vi, Eigen::VectorXd& lambda, double eps, unsigned int numConstraints,
	SparseMatrixD& S, Eigen::VectorXd& r, SparseMatrixD& C, Eigen::VectorXd& cLower, Eigen::VectorXd& cUpper
){
	unsigned int& m = numConstraints;
#ifdef QPSOLVE_USE_MOSEK
	MSKtask_t task = NULL;
	MSKint32t nnz, *rowIdx=NULL, *colIdx=NULL; MSKrealt* vals=NULL;
	try{
		// init mosek stuff
		if( MSK_maketask(mosekEnv, m, vi.size(), &task) != MSK_RES_OK) throw "MSK_maketask";
		if( MSK_putdouparam(task, MSK_DPAR_INTPNT_QO_TOL_REL_GAP, 1e-14) != MSK_RES_OK) throw "MSK_putdouparam";
		if( MSK_putdouparam(task, MSK_DPAR_INTPNT_QO_TOL_INFEAS , 1e-14) != MSK_RES_OK) throw "MSK_putdouparam";
		if( MSK_putdouparam(task, MSK_DPAR_INTPNT_QO_TOL_MU_RED , 1e-14) != MSK_RES_OK) throw "MSK_putdouparam";
		if( MSK_putdouparam(task, MSK_DPAR_INTPNT_QO_TOL_PFEAS, 1e-14) != MSK_RES_OK) throw "MSK_putdouparam";
		if( MSK_putdouparam(task, MSK_DPAR_INTPNT_QO_TOL_DFEAS, 1e-14) != MSK_RES_OK) throw "MSK_putdouparam";
		//if( MSK_linkfunctotaskstream(task, MSK_STREAM_LOG, NULL, printstr) != MSK_RES_OK) throw "MSK_linkfunctotaskstream";
		//if( MSK_linkfunctotaskstream(task, MSK_STREAM_ERR, NULL, printstr) != MSK_RES_OK) throw "MSK_linkfunctotaskstream";
		//if( MSK_linkfunctotaskstream(task, MSK_STREAM_WRN, NULL, printstr) != MSK_RES_OK) throw "MSK_linkfunctotaskstream";
		if( MSK_appendvars(task, vi.size()) != MSK_RES_OK) throw "MSK_appendvars";
		// mark free variables
		for(unsigned int i=0; i<vi.size(); ++i){
			MSK_putvarbound(task, i, MSK_BK_FR, 0.0,0.0);
		}
		if( MSK_appendcons(task, m) != MSK_RES_OK) throw "MSK_appendcons";
		// cLower, cUpper
		for(unsigned int i=0; i<m; ++i){
			if( cLower(i)==cUpper(i) ){ // equality constraint
				if( MSK_putconbound(task, i, MSK_BK_FX, cLower(i), cUpper(i)) != MSK_RES_OK) throw "MSK_putconbound";
			}else //inequality -> lower bound constraint type ...
				if( MSK_putconbound(task, i, MSK_BK_LO, cLower(i), MSK_INFINITY) != MSK_RES_OK) throw "MSK_putconbound";
		}
		// C
		nnz = C.nonZeros(); rowIdx = new MSKint32t[nnz]; colIdx = new MSKint32t[nnz]; vals = new MSKrealt[nnz];
		nnz = 0;
		for(Eigen::Index k=0; k < C.outerSize(); ++k){ for(SparseMatrixD::InnerIterator it(C,k); it; ++it){
			rowIdx[nnz] = it.row(); colIdx[nnz] = it.col(); vals[nnz] = it.value(); ++nnz;
		}}
		if( MSK_putaijlist(task, nnz, rowIdx, colIdx, vals) != MSK_RES_OK) throw "MSK_putaijlist";
		delete[] rowIdx; delete[] colIdx; delete[] vals; rowIdx=colIdx=NULL; vals=NULL;
					
		// r
		for(unsigned int i=0; i<vi.size(); ++i){
			if( MSK_putcj(task, i, r(i) ) != MSK_RES_OK) throw "MSK_putcj";
		}
		// S
		nnz = S.nonZeros(); rowIdx = new MSKint32t[nnz]; colIdx = new MSKint32t[nnz]; vals = new MSKrealt[nnz];
		nnz = 0;
		for(Eigen::Index k=0; k < S.outerSize(); ++k){ for(SparseMatrixD::InnerIterator it(S,k); it; ++it){
			if( it.row() >= it.col() ){ // mosek wants ONLY the lower triangular part and assumes symmetry internally
				rowIdx[nnz] = it.row(); colIdx[nnz] = it.col(); vals[nnz] = it.value(); ++nnz;
			}
		}}

		MSKrescodee res;
		if((res=MSK_putqobj(task, nnz, rowIdx, colIdx, vals)) != MSK_RES_OK) throw res;//"MSK_putqobj";
		delete[] rowIdx; delete[] colIdx; delete[] vals; rowIdx=colIdx=NULL; vals=NULL;
		// solve
		res = MSK_optimize(task);
		if( res != MSK_RES_OK && res != MSK_RES_TRM_STALL) throw res;
		MSK_getxx(task, MSK_SOL_ITR, vi.data() );
		lambda.resize(m);
		MSK_gety(task, MSK_SOL_ITR, lambda.data());
		//fc = (C.transpose()*lambda);
		//fc = (1.0/dt)*(C.transpose()*lambda);
		//MSK_solutionsummary (task, MSK_STREAM_LOG);
		//MSKrealt* sol; sol = new MSKrealt[vi.size()];
		//MSK_getxx(task, MSK_SOL_ITR, sol );
		//for(unsigned int i=0; i<vi.size(); ++i) printf(" %.2lg", sol[i]);
		//delete[] sol; sol=NULL;
		//MSK_writedata(task, "lastMosekTask.opf");
		return 0;
	}catch(MSKrescodee res){
		char symname[MSK_MAX_STR_LEN];
		char desc[MSK_MAX_STR_LEN];
		MSK_getcodedesc (res, symname, desc);
		printf("MOSEK error: %s - '%s'\n", symname, desc);
		return -1;
	}catch(char* msg){ printf("\n%% MOSEK error: %s", msg); return -2;
	}catch(...){ printf("\n%% MOSEK error: unknown"); return -3;
	}
	MSK_deletetask(&task);
#elif _USE_OSQP
	OsqpEigen::Solver solver;
	solver.data()->setNumberOfVariables(vi.size());
	solver.data()->setNumberOfConstraints(m);
	solver.settings()->setVerbosity(false);
	solver.settings()->setPolish(true); // actually improves accuracy quite a bit
	solver.settings()->setAbsoluteTolerance(eps);
	solver.settings()->setDelta(1e-20);
	if(!solver.data()->setHessianMatrix<double>(S)) return -2; //if(!solver.data()->setHessianMatrix<double>(0.5*(S+SparseMatrixD(S.transpose())))) return -2;
	if(!solver.data()->setGradient<-1>(r)) return -3;
	if(!solver.data()->setLinearConstraintsMatrix<double>(C)) return -4;
	if(!solver.data()->setLowerBound<-1>(cLower)) return -5;
	if(!solver.data()->setUpperBound<-1>(cUpper)) return -6;
	if(!solver.initSolver()) return -7;
	//if(!solver.setPrimalVariable<double,-1>(vi)) return -8;
	//solver.settings()->setWarmStart(true); // how to set initial guess properly?
	if(!solver.solve()) return -9;
	vi = solver.getSolution();
	lambda.resize(m);
	if(!solver.getSolutionDualVariable(lambda)) return -10; // there seems to be a bug in osqp-eigen (0.3.0) -- getDualVariable returns m_workspace->y but we need m_workspace->solution->y here -- added new getter function manually
	//cout << endl << "lagrange multipliers" << lambda << endl;
	lambda*=-1.0;
	//fc = -(C.transpose()*lambda);
	//cout << endl << "|lambda| " << lambda.size() << "; m " << m << "; |fc| " << fc.size() << endl;
	//cout << endl << "lamda = [" << endl << lambda << " ];" << endl;
	return 0;
#else
	printf("\n%%!!! NO QP LIBRARY ENABLED IN BUILD CONFIGURATION !!!\n");
	return 1;
#endif
}
