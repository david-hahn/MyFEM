#include "ViscousMaterialParameterHandler.h"
#include "Materials.h"
#include "EigenmodeViscosityModel.h"
#include "LinearFEM.h"

#include <vtkDoubleArray.h>
#include <vtkUnstructuredGrid.h>
#include <vtkCellData.h>

using namespace MyFEM;

class ViscousForceDerivativeAssemblyOp{
public:
	double phiQ;
	ParameterHandler& qHdl;
	Eigen::MatrixXd& g_q; Eigen::VectorXd& phiQ_q;
	Eigen::Matrix3d dv;
	Vector9d stress_nu, stress_h;
	Eigen::VectorXd f_nu, f_h;    // derivatives of force (elem-wise) wrt. params
	// for PowerLawViscosityModel note that this operator also writes to qHdl.phiQ_qq (must have size 2)
	ViscousForceDerivativeAssemblyOp(Eigen::MatrixXd& g_q_, Eigen::VectorXd& phiQ_q_, ParameterHandler& qHdl_, LinearFEM& fem) : g_q(g_q_), phiQ_q(phiQ_q_), qHdl(qHdl_) {}
	inline void initialize(LinearFEM& fem){
		stress_nu.setZero(); stress_h.setZero();
		phiQ=0.0;
		g_q.resize( qHdl.getNumberOfDOFs(fem), qHdl.getNumberOfParams(fem) );
		g_q.setZero();
		phiQ_q.resize( qHdl.getNumberOfParams(fem) );
		phiQ_q.setZero();
	}
	inline void calculateElement(LinearFEM& fem, unsigned int k){
		fem.computeVelocityGradient(dv, k, fem.getVectorVelocities());
		matrixComponentRefs(dv);
		
		double nu,h;
		vtkSmartPointer<PowerLawViscosityModel> vmPL;
		vtkSmartPointer<RotationInvariantViscosityModel> vmRI;
		if( dv.squaredNorm() > sqrt(DBL_MIN) && (vmPL=PowerLawViscosityModel::SafeDownCast( fem.getViscosityModel(k) )) ){
			vmPL->getViscosity(fem.getViscosityParameters(k).data(), nu);
			vmPL->getPowerLawH(fem.getViscosityParameters(k).data(),  h);

			//ToDo:                                          vvvvvvv -- this is bad style, store by ref in constructor ...
			phiQ+=ParameterHandler::regFcn(nu,phiQ_q[0],qHdl.phiQ_qq[0], 0.0,  1e-3 ); // penalize negative viscosity - unphysical
			phiQ+=ParameterHandler::regFcn(h, phiQ_q[1],qHdl.phiQ_qq[1], 0.1 , 1e-7 ); // penalize low power-law index (simulation stability suffers for low h)
			phiQ+=ParameterHandler::regFcn(h, phiQ_q[1],qHdl.phiQ_qq[1], 0.05, 1e-4 ); // and some more for really low values
			// non-Newtonian viscous stress derivatives ... auto-generated, reads dv-components, nu and h, writes stress_nu and stress_h
			#include "codegen_PowerLawViscosity_derivs.h"
		}else if( dv.squaredNorm() > FLT_EPSILON && (vmRI=RotationInvariantViscosityModel::SafeDownCast( fem.getViscosityModel(k) )) ){
			Eigen::Matrix3d F;
			fem.computeDeformationGradient(F,k);
			matrixComponentRefs(F);
			const double
				&dv_1_1 = dv(0,0), &dv_1_2 = dv(0,1), &dv_1_3 = dv(0,2),
				&dv_2_1 = dv(1,0), &dv_2_2 = dv(1,1), &dv_2_3 = dv(1,2),
				&dv_3_1 = dv(2,0), &dv_3_2 = dv(2,1), &dv_3_3 = dv(2,2);
			#include "codegen_RotInvariantViscosity_pot_derivs.h" // writes stress_nu
			stress_h.setZero(); // not used
		}else{
			stress_nu.setZero(); stress_h.setZero(); // viscous force is always zero if velocity gradient is zero - or there is no viscosity model applied on this element
		}

		f_nu = -fem.getVolume(k)*fem.getDFdx(k).transpose()*stress_nu;
		f_h  = -fem.getVolume(k)*fem.getDFdx(k).transpose()*stress_h;
	}

	inline void assembleVectorDOF(LinearFEM& fem, unsigned int k, unsigned int i, unsigned int idof, unsigned int lidof, unsigned int gidof){
		g_q(gidof,0) += f_nu(lidof);
		if( g_q.cols()==2) g_q(gidof,1) += f_h(lidof);
	}

	inline void assembleMatrixDOF(LinearFEM& fem, unsigned int k, unsigned int i, unsigned int idof, unsigned int lidof, unsigned int gidof, unsigned int j, unsigned int jdof, unsigned int ljdof, unsigned int gjdof){}
	inline void finalize(LinearFEM& fem){}

};

class ViscousSeriesForceDerivativeAssemblyOp : public ViscousForceDerivativeAssemblyOp{
public:
	Eigen::MatrixXd f_nus;
	ViscousSeriesForceDerivativeAssemblyOp(Eigen::MatrixXd& g_q_, Eigen::VectorXd& phiQ_q_, ParameterHandler& qHdl_, LinearFEM& fem) : ViscousForceDerivativeAssemblyOp(g_q_,phiQ_q_,qHdl_,fem) {}
	inline void calculateElement(LinearFEM& fem, unsigned int k){
		f_nus.resize(12,qHdl.getNumberOfParams(fem));
		f_nus.setZero();
		fem.computeVelocityGradient(dv, k, fem.getVectorVelocities());
		matrixComponentRefs(dv);
		
		vtkSmartPointer<PowerSeriesViscosityModel> vm;
		if( dv.squaredNorm() > sqrt(DBL_MIN) && (vm=PowerSeriesViscosityModel::SafeDownCast( fem.getViscosityModel(k) )) ){
			for(unsigned int i=0; i < qHdl.getNumberOfParams(fem); ++i){
				double nu = fem.getViscosityParameters(k)(i);
				double h  = vm->powerIndices(i);
				// non-Newtonian viscous stress derivatives ... auto-generated, reads dv-components, nu and h, writes stress_nu and stress_h
				#include "codegen_PowerLawViscosity_derivs.h"
				f_nus.col(i) = -fem.getVolume(k)*fem.getDFdx(k).transpose()*stress_nu;
			}
		}

		for(unsigned int i=0; i < qHdl.getNumberOfParams(fem); ++i){
			phiQ+=ParameterHandler::regFcn(fem.getViscosityParameters(k)(i), phiQ_q(i), qHdl.phiQ_qq(i));
			//phiQ+=ParameterHandler::barFcn(fem.getViscosityParameters(k)(i), phiQ_q(i), qHdl.phiQ_qq(i),0.0,1e-10); // will ensure parameters > 0.0, but contribute a penalty up to 1.0 (non-smooth cut-off)
		}
	}
	inline void assembleVectorDOF(LinearFEM& fem, unsigned int k, unsigned int i_, unsigned int idof, unsigned int lidof, unsigned int gidof){
		for(unsigned int i=0; i < qHdl.getNumberOfParams(fem); ++i){
			g_q(gidof,i) += f_nus(lidof,i);
		}
	}
};






void GlobalPLViscosityMaterialParameterHandler::getCurrentParams(  Eigen::VectorXd& q, LinearFEM& fem){
	vtkSmartPointer<PowerLawViscosityModel> vm;
	bool done=false;
	for(unsigned int k=0; k<fem.getNumberOfElems() && !done; ++k){
		if( vm=PowerLawViscosityModel::SafeDownCast( fem.getViscosityModel(k) ) ){
			vm->getViscosity(fem.getViscosityParameters(0).data(), q[0]);
			vm->getPowerLawH(fem.getViscosityParameters(0).data(), q[1]);
			done=true;
		}
	}
}
void GlobalPLViscosityMaterialParameterHandler::setNewParams(const Eigen::VectorXd& q, LinearFEM& fem){
	vtkSmartPointer<PowerLawViscosityModel> vm;
	for(unsigned int k=0; k<fem.getNumberOfElems(); ++k){
		if( vm=PowerLawViscosityModel::SafeDownCast( fem.getViscosityModel(k) ) ){
			vm->setViscosity(fem.getViscosityParameters(k).data(), q[0]);
			vm->setPowerLawH(fem.getViscosityParameters(k).data(), q[1]);
		}
	}
}
double GlobalPLViscosityMaterialParameterHandler::computeConstraintDerivatives(Eigen::MatrixXd& g_q, Eigen::VectorXd& phiQ_q, LinearFEM& fem){
	phiQ_qq.resize(2); phiQ_qq.setZero();
	ViscousForceDerivativeAssemblyOp dfop(g_q,phiQ_q, *this,fem);
	LinearFEM::assemblyLoop(fem, dfop);
	return dfop.phiQ;
}




unsigned int GlobalPSViscosityMaterialParameterHandler::getNumberOfParams(const LinearFEM& fem){
	//ToDo: specify which body in the FEM mesh we're working on, otherwise we can't support multiple bodies with different viscosity models
	if( nParams==0 ){ //not initialized ...
		std::map<unsigned int,vtkSmartPointer<ViscosityModel> > vmdls = fem.getViscosityModels();
		vtkSmartPointer<PowerSeriesViscosityModel> vm;
		for(std::map<unsigned int,vtkSmartPointer<ViscosityModel> >::iterator it=vmdls.begin(); it!=vmdls.end(); ++it){
			if( vm=PowerSeriesViscosityModel::SafeDownCast( it->second ) ){
				nParams = std::max(nParams, vm->getNumberOfParameters() );
			}
		}
	}
	return nParams;
}
void GlobalPSViscosityMaterialParameterHandler::getCurrentParams(  Eigen::VectorXd& q, LinearFEM& fem){
	vtkSmartPointer<PowerSeriesViscosityModel> vm;
	bool done=false;
	for(unsigned int k=0; k<fem.getNumberOfElems() && !done; ++k){
		if( vm=PowerSeriesViscosityModel::SafeDownCast( fem.getViscosityModel(k) ) ){
			if( fem.getViscosityParameters(k).size()==getNumberOfParams(fem) ){
				q = fem.getViscosityParameters(k);
				done=true;
			}
		}
	}
}
void GlobalPSViscosityMaterialParameterHandler::setNewParams(const Eigen::VectorXd& q, LinearFEM& fem){
	vtkSmartPointer<PowerSeriesViscosityModel> vm;
	for(unsigned int k=0; k<fem.getNumberOfElems(); ++k){
		if( vm=PowerSeriesViscosityModel::SafeDownCast( fem.getViscosityModel(k) ) ){
			fem.getViscosityParameters(k) = q;
		}
	}
}
double GlobalPSViscosityMaterialParameterHandler::computeConstraintDerivatives(Eigen::MatrixXd& g_q, Eigen::VectorXd& phiQ_q, LinearFEM& fem){// partial derivatives of the constraint function wrt. the design parameters + regularizer term phiQ(q) and dphiQ_dq
	phiQ_qq.resize( getNumberOfParams(fem) ); phiQ_qq.setZero();
	ViscousSeriesForceDerivativeAssemblyOp dfop(g_q,phiQ_q, *this,fem);
	LinearFEM::assemblyLoop(fem, dfop);
	return dfop.phiQ;
}





void GlobalRotationInvariantViscosityMaterialParameterHandler::getCurrentParams(  Eigen::VectorXd& q, LinearFEM& fem){
	vtkSmartPointer<RotationInvariantViscosityModel> vm;
	bool done=false;
	for(unsigned int k=0; k<fem.getNumberOfElems() && !done; ++k){
		if( vm=RotationInvariantViscosityModel::SafeDownCast( fem.getViscosityModel(k) ) ){
			vm->getViscosity(fem.getViscosityParameters(0).data(), q[0]);
			done=true;
		}
	}
}
void GlobalRotationInvariantViscosityMaterialParameterHandler::setNewParams(const Eigen::VectorXd& q, LinearFEM& fem){
	vtkSmartPointer<RotationInvariantViscosityModel> vm;
	for(unsigned int k=0; k<fem.getNumberOfElems(); ++k){
		if( vm=RotationInvariantViscosityModel::SafeDownCast( fem.getViscosityModel(k) ) ){
			vm->setViscosity(fem.getViscosityParameters(k).data(), q[0]);
		}
	}
}
double GlobalRotationInvariantViscosityMaterialParameterHandler::computeConstraintDerivatives(Eigen::MatrixXd& g_q, Eigen::VectorXd& phiQ_q, LinearFEM& fem){
	ViscousForceDerivativeAssemblyOp dfop(g_q,phiQ_q, *this,fem);
	LinearFEM::assemblyLoop(fem, dfop);
	return dfop.phiQ;
}







#include "vtkPiecewiseFunction.h"

class EigenmodeViscosityForceDerivativeAssemblyOp : public ViscousForceDerivativeAssemblyOp{
public:
	EigenmodeViscosityForceDerivativeAssemblyOp(Eigen::MatrixXd& g_q_, Eigen::VectorXd& phiQ_q_, ParameterHandler& qHdl_, LinearFEM& fem) : ViscousForceDerivativeAssemblyOp(g_q_,phiQ_q_,qHdl_,fem) {}
	inline void calculateElement(LinearFEM& fem, unsigned int k){
		fem.computeVelocityGradient(dv, k, fem.getVectorVelocities());
		matrixComponentRefs(dv);
		
		vtkSmartPointer<EigenmodeViscosityModel> vm;
		if( vm=EigenmodeViscosityModel::SafeDownCast( fem.getViscosityModel(k) ) ){
			double nu = vm->getBaseViscosity();
			#include "codegen_NewtonianViscosity_derivs.h"
			f_nu = -fem.getVolume(k)*fem.getDFdx(k).transpose()*stress_nu;

			//                                       base viscosity is the last param
			phiQ+=ParameterHandler::regFcn(nu, phiQ_q(qHdl.getNumberOfParams(fem)-1), qHdl.phiQ_qq(qHdl.getNumberOfParams(fem)-1));
		}

	}
	inline void assembleVectorDOF(LinearFEM& fem, unsigned int k, unsigned int i_, unsigned int idof, unsigned int lidof, unsigned int gidof){
		//        base viscosity is the last param
		g_q(gidof, qHdl.getNumberOfParams(fem)-1) += f_nu(lidof);
	}
};

void EigenmodeViscosityMaterialParameterHandler::getCurrentParams(  Eigen::VectorXd& q, LinearFEM& fem){ // writes q, does not change fem
	vtkSmartPointer<EigenmodeViscosityModel> vm;
	bool done=false;
	for(unsigned int k=0; k<fem.getNumberOfElems() && !done; ++k){
		if( vm=EigenmodeViscosityModel::SafeDownCast( fem.getViscosityModel(k) ) ){
			//ToDo: specify which body in the FEM mesh we're working on, otherwise we can't support multiple bodies with different viscosity models
			vm->getViscosityCoefficients(q); // will resize q if required
			done=true;
		}
	}
}
void EigenmodeViscosityMaterialParameterHandler::setNewParams(const Eigen::VectorXd& q, LinearFEM& fem){ // writes to fem, does not change q
	bool done=false;
	std::map<unsigned int,vtkSmartPointer<ViscosityModel> > vmdls = fem.getViscosityModels();
	vtkSmartPointer<EigenmodeViscosityModel> vm;
	for(std::map<unsigned int,vtkSmartPointer<ViscosityModel> >::iterator it=vmdls.begin(); it!=vmdls.end() && !done; ++it){
		if( vm=EigenmodeViscosityModel::SafeDownCast( it->second ) ){
			//ToDo: specify which body in the FEM mesh we're working on, otherwise we can't support multiple bodies with different viscosity models
			vm->setViscosityCoefficients(q,fem);
			done = true; // assumes that we only have one global EigenmodeViscosityModel!
			//printf("\n%% set params for body %d: ", it->first); cout << q.transpose();
		}
	}
}
unsigned int EigenmodeViscosityMaterialParameterHandler::getNumberOfParams(const LinearFEM& fem){ // how many parameters are we working with (rows in q)
	//ToDo: specify which body in the FEM mesh we're working on, otherwise we can't support multiple bodies with different viscosity models
	if( nParams==0 ){ //not initialized ...
		std::map<unsigned int,vtkSmartPointer<ViscosityModel> > vmdls = fem.getViscosityModels();
		vtkSmartPointer<EigenmodeViscosityModel> vm;
		bool done=false;
		for(std::map<unsigned int,vtkSmartPointer<ViscosityModel> >::iterator it=vmdls.begin(); it!=vmdls.end() && !done; ++it){
			if( vm=EigenmodeViscosityModel::SafeDownCast( it->second ) ){
				Eigen::VectorXd f; vm->getAdjustmentFrequencies(f);
				nParams = f.size(); done=true;
			}
		}
	}
	return nParams;
}
double EigenmodeViscosityMaterialParameterHandler::computeConstraintDerivatives(Eigen::MatrixXd& g_q, Eigen::VectorXd& phiQ_q, LinearFEM& fem){ // partial derivatives of the constraint function wrt. the design parameters + regularizer term phiQ(q) and dphiQ_dq
	// here we need to compute the derivative of the viscous force wrt. the viscosity coefficients
	// we have in general f_v = -D*v, so df_v/dq = -dD/dq * v
	// where D is composed of per-mode adjustments and the remaining Newtonian damping matrix
	// similarly we can compute -dD/dq*v first for the standard Newtonian case - depending only on the last entry in q -- the default viscosity
	// and then add the per-mode adjustments - as we use a piecewise-linear adjustment function, these depend on at most 2 entries in q.
	phiQ_qq.resize( getNumberOfParams(fem) ); phiQ_qq.setZero();
	EigenmodeViscosityForceDerivativeAssemblyOp dfop(g_q,phiQ_q, *this,fem);
	LinearFEM::assemblyLoop(fem, dfop); // the standard assembly loop takes care of the base viscosity assuming Newtonian behaviour

	// now the adjustments similar to EigenmodeViscosityModel::assembleForceAndDampingMatrix ...
	std::map<unsigned int,vtkSmartPointer<ViscosityModel> > vmdls = fem.getViscosityModels();
	vtkSmartPointer<EigenmodeViscosityModel> vm;
	bool done=false;
	for(std::map<unsigned int,vtkSmartPointer<ViscosityModel> >::iterator it=vmdls.begin(); it!=vmdls.end() && !done; ++it){
		if( vm=EigenmodeViscosityModel::SafeDownCast( it->second ) ) done=true;
	}

	//ToDo: ... this ...
	// for each adjusted mode, we add the force update 	 (dFactor-1.0)*ratioDtoK*eigvals(i)* (evn*evn.transpose()) * (-v) where dFactor = nu_fcn->GetValue( frq ) / nu and  ratioDtoK = nu * trace(D_unit) / trace(K) and D_unit is the Newtonian damping matrix for unit viscosity
	// which simplifies to (fn-nu)*c*(n*n')*(-v), where c is a constant independent of any parameter, n is an eigenvector, v is the current velocity, fn is the value of the adjustment function and nu is the base viscosity
	// consequently, we have derivatives c*(n*n')*(-v) wrt. fn and c*(n*n')*(+v) wrt. nu
	// finally, fn depends linearly on two parameters (depending on where on the pw-linear adjustment function the current frequency lies), so we need to split the first term proportionally
	unsigned int j=1;
	Eigen::VectorXd frequencies; vm->getAdjustmentFrequencies(frequencies);
	for(int i=0; i<vm->getNumberOfComputedModes(); ++i){ // adjust damping for mode i -- i==0 corresponds to the highest frequency mode we computed, i==n-1 to the lowest possible frequency (not counting rigid modes)
		Eigen::VectorXd evn = fem.M*vm->getEigenmodes().col(i);
		double a, frq = sqrt(vm->getEigenvalues()(i));
		Eigen::VectorXd cnntv = vm->getBaseDampingRatio()*vm->getEigenvalues()(i)*evn*(evn.dot(fem.getVectorVelocities()));
		
		// find index j such that frequencies(j-1) <= frq <= frequencies(j)
		j=1; // we can probably start from the previous value as both frequencies and eigenvalues should be sorted -- however the sort type (asc/desc) might be different.
		if( frequencies(0) > frq ) // clamp low end -- lowest mode is already higher than current frequency (can happen if frequencies change due to stiffness change)
			a=0.0;
		else{
			while( j<frequencies.size() && !( frequencies(j-1) <= frq && frq <= frequencies(j) ) ) ++j;
			if( j==frequencies.size() ){ // clamp high end (frq > frequencies(end))
				a=1.0; --j;
				//printf("\n%% mode %d: low %.2lf, frq %.2lf, high %.2lf, intv %d (%.3lf) ", i, frequencies(j-1), frq, frequencies(j), j,a);
			}else // default case
				a = (frq-frequencies(j-1))/(frequencies(j)-frequencies(j-1));
		}
		

		g_q.col( nParams-1 ) +=  cnntv;
		g_q.col( j-1 ) -= (cnntv*(1.0-a));
		g_q.col( j   ) -= (cnntv*     a);

	}

	return dfop.phiQ;
}
