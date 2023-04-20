#include "LinearFEM.h"
#include "Materials.h"
#include "ElasticMaterialParameterHandler.h"
#include "DifferentiableSpline.h"
#include <unsupported\Eigen\AutoDiff>

using namespace MyFEM;

// local helper function
void computePrincipalStretchAndDerivatives(Eigen::Vector3d& ps, Eigen::Matrix<double, 3,9>& dpsdF, const Eigen::Matrix3d& F){ 
	ps.setZero(); dpsdF.setZero();
	//{ // auto-gen does not really work for SVD derivs
	//	matrixComponentRefs(F);
	//	#include "codegen_SVDderivs.h" // auto-generated code for singular values and derivatives, writes ps and dpsdF (not neccessarily all components)
	//	if( isnan(ps(0)) || isnan(ps(1)) || isnan(ps(2)) ) ps.setZero();
	//}
	//if( 1 ){ // get dpsdF from finite differences (needs to compute 9+1 SVDs of a 3x3 matrix)
	//	Eigen::JacobiSVD<Eigen::Matrix3d> svd(F); ps=svd.singularValues();
	//	const double fdh = 1e-5; // don't go too low, or we'll see rounding errors
	//	for(int i=0;i<3;++i) for(int j=0;j<3;++j){
	//		F(i,j) += fdh;
	//		svd.compute(F);
	//		dpsdF.col(3*i+j) = (svd.singularValues()-ps)/fdh; //ToDo: could use a faster specialized implementation for 3x3 SVD
	//		F(i,j) -= fdh;
	//	}
	//	// cout << "FD dpsdF:" << endl << dpsdF << endl << endl;
	//}else
	{ // let's see if Eigen's autodiff module works for SVD -- it does!
		typedef Eigen::AutoDiffScalar<Vector9d> adouble;
		Eigen::Matrix<adouble,3,3> aF = F;
		for(int i=0;i<3;++i) for(int j=0;j<3;++j){ 
			aF(i,j).derivatives() = Vector9d::Unit( 9, 3*i+j );
		}
		Eigen::JacobiSVD<Eigen::Matrix<adouble,3,3>, Eigen::HouseholderQRPreconditioner > svd(aF);
		Eigen::Matrix<adouble,3,1> aps = svd.singularValues();
		for(int i=0;i<3;++i){
			ps(i) = aps(i).value();
			dpsdF.row(i) = aps(i).derivatives().transpose();
			//if( (F-Eigen::Matrix3d::Identity()).norm()>1e-5 ) cout << aps(i).derivatives().transpose() << endl;
		}
	}
}

void PrincipalStretchMaterial::setElasticParams(double* unused, double lameLambda, double lameMu){
	fPrime = vtkSmartPointer<DifferentiableSpline>::New(); 
	hPrime = vtkSmartPointer<DifferentiableSpline>::New();
	// set fPrime and hPrime to Neohookean-like behaviour ... see Barbic et al. 2015 ...

	fPrime->ClosedOff();
	fPrime->SetLeftConstraint(2);  fPrime->SetLeftValue(0.0);
	fPrime->SetRightConstraint(2); fPrime->SetRightValue(0.0);

	hPrime->ClosedOff();
	hPrime->SetLeftConstraint(2);  hPrime->SetLeftValue(0.0);
	hPrime->SetRightConstraint(2); hPrime->SetRightValue(0.0);

	fPrime->SetDefaultTension(0.3); hPrime->SetDefaultTension(0.3);
	//ToDo: let the user choose the number of control points and the range
	//Note: it's quite important to have x==1.0 in there -- could add that manually later, and that fPrime(1)+hPrime(1)==0, otherwise we change the rest shape
	//      it's also very important to select control points such that the spline does not oscillate -- maybe a different spline formulation would be better here?
	//xp.resize(7); xp << 0.05, 0.2, 0.6, 1.0, 2.0, 5.0, 1000.0;
	xp.resize(5); xp << 0.1, 0.3, 1.0, 3.0, 1000.0; // we'd want a fairly high upper limit here, as h will be evaluated on the volume change (product of all 3 principal stretches, so 10^3 allows for 10x uniform scaling)
	//int np=20; xp.resize(np); xp.setLinSpaced(std::log(0.001), std::log(1000.0)); xp = xp.array().exp();
	//xp.conservativeResize(np+1); xp(np)=1.0; std::sort(xp.data(), xp.data() + xp.size());
	cout << endl << "psm-xp: " << xp.transpose() << endl;
	for(int i=0; i<xp.size(); ++i){
		double x = xp(i),
			fp = lameMu*x,
			hp = lameLambda*log(x)/x - lameMu/x;
		if( i==0 ){ xp(i)=x=0.0;} //move first point to 0.0 so we always get defined derivatives under extreme compression
		fPrime->AddPoint(x, fp);
		hPrime->AddPoint(x, hp);
	}

	//printf("\n xfPhPdfPdhP = [ ");
	//for(double x=0; x<22.0; x+=0.01){
	//	printf("\n %.4lg  %.4lg  %.4lg", x, fPrime->Evaluate(x), hPrime->Evaluate(x));
	//	//printf("   %.4lg  %.4lg ",fPrime->EvaluateTDerivative(x), hPrime->EvaluateTDerivative(x)); 
	//}
	//printf(" ];\n");

}

void PrincipalStretchMaterial::computeEnergyStressAndHessian(Eigen::Matrix3d& F, double* params, double& energy, Vector9d& PK1stress, Matrix9d& H){
	Eigen::Vector3d ps; // principal stretches (singular values of F)
	Eigen::Matrix<double, 3,9> dpsdF; // derivatives of principal stretches wrt. components of F

	computePrincipalStretchAndDerivatives(ps, dpsdF, F);

	Eigen::Vector3d  pStress;
	Eigen::Matrix3d dpStressdps;
	// see Barbic et al 2015, Eq (13)-(15)
	pStress(0) = fPrime->Evaluate(ps(0)) + hPrime->Evaluate(ps.prod())*ps(1)*ps(2);
	pStress(1) = fPrime->Evaluate(ps(1)) + hPrime->Evaluate(ps.prod())*ps(2)*ps(0);
	pStress(2) = fPrime->Evaluate(ps(2)) + hPrime->Evaluate(ps.prod())*ps(0)*ps(1);

	dpStressdps.setZero();
	dpStressdps(0,0) = fPrime->EvaluateTDerivative(ps(0)) + hPrime->EvaluateTDerivative(ps.prod())*ps(1)*ps(2)*ps(1)*ps(2);
	dpStressdps(1,1) = fPrime->EvaluateTDerivative(ps(1)) + hPrime->EvaluateTDerivative(ps.prod())*ps(2)*ps(0)*ps(2)*ps(0);
	dpStressdps(2,2) = fPrime->EvaluateTDerivative(ps(2)) + hPrime->EvaluateTDerivative(ps.prod())*ps(0)*ps(1)*ps(0)*ps(1);

	dpStressdps(0,1) = hPrime->Evaluate(ps.prod())*ps(2)  + hPrime->EvaluateTDerivative(ps.prod())*ps.prod()*ps(2);
	dpStressdps(0,2) = hPrime->Evaluate(ps.prod())*ps(1)  + hPrime->EvaluateTDerivative(ps.prod())*ps.prod()*ps(1);
	dpStressdps(1,2) = hPrime->Evaluate(ps.prod())*ps(0)  + hPrime->EvaluateTDerivative(ps.prod())*ps.prod()*ps(0);

	dpStressdps(1,0) = dpStressdps(0,1);
	dpStressdps(2,0) = dpStressdps(0,2);
	dpStressdps(2,1) = dpStressdps(1,2);

	//cout << endl << "pStress = [" << pStress.transpose() << "];";
	PK1stress = pStress.transpose() * dpsdF;// cout << endl << "PK1stress = [" << endl << PK1stress.transpose() << "];" << endl;
	H = dpsdF.transpose() * dpStressdps * dpsdF;
	energy = 0.0; // not implemented at the moment
}



class PrincipalStretchForceDerivativeAssemblyOp{
public:
	double phiQ;
	ParameterHandler& qHdl;
	Eigen::MatrixXd& g_q; Eigen::VectorXd& phiQ_q;
	Eigen::Matrix3d F;
	Eigen::MatrixXd f_qT;    // derivatives of force (elem-wise) wrt. params (p x 12)
	PrincipalStretchForceDerivativeAssemblyOp(Eigen::MatrixXd& g_q_, Eigen::VectorXd& phiQ_q_, ParameterHandler& qHdl_, LinearFEM& fem) : g_q(g_q_), phiQ_q(phiQ_q_), qHdl(qHdl_) {}

	inline void initialize(LinearFEM& fem){
		phiQ=0.0;
		f_qT.resize( qHdl.getNumberOfParams(fem), 12 );
		g_q.resize( qHdl.getNumberOfDOFs(fem), qHdl.getNumberOfParams(fem) );
		g_q.setZero();
	}

	inline void calculateElement(LinearFEM& fem, unsigned int k){
		fem.computeDeformationGradient(F, k);
		vtkSmartPointer<PrincipalStretchMaterial> psm;
		if( psm = PrincipalStretchMaterial::SafeDownCast( fem.getMaterialModel(k) ) ){
			Eigen::Vector3d ps; // principal stretches (singular values of F)
			Eigen::Matrix<double, 3,9> dpsdF; // derivatives of principal stretches wrt. components of F

			computePrincipalStretchAndDerivatives(ps, dpsdF, F); //ToDo: consider storing result from last stress evaluation?

			Eigen::MatrixXd dpStressdqT( qHdl.getNumberOfParams(fem) ,3);
			Eigen::VectorXd dfPdq, dhPdq;
			//pStress(0) = fPrime->Evaluate(ps(0)) + hPrime->Evaluate(ps.prod())*ps(1)*ps(2);
			//pStress(1) = fPrime->Evaluate(ps(1)) + hPrime->Evaluate(ps.prod())*ps(2)*ps(0);
			//pStress(2) = fPrime->Evaluate(ps(2)) + hPrime->Evaluate(ps.prod())*ps(0)*ps(1);
			unsigned int n = psm->xp.size();
			psm->fPrime->EvaluateYDerivative( ps(0), dfPdq ); dpStressdqT.block(0,0,n,1) = dfPdq;
			psm->fPrime->EvaluateYDerivative( ps(1), dfPdq ); dpStressdqT.block(0,1,n,1) = dfPdq;
			psm->fPrime->EvaluateYDerivative( ps(2), dfPdq ); dpStressdqT.block(0,2,n,1) = dfPdq;

			psm->hPrime->EvaluateYDerivative( ps.prod(), dhPdq );
			dpStressdqT.block(n,0,n,1) =  dhPdq*ps(1)*ps(2);
			dpStressdqT.block(n,1,n,1) =  dhPdq*ps(2)*ps(0);
			dpStressdqT.block(n,2,n,1) =  dhPdq*ps(0)*ps(1);

			f_qT = -fem.getVolume(k)*dpStressdqT*dpsdF*fem.getDFdx(k);
		}else
			f_qT.setZero();

		//f_la = -fem.getVolume(k)*fem.getDFdx(k).transpose()*stress_la;
	}

	inline void assembleVectorDOF(LinearFEM& fem, unsigned int k, unsigned int i, unsigned int idof, unsigned int lidof, unsigned int gidof){
		g_q.row(gidof) += f_qT.col(lidof);
	}

	inline void assembleMatrixDOF(LinearFEM& fem, unsigned int k, unsigned int i, unsigned int idof, unsigned int lidof, unsigned int gidof, unsigned int j, unsigned int jdof, unsigned int ljdof, unsigned int gjdof){}
	inline void finalize(LinearFEM& fem){}
};

void GlobalPrincipalStretchMaterialParameterHandler::getCurrentParams(  Eigen::VectorXd& q, LinearFEM& fem){ // writes q, does not change fem
	vtkSmartPointer<PrincipalStretchMaterial> psm;
	if( fem.getMaterialModels().count(bodyId)==1 ){
		if( psm=PrincipalStretchMaterial::SafeDownCast( fem.getMaterialModels()[bodyId] ) ){
			q.resize(2*(psm->xp.size()));
			for(unsigned int i=0; i<psm->xp.size(); ++i){
				q(i)                = psm->fPrime->Evaluate(psm->xp(i));
				q(i+psm->xp.size()) = psm->hPrime->Evaluate(psm->xp(i));
			}
		}
	}
}
void GlobalPrincipalStretchMaterialParameterHandler::setNewParams(const Eigen::VectorXd& q, LinearFEM& fem){ // writes to fem, does not change q
	vtkSmartPointer<PrincipalStretchMaterial> psm;
	if( fem.getMaterialModels().count(bodyId)==1 ){
		if( psm=PrincipalStretchMaterial::SafeDownCast( fem.getMaterialModels()[bodyId] ) ){
			psm->fPrime->RemoveAllPoints();
			psm->hPrime->RemoveAllPoints();
			double fMin=q(0), hMin=q(psm->xp.size());
			for(unsigned int i=0; i<psm->xp.size(); ++i){
				psm->fPrime->AddPoint(psm->xp(i),q(i));
				psm->hPrime->AddPoint(psm->xp(i),q(i+psm->xp.size()));
				// the following helps in maintaining stability, but causes the optimizer to fail because it gets no feedback about the parameter changes made here
				//psm->fPrime->AddPoint(psm->xp(i),  fMin=std::max(fMin,q(i))  );
				//psm->hPrime->AddPoint(psm->xp(i),  hMin=std::max(hMin,q(i+psm->xp.size()))  );
			}

			// debug ...
			std::stringstream fname; fname << "z_psmDebug_" << debugTmp++ << ".m";
			std::ofstream out(fname.str());
			out << "xi_fp_hp = [" << endl;
			for(unsigned int i=0; i<psm->xp.size()-1; ++i) for(double xi=psm->xp[i]; xi<=psm->xp[i+1]; xi+=((psm->xp[i+1]-psm->xp[i])/7.0)){
				out << " " << xi << " " << psm->fPrime->Evaluate(xi) << " " << psm->hPrime->Evaluate(xi) << endl;
			}	out << "];" << endl;
			out.close();
		}
	}
}
unsigned int GlobalPrincipalStretchMaterialParameterHandler::getNumberOfParams(const LinearFEM& fem){ // how many parameters are we working with (rows in q)
	vtkSmartPointer<PrincipalStretchMaterial> psm;
	if( fem.getMaterialModels().count(bodyId)==1 ){
		if( psm=PrincipalStretchMaterial::SafeDownCast( fem.getMaterialModels()[bodyId] ) ){
			return 2*(psm->xp.size());
		}
	}
	return 0;
}
double GlobalPrincipalStretchMaterialParameterHandler::computeConstraintDerivatives(Eigen::MatrixXd& g_q, Eigen::VectorXd& phiQ_q, LinearFEM& fem){
	PrincipalStretchForceDerivativeAssemblyOp dfop(g_q,phiQ_q, *this,fem);
	LinearFEM::assemblyLoop(fem, dfop);
	//return dfop.phiQ;

	// add global regularizer to enforce monotonicity of fPrime and hPrime ... penalize negative t-derivatives?
	if( phiQ_q.size() != getNumberOfParams(fem) ){
		phiQ_q.resize( getNumberOfParams(fem) );
		phiQ_q.setZero();
	}
	double splinePhiQ =0.0; Eigen::VectorXd splinePhiQ_q;
	if( fem.getMaterialModels().count(bodyId)==1 ){
		vtkSmartPointer<PrincipalStretchMaterial> psm;
		if( psm=PrincipalStretchMaterial::SafeDownCast( fem.getMaterialModels()[bodyId] ) ){
			splinePhiQ += alpha * psm->fPrime->firstDerivRegularizer( splinePhiQ_q );
			phiQ_q.block(0,0, psm->xp.size(),1 ) += alpha * splinePhiQ_q;
			splinePhiQ += alpha * psm->hPrime->firstDerivRegularizer( splinePhiQ_q );
			phiQ_q.block(psm->xp.size(),0, psm->xp.size(),1 ) += alpha * splinePhiQ_q;
		}
	}	//printf("(psmPhiQ %.2lg) ", splinePhiQ);
	return (dfop.phiQ + splinePhiQ);
}


