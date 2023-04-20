#include "ElasticMaterialParameterHandler.h"
#include "Materials.h"
#include "LinearFEM.h"

#include <vtkDoubleArray.h>
#include <vtkUnstructuredGrid.h>
#include <vtkCellData.h>

using namespace MyFEM;

// we need material models to be able to compute d(stress)_d(lameLambda) and d(stress)_d(lameMu)
// -- for now, we keep this functionality separate from the basic material implementation for modularity
class ElasticForceDerivativeAssemblyOp{
public:
	double phiQ;
	ParameterHandler& qHdl;
	Eigen::MatrixXd& g_q; Eigen::VectorXd& phiQ_q;
	Eigen::Matrix3d F;
	Eigen::VectorXd f_la, f_mu;    // derivatives of force (elem-wise) wrt. params
	Vector9d stress_la, stress_mu; // derivatives of stress wrt. Lame parameters
	ElasticForceDerivativeAssemblyOp(Eigen::MatrixXd& g_q_, Eigen::VectorXd& phiQ_q_, ParameterHandler& qHdl_, LinearFEM& fem) : g_q(g_q_), phiQ_q(phiQ_q_), qHdl(qHdl_) {}

	inline void initialize(LinearFEM& fem){
		phiQ=0.0;
		g_q.resize( qHdl.getNumberOfDOFs(fem), qHdl.getNumberOfParams(fem) );
		g_q.setZero();
		stress_la.setZero(); stress_mu.setZero();
	}

	inline void calculateElement(LinearFEM& fem, unsigned int k){
		fem.computeDeformationGradient(F, k);
		matrixComponentRefs(F);

		if( HomogeneousIsotropicNeohookeanMaterial::SafeDownCast( fem.getMaterialModel(k) ) ){
			// Neohookean stress derivatives ... auto-generated, reads F-components, writes stress_la and stress_mu
			#include "codegen_Neohookean_derivs.h"
		}else // note that Neohookean class is derived from linear elastic class, so we need to check in this order!
		if( HomogeneousIsotropicLinearElasticMaterial::SafeDownCast( fem.getMaterialModel(k) ) ){
			// Neohookean stress derivatives ... auto-generated, reads F-components, writes stress_la and stress_mu
			#include "codegen_linElast_derivs.h"
		}

		f_la = -fem.getVolume(k)*fem.getDFdx(k).transpose()*stress_la;
		f_mu = -fem.getVolume(k)*fem.getDFdx(k).transpose()*stress_mu;
	}

	inline void assembleVectorDOF(LinearFEM& fem, unsigned int k, unsigned int i, unsigned int idof, unsigned int lidof, unsigned int gidof){
		g_q(gidof,0) += f_la(lidof);
		g_q(gidof,1) += f_mu(lidof);
	}

	inline void assembleMatrixDOF(LinearFEM& fem, unsigned int k, unsigned int i, unsigned int idof, unsigned int lidof, unsigned int gidof, unsigned int j, unsigned int jdof, unsigned int ljdof, unsigned int gjdof){}
	inline void finalize(LinearFEM& fem){}
};

// basic parameter handler: q are material parameters, all elements are set to the same parameters
// we only handle a single material and only the elastic Lame parameters here
unsigned int GlobalElasticMaterialParameterHandler::getNumberOfParams(const LinearFEM& fem){
	return 2;
}
void GlobalElasticMaterialParameterHandler::getCurrentParams(Eigen::VectorXd& q, LinearFEM& fem){
	double la, mu;
	if(!initialized){
		for(unsigned int k=0; k<fem.getNumberOfElems(); ++k) if(fem.getBodyId(k)==bodyId){
			firstElemInActiveBody = k;
			initialized=true;
		}
	}
	fem.getMaterialModels()[bodyId]->getElasticParams(
		fem.getMaterialParameters(firstElemInActiveBody).data(),
		la, mu
	);
	q[0]=la; q[1]=mu;
}
void GlobalElasticMaterialParameterHandler::setNewParams(const Eigen::VectorXd& q, LinearFEM& fem){
	for(unsigned int k=0; k<fem.getNumberOfElems(); ++k) if(fem.getBodyId(k)==bodyId){
		fem.getMaterialModel(k)->setElasticParams(
			fem.getMaterialParameters(k).data(),
			q[0], q[1]
		);
		if(!initialized){
			firstElemInActiveBody = k;
			initialized=true;
		}
	}
}
double GlobalElasticMaterialParameterHandler::computeConstraintDerivatives(Eigen::MatrixXd& g_q, Eigen::VectorXd& phiQ_q, LinearFEM& fem){
	ElasticForceDerivativeAssemblyOp dfop(g_q,phiQ_q, *this,fem);
	LinearFEM::assemblyLoop(fem, dfop);
	return dfop.phiQ;
}



class PerElementElasticForceDerivativeAssemblyOp : public ElasticForceDerivativeAssemblyOp{
public:
	Eigen::MatrixXd unused;
	SparseMatrixD& g_q;
	std::vector<Eigen::Triplet<double> > g_q_triplets;
	PerElementElasticForceDerivativeAssemblyOp(SparseMatrixD& g_q_, Eigen::VectorXd& phiQ_q, ParameterHandler& qHdl, LinearFEM& fem) : ElasticForceDerivativeAssemblyOp(unused,phiQ_q,qHdl,fem), g_q(g_q_) {}
	inline void initialize(LinearFEM& fem){
		phiQ_q.resize( qHdl.getNumberOfParams(fem) );
		phiQ_q.setZero();
		phiQ=0.0;
		stress_la.setZero(); stress_mu.setZero();
		g_q_triplets.clear();
		g_q.resize( qHdl.getNumberOfDOFs(fem), qHdl.getNumberOfParams(fem) );
	}
	inline void calculateElement(LinearFEM& fem, unsigned int k){
		double la,mu;
		fem.getMaterialModel(k)->getElasticParams(
			fem.getMaterialParameters(k).data(), la,mu
		);
		double unused=0.0; // we don't store second derivs of regularizer here, because per-element params won't be feasibly optimized by direct sensitivity analysis and Gauss-Newton solvers anyway
		phiQ+=ParameterHandler::regFcn(la, phiQ_q(2*k)   ,unused);
		phiQ+=ParameterHandler::regFcn(mu, phiQ_q(2*k+1) ,unused);

		ElasticForceDerivativeAssemblyOp::calculateElement(fem,k);
	}
	inline void assembleVectorDOF(LinearFEM& fem, unsigned int k, unsigned int i, unsigned int idof, unsigned int lidof, unsigned int gidof){
		//g_q(gidof,2*k  ) += f_la(lidof);
		//g_q(gidof,2*k+1) += f_mu(lidof);
		g_q_triplets.push_back(Eigen::Triplet<double>( gidof,2*k,   f_la(lidof)  ));
		g_q_triplets.push_back(Eigen::Triplet<double>( gidof,2*k+1, f_mu(lidof)  ));
	}
	inline void finalize(LinearFEM& fem){
		g_q.setFromTriplets(g_q_triplets.begin(),g_q_triplets.end());
	}
};
unsigned int PerElementElasticMaterialParameterHandler::getNumberOfParams(const LinearFEM& fem){
	//ToDo: properly support multi-body sims (filter by bodyId)
	return 2 * fem.getNumberOfElems(); // Lame lambda and mu per element
}	// from now on we'll assume q has been resized to this number of coefficients already
void PerElementElasticMaterialParameterHandler::getCurrentParams(  Eigen::VectorXd& q, LinearFEM& fem){
	double la, mu;
	for(unsigned int k=0; k<fem.getNumberOfElems(); ++k){
		fem.getMaterialModel(k)->getElasticParams(
			fem.getMaterialParameters(k).data(), q[2*k], q[2*k+1]
		);
	}
}
void PerElementElasticMaterialParameterHandler::setNewParams(const Eigen::VectorXd& q, LinearFEM& fem){
	// add to vtk structure for output ...
	vtkSmartPointer<vtkDoubleArray> vtkLambda;
	vtkLambda = vtkDoubleArray::SafeDownCast( fem.mesh->GetCellData()->GetAbstractArray("lambda") );
	if( vtkLambda==NULL ){ // if the array does not exist yet, create it
		vtkLambda = vtkSmartPointer<vtkDoubleArray>::New();
		vtkLambda->SetName("lambda");
		vtkLambda->SetNumberOfComponents(1);
		vtkLambda->SetNumberOfTuples(fem.getNumberOfElems());
		fem.mesh->GetCellData()->AddArray(vtkLambda);
	}
	vtkSmartPointer<vtkDoubleArray> vtkMu;
	vtkMu = vtkDoubleArray::SafeDownCast( fem.mesh->GetCellData()->GetAbstractArray("mu") );
	if( vtkMu==NULL ){
		vtkMu = vtkSmartPointer<vtkDoubleArray>::New();
		vtkMu->SetName("mu");
		vtkMu->SetNumberOfComponents(1);
		vtkMu->SetNumberOfTuples(fem.getNumberOfElems());
		fem.mesh->GetCellData()->AddArray(vtkMu);
	}

	for(unsigned int k=0; k<fem.getNumberOfElems(); ++k){
		fem.getMaterialModel(k)->setElasticParams(
			fem.getMaterialParameters(k).data(), q[2*k], q[2*k+1]
		);
		vtkLambda->SetTuple1(k,q[2*k  ]);
		vtkMu->    SetTuple1(k,q[2*k+1]);
	}
}
double PerElementElasticMaterialParameterHandler::computeConstraintDerivatives(SparseMatrixD& g_q, Eigen::VectorXd& phiQ_q, LinearFEM& fem){
	PerElementElasticForceDerivativeAssemblyOp dfop(g_q,phiQ_q, *this,fem);
	LinearFEM::assemblyLoop(fem, dfop);
	return dfop.phiQ;
}



class ConsistentInertiaDerivativeAssemblyOp{ // compute dfext/drho - dM/drho*a
public:
	double elemMassDeriv;
	Eigen::MatrixXd& dMdqa;
	Eigen::Vector3d x0,xc,gx;
	ConsistentInertiaDerivativeAssemblyOp(Eigen::MatrixXd& g_q) : dMdqa(g_q) {}
	inline void initialize(LinearFEM& fem){
		dMdqa.resize(fem.v.size(),1); dMdqa.setZero();
	}
	inline void calculateElement(LinearFEM& fem, unsigned int k){
		Tetra ek = fem.getElement(k);
		if( fem.fExtFields.count( fem.getBodyId(k) ) ){
			x0 = 1.0/4.0*(fem.getRestCoord(ek(0)) + fem.getRestCoord(ek(1)) + fem.getRestCoord(ek(2)) + fem.getRestCoord(ek(3)));
			xc = 1.0/4.0*(fem.getDeformedCoord(ek(0)) + fem.getDeformedCoord(ek(1)) + fem.getDeformedCoord(ek(2)) + fem.getDeformedCoord(ek(3)));
			fem.fExtFields[fem.getBodyId(k)]->eval(gx,x0,xc, fem.simTime); // evaluate external acceleration at element centroid
		}else{
			gx.setZero();
		}
		elemMassDeriv = fem.getVolume(k); // elem mass = elem volume * density
	}
	inline void assembleVectorDOF(LinearFEM& fem, unsigned int k, unsigned int i, unsigned int idof, unsigned int lidof, unsigned int gidof){
		dMdqa(gidof) += 1.0/4.0* elemMassDeriv * gx(idof); //ToDo: support non-const external acceleration --> use quadrature over element (along with LinearFEM impl)
	}
	inline void assembleMatrixDOF(LinearFEM& fem, unsigned int k, unsigned int i, unsigned int idof, unsigned int lidof, unsigned int gidof, unsigned int j, unsigned int jdof, unsigned int ljdof, unsigned int gjdof){
		if( fem.currentAcceleration.size()==fem.v.size() ){ // compute dM/dq * fem.currentAcceleration - only available if we run a dynamic sim
			if( idof==jdof ) //M_triplets.push_back( Eigen::Triplet<double>(gidof,gjdof, elemMass* 1.0/((i==j)?10.0:20.0) ));
				dMdqa(gjdof) -= (elemMassDeriv* 1.0/((i==j)?10.0:20.0)) * fem.currentAcceleration(gidof);
		}
	}
	inline void finalize(LinearFEM& fem){}
};
void GlobalDensityParameterHandler::getCurrentParams(  Eigen::VectorXd& q, LinearFEM& fem){
	if(!initialized){
		for(unsigned int k=0; k<fem.getNumberOfElems(); ++k) if(fem.getBodyId(k)==bodyId){
			firstElemInActiveBody = k;
			initialized=true;
		}
	}
	q[0] = fem.getMaterialModels()[bodyId]->getDensity( fem.getMaterialParameters(firstElemInActiveBody).data() );
}
void GlobalDensityParameterHandler::setNewParams(const Eigen::VectorXd& q, LinearFEM& fem){
	for(unsigned int k=0; k<fem.getNumberOfElems(); ++k) if(fem.getBodyId(k)==bodyId){
		fem.getMaterialModel(k)->setDensity( fem.getMaterialParameters(k).data(), q[0] );
		if(!initialized){
			firstElemInActiveBody = k;
			initialized=true;
		}
	}
}
unsigned int GlobalDensityParameterHandler::getNumberOfParams(const LinearFEM& fem){ return 1; }
double GlobalDensityParameterHandler::computeConstraintDerivatives(Eigen::MatrixXd& g_q, Eigen::VectorXd& phiQ_q, LinearFEM& fem){
	ConsistentInertiaDerivativeAssemblyOp dfop(g_q);
	LinearFEM::assemblyLoop(fem, dfop);
	return 0.0; // no local regularizer
}


class ConsistentPerElementInertiaDerivativeAssemblyOp : public ConsistentInertiaDerivativeAssemblyOp{
public:
	PerElementDensityMaterialParameterHandler& qHdl;
	SparseMatrixD& g_q;
	double phiQ = 0.0, referenceMass; // use negative referenceMass to disable regularizer
	Eigen::VectorXd& phiQ_q; // regularize to total reference mass
	std::vector<Eigen::Triplet<double> > g_q_triplets;
	ConsistentPerElementInertiaDerivativeAssemblyOp(SparseMatrixD& g_q_, Eigen::VectorXd& phiQ_q_, PerElementDensityMaterialParameterHandler& qHdl_, double referenceMass_=-1.0) : ConsistentInertiaDerivativeAssemblyOp(Eigen::MatrixXd()), g_q(g_q_), qHdl(qHdl_), phiQ_q(phiQ_q_) {
		referenceMass = referenceMass_;
	}
	inline void initialize(LinearFEM& fem){
		phiQ = 0.0;
		if( referenceMass>0.0 ){
			phiQ_q.resize(qHdl.getNumberOfParams(fem)); phiQ_q.setZero();
		}
	}
	inline void calculateElement(LinearFEM& fem, unsigned int k){
		ConsistentInertiaDerivativeAssemblyOp::calculateElement(fem,k);
		if( referenceMass>0.0 ){
			phiQ += fem.getVolume(k) * fem.getMaterialModel(k)->getDensity( fem.getMaterialParameters(k).data() ); // sum up current mass in phiQ
			phiQ_q(k) += fem.getVolume(k); // partial derivative is element volume here
		}
	}
	inline void assembleVectorDOF(LinearFEM& fem, unsigned int k, unsigned int i, unsigned int idof, unsigned int lidof, unsigned int gidof){
		//g_q(gidof,k) += 1.0/4.0* elemMassDeriv * gx(idof); //ToDo: support non-const external acceleration --> use quadrature over element (along with LinearFEM impl)
		g_q_triplets.push_back( Eigen::Triplet<double>(
			gidof, k, 1.0/4.0* elemMassDeriv * gx(idof)
		) );
	}
	inline void assembleMatrixDOF(LinearFEM& fem, unsigned int k, unsigned int i, unsigned int idof, unsigned int lidof, unsigned int gidof, unsigned int j, unsigned int jdof, unsigned int ljdof, unsigned int gjdof){
		if( fem.currentAcceleration.size()==fem.v.size() ){ // compute dM/dq * fem.currentAcceleration - only available if we run a dynamic sim
			if( idof==jdof ) //M_triplets.push_back( Eigen::Triplet<double>(gidof,gjdof, elemMass* 1.0/((i==j)?10.0:20.0) ));
				//g_q(gjdof,k) -= (elemMassDeriv* 1.0/((i==j)?10.0:20.0)) * fem.currentAcceleration(gidof);
				g_q_triplets.push_back( Eigen::Triplet<double>(
					gjdof, k, -(elemMassDeriv* 1.0/((i==j)?10.0:20.0)) * fem.currentAcceleration(gidof)
				) );
		}
	}
	inline void finalize(LinearFEM& fem){
		g_q.resize( qHdl.getNumberOfDOFs(fem), qHdl.getNumberOfParams(fem) );
		g_q.setFromTriplets(g_q_triplets.begin(),g_q_triplets.end());

		if( referenceMass>0.0 ){
			phiQ = 0.5*(referenceMass-phiQ)*(referenceMass-phiQ); // phiQ stored current mass here - now compute squared difference to reference mass as regularizer value
			phiQ_q *= (referenceMass-phiQ); // update partial derivatives (contains element volumes so far)
		}
	}
};
void PerElementDensityMaterialParameterHandler::getCurrentParams(  Eigen::VectorXd& q, LinearFEM& fem){
	for(unsigned int k=0; k<fem.getNumberOfElems(); ++k){ //if(fem.getBodyId(k)==bodyId)
		q[k] = fem.getMaterialModel(k)->getDensity( fem.getMaterialParameters(k).data() );
	}
}
void PerElementDensityMaterialParameterHandler::setNewParams(const Eigen::VectorXd& q, LinearFEM& fem){
	bool computeTotalMassNow=false;
	if( referenceMass < 0.0 && useReferenceMassRegularizer ){
		computeTotalMassNow = true;
		referenceMass = 0.0;
	} 

	// add to vtk structure for output ...
	vtkSmartPointer<vtkDoubleArray> vtkRho;
	vtkRho = vtkDoubleArray::SafeDownCast( fem.mesh->GetCellData()->GetAbstractArray("rho") );
	if( vtkRho==NULL ){ // if the array does not exist yet, create it
		vtkRho = vtkSmartPointer<vtkDoubleArray>::New();
		vtkRho->SetName("rho");
		vtkRho->SetNumberOfComponents(1);
		vtkRho->SetNumberOfTuples(fem.getNumberOfElems());
		fem.mesh->GetCellData()->AddArray(vtkRho);
	}

	for(unsigned int k=0; k<fem.getNumberOfElems(); ++k){
		if( computeTotalMassNow ) referenceMass += fem.getVolume(k) * fem.getMaterialModel(k)->getDensity( fem.getMaterialParameters(k).data() );
		fem.getMaterialModel(k)->setDensity( fem.getMaterialParameters(k).data(), q[k] );
		vtkRho->SetTuple1(k,q[k]);
	}
}
unsigned int PerElementDensityMaterialParameterHandler::getNumberOfParams(const LinearFEM& fem){
	//ToDo: properly support multi-body sims (filter by bodyId)
	return fem.getNumberOfElems(); 
}
double PerElementDensityMaterialParameterHandler::computeConstraintDerivatives(SparseMatrixD& g_q, Eigen::VectorXd& phiQ_q, LinearFEM& fem){
	ConsistentPerElementInertiaDerivativeAssemblyOp dfop(g_q, phiQ_q, *this, referenceMass);
	LinearFEM::assemblyLoop(fem, dfop);
	phiQ_q *= regularizerWeight;
	return    regularizerWeight * dfop.phiQ;
}


void CombinedParameterHandler::getCurrentParams(  Eigen::VectorXd& q, LinearFEM& fem){
	Eigen::VectorXd q1,q2;
	unsigned int
		np1 = qh1.getNumberOfParams(fem),
		np2 = qh2.getNumberOfParams(fem);
	q1.resize( np1 );
	q2.resize( np2 );
	qh1.getCurrentParams(q1,fem);
	qh2.getCurrentParams(q2,fem);
	//q.block( 0 ,0,np1,1) = q1;
	//q.block(np1,0,np2,1) = q2;
	if( useLogOfParams ){
		q.block( 0 ,0,np1,1) = q1.array().log().matrix();
		q.block(np1,0,np2,1) = q2.array().log().matrix();
	}else{
		q.block( 0 ,0,np1,1) = q1;
		q.block(np1,0,np2,1) = q2;
	}
}
void CombinedParameterHandler::setNewParams(const Eigen::VectorXd& q, LinearFEM& fem){
	Eigen::VectorXd q1,q2;
	unsigned int
		np1 = qh1.getNumberOfParams(fem),
		np2 = qh2.getNumberOfParams(fem);
	q1.resize( np1 );
	q2.resize( np2 );
	q1 = q.block( 0 ,0, np1,1);
	q2 = q.block(np1,0, np2,1);

	if( useLogOfParams ){
		qh1.setNewParams(q1.array().exp().matrix(),fem);
		qh2.setNewParams(q2.array().exp().matrix(),fem);
	}else{
		qh1.setNewParams(q1,fem);
		qh2.setNewParams(q2,fem);
	}
}
unsigned int CombinedParameterHandler::getNumberOfParams(const LinearFEM& fem){
	return qh1.getNumberOfParams(fem) + qh2.getNumberOfParams(fem);
}
double CombinedParameterHandler::computeConstraintDerivatives(Eigen::MatrixXd& g_q, Eigen::VectorXd& phiQ_q, LinearFEM& fem){
	unsigned int
		np1 = qh1.getNumberOfParams(fem),
		np2 = qh2.getNumberOfParams(fem);
	double phiQ1,phiQ2;
	Eigen::MatrixXd g_q1, g_q2;
	Eigen::VectorXd phiQ_q1, phiQ_q2;

	phiQ1 = qh1.computeConstraintDerivatives(g_q1,phiQ_q1,fem);
	phiQ2 = qh2.computeConstraintDerivatives(g_q2,phiQ_q2,fem);

	g_q.resize( getNumberOfDOFs(fem), getNumberOfParams(fem) );
	if( np1>0 ) g_q.block(0, 0 , getNumberOfDOFs(fem),np1) = g_q1;
	if( np2>0 ) g_q.block(0,np1, getNumberOfDOFs(fem),np2) = g_q2;

	if( phiQ_q1.size()>0 || phiQ_q2.size()>0 ){
		phiQ_q.resize( getNumberOfParams(fem) );
		phiQ_q.setZero(); // phiQ_q need not be assigned by parameter handlers check if we have any ...
		if( phiQ_q1.size()>0 ) phiQ_q.block( 0 ,0,np1,1) = phiQ_q1;
		if( phiQ_q2.size()>0 ) phiQ_q.block(np1,0,np2,1) = phiQ_q2;
	}

	if( qh1.phiQ_qq.size()>0 || qh2.phiQ_qq.size()>0 ){
		phiQ_qq.resize( getNumberOfParams(fem) );
		phiQ_qq.setZero(); // phiQ_qq need not be assigned by parameter handlers check if we have any ...
		if( qh1.phiQ_qq.size()>0 ) phiQ_qq.block( 0 ,0,np1,1) = qh1.phiQ_qq;
		if( qh2.phiQ_qq.size()>0 ) phiQ_qq.block(np1,0,np2,1) = qh2.phiQ_qq;
	}

	if( useLogOfParams ){
		// for log-based parameters: gradient wrt. log(param) is (original gradient) * (original param value)
		Eigen::VectorXd q1(qh1.getNumberOfParams(fem)),q2(qh2.getNumberOfParams(fem)),q(getNumberOfParams(fem));
		qh1.getCurrentParams(q1,fem); qh2.getCurrentParams(q2,fem); q.block( 0 ,0,np1,1) = q1; q.block(np1,0,np2,1) = q2;
		if( phiQ_q.size()>0 ) phiQ_q = phiQ_q.cwiseProduct( q ).eval();
		for(unsigned int i=0; i<getNumberOfParams(fem); ++i) g_q.col(i) *= q(i);

		//if we use dense parameter handlers, either of them may also compute second derivatives of regularizers phiQ_qq, which also need to be scaled properly
		if( phiQ_qq.size() > 0 ){
			phiQ_qq = phiQ_qq.cwiseProduct( q.array().square().matrix() ).eval(); // first term of product rule: multiply by q twice
			phiQ_qq += phiQ_q; // second term of product rule: add first derivative (already multipled by q, as d2q/dq*2 = dq/dq* = exp(q*) = q)
		}

	}


	return phiQ1 + phiQ2;
}
double CombinedParameterHandler::computeConstraintDerivatives(SparseMatrixD& g_q, Eigen::VectorXd& phiQ_q, LinearFEM& fem){
	unsigned int
		np1 = qh1.getNumberOfParams(fem),
		np2 = qh2.getNumberOfParams(fem);
	double phiQ1,phiQ2;
	SparseMatrixD g_q1, g_q2;
	Eigen::MatrixXd tmp;
	Eigen::VectorXd phiQ_q1, phiQ_q2;

	if( qh1.useSparseDgDq() ){
		phiQ1 = qh1.computeConstraintDerivatives(g_q1,phiQ_q1,fem);
	}else{
		phiQ1 = qh1.computeConstraintDerivatives(tmp,phiQ_q1,fem);
		g_q1 = tmp.sparseView();
	}
	if( qh2.useSparseDgDq() ){
		phiQ2 = qh2.computeConstraintDerivatives(g_q2,phiQ_q2,fem);
	}else{
		phiQ2 = qh2.computeConstraintDerivatives(tmp,phiQ_q2,fem);
		g_q2 = tmp.sparseView();
	}

	g_q.resize( getNumberOfDOFs(fem), getNumberOfParams(fem) );
	//g_q.block(0, 0 , getNumberOfDOFs(fem),np1) = g_q1;
	//g_q.block(0,np1, getNumberOfDOFs(fem),np2) = g_q2;
	g_q.reserve(g_q1.nonZeros() + g_q2.nonZeros());
	for(Eigen::Index i=0; i<g_q1.outerSize(); ++i){
		g_q.startVec(i);
		if( g_q1.size()>0) for(SparseMatrixD::InnerIterator it(g_q1, i); it; ++it) g_q.insertBack(it.row(), it.col()      ) = it.value();
		if( g_q2.size()>0) for(SparseMatrixD::InnerIterator it(g_q2, i); it; ++it) g_q.insertBack(it.row(), it.col() + np1) = it.value();
	}
	g_q.finalize();

	if( phiQ_q1.size()>0 || phiQ_q2.size()>0 ){
		phiQ_q.resize( getNumberOfParams(fem) );
		phiQ_q.setZero(); // phiQ_q need not be assigned by parameter handlers check if we have any ...
		if( phiQ_q1.size()>0 ) phiQ_q.block( 0 ,0,np1,1) = phiQ_q1;
		if( phiQ_q2.size()>0 ) phiQ_q.block(np1,0,np2,1) = phiQ_q2;
	}

	if( useLogOfParams ){
		// for log-based parameters: gradient wrt. log(param) is (original gradient) * (original param value) ^ -1
		Eigen::VectorXd q1(qh1.getNumberOfParams(fem)),q2(qh2.getNumberOfParams(fem)),q(getNumberOfParams(fem));
		qh1.getCurrentParams(q1,fem); qh2.getCurrentParams(q2,fem); q.block( 0 ,0,np1,1) = q1; q.block(np1,0,np2,1) = q2;
		if( phiQ_q.size()>0 ) phiQ_q.cwiseProduct( q );
		for(unsigned int i=0; i<getNumberOfParams(fem); ++i) g_q.col(i) *= q(i);
	}

	return phiQ1 + phiQ2;
}
void CombinedParameterHandler::applyInitialConditions( LinearFEM& fem ){
	InitialConditionParameterHandler* icQhdl = dynamic_cast<InitialConditionParameterHandler*>( &qh1 );
	if( icQhdl == NULL ) icQhdl = dynamic_cast<InitialConditionParameterHandler*>( &qh2 );
	if( icQhdl != NULL ){ 	//printf("\n%% CombinedParameterHandler::applyInitialConditions -- applying initial conditions ");
		icQhdl->applyInitialConditions(fem);
	}
}
void CombinedParameterHandler::computeInitialDerivatives( Eigen::MatrixXd& dx_dq, Eigen::MatrixXd& dv_dq, LinearFEM& fem ){
	InitialConditionParameterHandler* icQhdl = dynamic_cast<InitialConditionParameterHandler*>( &qh1 );
	if( icQhdl == NULL ) icQhdl = dynamic_cast<InitialConditionParameterHandler*>( &qh2 );
	if( icQhdl != NULL ){ 	//printf("\n%% CombinedParameterHandler::computeInitialDerivatives -- computing initial condition derivatives ");
		icQhdl->computeInitialDerivatives( dx_dq,dv_dq, fem );
	}
}
