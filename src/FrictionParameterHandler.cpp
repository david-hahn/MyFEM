
#include "FrictionParameterHandler.h"
#include "ContactFEM.h"

using namespace MyFEM;

void FrictionParameterHandler::getCurrentParams(  Eigen::VectorXd& q, LinearFEM& fem){
	if( dynamic_cast<ContactFEM*>(&fem)!=NULL ){
		ContactFEM& cFEM = *(dynamic_cast<ContactFEM*>(&fem));
		for(unsigned int k=0; k<cFEM.frictionCoefficient.size(); ++k){
			q[k] = cFEM.frictionCoefficient[k];
		}
	}
}

void FrictionParameterHandler::setNewParams(const Eigen::VectorXd& q, LinearFEM& fem){
	if( dynamic_cast<ContactFEM*>(&fem)!=NULL ){
		ContactFEM& cFEM = *(dynamic_cast<ContactFEM*>(&fem));
		for(unsigned int k=0; k<cFEM.frictionCoefficient.size(); ++k){
			cFEM.frictionCoefficient[k] = q[k];
		}
	}
}

unsigned int FrictionParameterHandler::getNumberOfParams(const LinearFEM& fem){
	if( dynamic_cast<const ContactFEM*>(&fem)!=NULL ){
		const ContactFEM& cFEM = *(dynamic_cast<const ContactFEM*>(&fem));
		return cFEM.frictionCoefficient.size();
	}
	return 0;
}

double FrictionParameterHandler::computeConstraintDerivatives(Eigen::MatrixXd& g_q, Eigen::VectorXd& phiQ_q, LinearFEM& fem){ // partial derivatives of the constraint function wrt. the design parameters + regularizer term phiQ(q) and dphiQ_dq
	if( dynamic_cast<ContactFEM*>(&fem)!=NULL ){
		g_q.resize( getNumberOfDOFs(fem), getNumberOfParams(fem) );
		g_q.setZero();

		ContactFEM& cFEM = *(dynamic_cast<ContactFEM*>(&fem));
		switch( cFEM.method ){
		case ContactFEM::CONTACT_TANH_PENALTY:
			return tanhPenaltyForceDerivatives(g_q,phiQ_q,cFEM);
		case ContactFEM::CONTACT_CLAMP_PENALTY:
		case ContactFEM::CONTACT_HYBRID:
			return linearPenaltyForceDerivatives(g_q,phiQ_q,cFEM); // both clamped and hybrid contacts use linear normal penalty forces and sliding friction based on them
		case ContactFEM::CONTACT_IGNORE:
		default:
			return 0.0;
		}
	}
	return 0.0;
}

double FrictionParameterHandler::linearPenaltyForceDerivatives(Eigen::MatrixXd& g_q, Eigen::VectorXd& phiQ_q, ContactFEM& fem){
	Eigen::Vector3d dFt_dCf; // local per-node derivative of tangential force wrt. friction coefficient

	for(unsigned int i=0; i<fem.getNumberOfNodes(); ++i){
		for(unsigned int oid=0; oid<fem.rigidObstacles.size(); ++oid) if( fem.contactStateByObstacle[oid][i]==ContactFEM::CONTACT_STATE::CONTACT_SLIP ){ // friction coefficient only affects forces of sliding nodes
			Eigen::Vector3d n;
			double g = fem.rigidObstacles[oid]->eval(n,fem.getRestCoord(i),fem.getDeformedCoord(i),fem.simTime);
			if( g<0.0 ){
				Eigen::Matrix3d N = n*n.transpose();
				Eigen::Matrix3d T = Eigen::Matrix3d::Identity() - N;
				Eigen::Vector3d t = timestep*T*fem.getVelocity(i);
				double fn = (-fem.normalPenaltyFactor*g);
				double tu = t.norm();// fMax = fem.frictionCoefficient[oid]*fn;
				t /= (tu+fem.epsD); // now t is the direction of tangential motion
				dFt_dCf = -fn*t;
				g_q.block<LinearFEM::N_DOFS,1>(fem.getNodalDof(i,0),oid) = dFt_dCf;
			}
		}
	}

	return 0.0;
}

double FrictionParameterHandler::tanhPenaltyForceDerivatives(Eigen::MatrixXd& g_q, Eigen::VectorXd& phiQ_q, ContactFEM& fem){
	std::vector<bool> done; done.assign(fem.getNumberOfNodes(), false); //ToDo: reduce allocated space to pre-counted number of boundary nodes
	Eigen::Vector3d n;
	Eigen::Vector3d dFt_dCf; // local per-node derivative of tangential force wrt. friction coefficient

	Eigen::Vector3d f_t, fd_f_t;

	for(unsigned int k = 0; k < fem.getNumberOfBndryElems(); ++k) for(int j = 0; j<3; ++j){ //ToDo: keep a list of boundary nodes to iterate directly?
		unsigned int i = fem.getBoundaryElement(k)(j);
		if(!done[i]){
			for(unsigned int oid=0; oid<fem.rigidObstacles.size(); ++oid){
				double g = fem.rigidObstacles[oid]->eval(n,fem.getRestCoord(i),fem.getDeformedCoord(i),fem.simTime);

				dFt_dCf.setZero();
				if( g<0.0 ){
					Eigen::Matrix3d N = n*n.transpose();
					Eigen::Matrix3d T = Eigen::Matrix3d::Identity() - N;

					//f_n = (-pF*g)*n; // normal penalty force
					//if( mode==UPDATE ) Kn =  pF*N;

					const double pf=fem.tangentialPenaltyFactor*timestep;
					Eigen::Vector3d t = T*fem.getVelocity(i);
					double fth, tu = t.norm(), fMax = (-fem.frictionCoefficient[oid] * fem.normalPenaltyFactor * g); // -pF*g >0 is the normal force magnitude -- (-cF*pF*g) is the Coulomb limit
					if( tu > DBL_EPSILON && fMax > DBL_EPSILON ){
						t /= tu; //n.normalize(); // now t is the direction of tangential motion

						fth = std::tanh(pf/fMax*tu);
						dFt_dCf = t*( g*fem.normalPenaltyFactor*fth + (pf*tu*(1.0-fth*fth))/(fem.frictionCoefficient[oid]) );
						//dFt_dCf = t*( -g*fem.normalPenaltyFactor*std::tanh((pf*tu)/((fem.frictionCoefficient[oid])*g*fem.normalPenaltyFactor)) - (pf*tu*(std::tanh((pf*tu)/((fem.frictionCoefficient[oid])*g*fem.normalPenaltyFactor))*std::tanh((pf*tu)/((fem.frictionCoefficient[oid])*g*fem.normalPenaltyFactor)) - 1))/(fem.frictionCoefficient[oid]) );

						//f_t = ( -fem.frictionCoefficient[oid]*fem.normalPenaltyFactor*g*tanh((pf*tu)/(fem.frictionCoefficient[oid]*fem.normalPenaltyFactor*g)) )*t;//f_t = (-fMax*fth)*t;
						//double fdH=1e-6;
						//fMax = (-(fem.frictionCoefficient[oid]+fdH) * fem.normalPenaltyFactor * g);
						//fth = std::tanh(pf/fMax*tu);
						//fd_f_t = ( -(fem.frictionCoefficient[oid]+fdH)*fem.normalPenaltyFactor*g*tanh((pf*tu)/((fem.frictionCoefficient[oid]+fdH)*fem.normalPenaltyFactor*g)) )*t;//fd_f_t = (-fMax*fth)*t;
						//fd_f_t -= f_t;
						//fd_f_t /= fdH;
						//cout << endl << "### FD = " << fd_f_t.transpose() << " ### DF = " << dFt_dCf.transpose() << " ### err = " << (fd_f_t-dFt_dCf).transpose() << endl;
						//cout << endl << "(((  fMax " << fMax << " ; fth " << fth << " ; tu " << tu << "   )))" << endl;
					}
				}
				//for(int lidof=0; lidof<LinearFEM::N_DOFS; ++lidof){
				//	unsigned int gidof = fem.getNodalDof(i,lidof); // global dof index
				//	g_q( gidof, oid ) += dFt_dCf(lidof);
				//}
				g_q.block<LinearFEM::N_DOFS,1>(fem.getNodalDof(i,0),oid) += dFt_dCf;
			}

			done[i]=true;
		}
	}


	////debugging: FD-test ... strangely works for some but not all nodes -- why???
	//Eigen::MatrixXd fd_g_q( getNumberOfDOFs(fem), getNumberOfParams(fem) ); fd_g_q.setZero();
	//// unperturbed friction forces ...
	//Eigen::VectorXd femFstorage = fem.f;
	//fem.f.setZero();
	//fem.assembleTanhPenaltyForceAndStiffness(fem.getVelocities(),timestep,MyFEM::SKIP);
	//for(unsigned int k=0; k<getNumberOfParams(fem); ++k) fd_g_q.col(k) = fem.f.eval();
	////cout << endl << "(( ori f " << fem.f.transpose() << " ))";
	//// perturbed forces
	//double fdH=1e-5;
	//for(unsigned int k=0; k<getNumberOfParams(fem); ++k){
	//	double tmp=fem.frictionCoefficient[k];
	//	fem.frictionCoefficient[k] += fdH;
	//	fem.f.setZero();
	//	fem.assembleTanhPenaltyForceAndStiffness(fem.getVelocities(),timestep,MyFEM::SKIP);
	//	//cout << endl << "(( fdh f " << fem.f.transpose() << " ))";
	//	fd_g_q.col(k) -= fem.f.eval();
	//	//cout << endl << "(( diff " << fd_g_q.col(k).transpose() << " ))";
	//	fd_g_q.col(k) /= -fdH;
	//	//cout << endl << "(( fdg " << fd_g_q.col(k).transpose() << " ))";
	//	fem.frictionCoefficient[k] = tmp;
	//}
	//fem.f = femFstorage;
	//cout << endl << "******************************************************";
	//cout << endl << "FD_df_dcf = [" << fd_g_q.transpose() << " ]; " << endl;
	//cout << endl << "df_dcf = [" << g_q.transpose() << " ]; " << endl;
	//cout << endl << "FD_err = [" << (g_q-fd_g_q).transpose() << " ]; " << endl;
	//cout << "******************************************************" << endl;


	return 0.0;
}
