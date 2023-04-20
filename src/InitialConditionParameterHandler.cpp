
#include "InitialConditionParameterHandler.h"
#include "LinearFEM.h"

using namespace MyFEM;

void InitialConditionParameterHandler::applyInitialConditions( LinearFEM& fem ){
	if( fem.simTime==0.0 && checkParamSize() ){
		Eigen::Vector3d com;
		if( setPostion || setOrientation || setAngularVelocity ) fem.computeBodyVolumeAndCentreOfMass(com,bodyId);

		for(unsigned int i=0; i<fem.getNumberOfNodes(); ++i){ //ToDo: only set nodes in matched body-ID ...
			if( setOrientation ){
				Eigen::Vector3d t = (fem.getRestCoord(i)-com), r = currentQ.segment<3>( getOrientationIndex() );
				Eigen::Matrix3d R;
				if( r.squaredNorm()<1e-10 ){ // small angle -- linear rotation ==> assume sin(th)~~th, cos(th)~~1
					R << 1.0, -r(2), r(1), r(2), 1.0, -r(0), -r(1), r(0), 1.0;
				}else{ // full rotation
					double th = r.norm(); r /= th;
					R << 0.0, -r(2), r(1), r(2), 0.0, -r(0), -r(1), r(0), 0.0;
					R = (Eigen::Matrix3d::Identity() + sin(th)*R + (1.0-cos(th))*R*R).eval();
				}
				fem.getDeformedCoord(i) = R*t+com;
			}else{ //no change in orientation - reset to rest shape (intentionally overruling fem reset pose)
				fem.getDeformedCoord(i) = fem.getRestCoord(i);
			}

			if( setPostion ) fem.getDeformedCoord(i) += currentQ.segment<3>( getPositionIndex() )-com;

			if( setVelocity ) fem.getVelocity(i) = currentQ.segment<3>( getVelocityIndex() );

			if( setAngularVelocity ){
				Eigen::Vector3d t = (fem.getRestCoord(i)-com);
				Eigen::Matrix3d xProd; xProd << 0.0, -t(2), t(1), t(2), 0.0, -t(0), -t(1), t(0), 0.0; // cross-product (t x . ) matrix representation
				fem.getVelocity(i) += angularVelocityScale*xProd*currentQ.segment<3>( getAngularVelocityIndex() );
			}

		}
	}
}

void InitialConditionParameterHandler::computeInitialDerivatives( Eigen::MatrixXd& dx_dq, Eigen::MatrixXd& dv_dq, LinearFEM& fem ){
	//Note: we assume dx_dq and dv_dq have been zeroed before, we're only writing the non-zero parts here
	if( fem.simTime==0.0 && checkParamSize() ){
		Eigen::Vector3d com;
		if( (setOrientation && orientationGradients) || (setAngularVelocity && angularVelocityGradients) ) fem.computeBodyVolumeAndCentreOfMass(com,bodyId);

		for(unsigned int i=0; i<fem.getNumberOfNodes(); ++i){ //ToDo: only set nodes in matched body-ID ...
			if( setPostion ){
				if( positionGradients ){
					dx_dq.block<3,3>(fem.getNodalDof(i,fem.X_DOF),getPositionIndex()).setIdentity();
				}
			}

			if( setOrientation ){
				if( orientationGradients ){
					//ToDo: implement orientation derivatives here ...
					Eigen::Vector3d t = (fem.getRestCoord(i)-com), r = currentQ.segment<3>( getOrientationIndex() );
					Eigen::Block<Eigen::MatrixXd,3,3> dR ( dx_dq, fem.getNodalDof(i,fem.X_DOF), getOrientationIndex() );
					if( r.squaredNorm()<1e-10 ){ // small angle -- linear rotation ==> assume sin(th)~~th, cos(th)~~1
						//[     0,  t2 ,  -t1 ;  -t2 ,   0,  t0 ;  t1 ,  -t0 ,   0]
						dR << 0.0, t(2), -t(1), -t(2), 0.0, t(0), t(1), -t(0), 0.0;
					}else{
						// auto-generated code, reads r and t, writes dR
						#include "codegen_rotationDerivs.h"
					}
				}
			}

			if( setVelocity ){
				if( velocityGradients ){
					dv_dq.block<3,3>(fem.getNodalDof(i,fem.X_DOF),getVelocityIndex()).setIdentity();
				}
			}

			if( setAngularVelocity ){
				if( angularVelocityGradients ){
					Eigen::Vector3d t = (fem.getRestCoord(i)-com);
					Eigen::Matrix3d xProd; xProd << 0.0, -t(2), t(1), t(2), 0.0, -t(0), -t(1), t(0), 0.0; // cross-product (t x . ) matrix representation
					dv_dq.block<3,3>(fem.getNodalDof(i,fem.X_DOF),getAngularVelocityIndex()) = angularVelocityScale*xProd;
				}
			}

		}
	}
}

