#include "BoundaryFieldObjectiveFunction.h"
#include "TemporalInterpolationField.h"
#include "LinearFEM.h"

using namespace MyFEM;

double BoundaryFieldObjectiveFunction::evaluate( LinearFEM& fem,
	Eigen::VectorXd& phi_x, Eigen::VectorXd& phi_v, Eigen::VectorXd& phi_f, Eigen::VectorXd& phi_q
){
	// iterate over all boundary elements
	// check if we have a target for that region
	// if so, evaluate the target displacement u*
	// sum up 0.5*A*(u-u*)^2 (use area weighting)
	// add A*(u-u*) to phi_x
	// finally, scale everything by sum(A)
	//ToDo: allow target force or position as well ...

	std::map<unsigned int, Eigen::VectorXd> debugData;

	double phiVal=0.0, totalArea=0.0;
	phi_x.resize( fem.getRestCoords().size() );
	phi_x.setZero();
	phi_xx.resize( fem.getRestCoords().size() );
	phi_xx.setZero();
	Eigen::Vector3d uStar, u;
	for(unsigned int k=0; k<fem.getNumberOfBndryElems(); ++k){
		if( targetFields.count(fem.getBoundaryId(k))>0 ){ // have a target for this one ...
			double areaOver3 = 1.0/3.0*fem.getArea(k);
			totalArea += 3.0*areaOver3;
			for(int i=0; i<3; ++i){
				targetFields[fem.getBoundaryId(k)]->eval(uStar, fem.getRestCoord( fem.getBoundaryElement(k)(i) ), fem.getDeformedCoord( fem.getBoundaryElement(k)(i) ), fem.simTime);

				u = fem.getDeformedCoord( fem.getBoundaryElement(k)(i) ) - fem.getRestCoord( fem.getBoundaryElement(k)(i) ) -uStar;

				for(int dof=0; dof<LinearFEM::N_DOFS; ++dof){
					unsigned int idof=fem.getNodalDof( fem.getBoundaryElement(k)(i) ,dof);

					phiVal +=  0.5*areaOver3*scale *u(dof)*u(dof);//   phi += 1/2*A*(x-x0-p)^2
					phi_x[idof] += areaOver3*scale *u(dof);       // d(phi)/dx += A*(x-x0-p) (= 1/2*A*1*(x-x0-p)+1/2*A*(x-x0-p)*1 )
					phi_xx[idof]+= areaOver3*scale;
				}

				if(!debugFileName.empty()){
					debugData[fem.getBoundaryElement(k)(i)].resize(9);
					debugData[fem.getBoundaryElement(k)(i)].block<3,1>(0,0) = fem.getDeformedCoord( fem.getBoundaryElement(k)(i) );
					debugData[fem.getBoundaryElement(k)(i)].block<3,1>(3,0) = fem.getRestCoord(     fem.getBoundaryElement(k)(i) );
					debugData[fem.getBoundaryElement(k)(i)].block<3,1>(6,0) = uStar;
				}

			}
		}
	}

	if(!debugFileName.empty() && phiVal > DBL_EPSILON){
		std::string fname(debugFileName); fname.append( std::to_string(debugCount) ); fname.append(".txt");
		ofstream out(fname);
		if( out.good() ){
			for(std::map<unsigned int, Eigen::VectorXd>::iterator it=debugData.begin(); it!=debugData.end(); ++it){
				out << it->second.transpose() << endl;
			}
			out.close();
		}
		++debugCount;
	}

	phi_x /= totalArea;
	phi_xx /= totalArea;
	return phiVal / totalArea;
}

double AverageBoundaryValueObjectiveFunction::evaluate( LinearFEM& fem,
	Eigen::VectorXd& phi_x, Eigen::VectorXd& phi_v, Eigen::VectorXd& phi_f, Eigen::VectorXd& phi_q
){
	//here we measure the average deformed location (or displacement or velocity depending on targetMode) of each boundary region of interest

	double phiVal=0.0;
	std::map<unsigned int, double> regionArea;
	std::map<unsigned int, Eigen::Vector3d> regionRestAvg, regionDefdAvg, regionVelocityAvg; // regionVelocityAvg remains empty unless targetMode == TARGET_VELOCITY

	if( targetMode == TARGET_VELOCITY ){
		phi_v.resize( fem.getRestCoords().size() );
		phi_v.setZero();
		phi_vv.resize( fem.getRestCoords().size() );
		phi_vv.setZero();
	}//else{ // displacement or location mode ...
		phi_x.resize( fem.getRestCoords().size() );
		phi_x.setZero();
		phi_xx.resize( fem.getRestCoords().size() );
		phi_xx.setZero();
	//}
	Eigen::Vector3d uStar, u;

	// first compute averages of boundary regions of interest
	for(unsigned int k=0; k<fem.getNumberOfBndryElems(); ++k){
		unsigned int bnd_id = fem.getBoundaryId(k);
		if( targetFields.count(bnd_id)>0 ){ // have a target for this one ...
			double areaOver3 = 1.0/3.0*fem.getArea(k);
			if( regionArea.count(bnd_id)==0 ){
				regionArea[bnd_id]=0.0;
				regionRestAvg[bnd_id]=Eigen::Vector3d::Zero();
				regionDefdAvg[bnd_id]=Eigen::Vector3d::Zero();
				if( targetMode == TARGET_VELOCITY ) regionVelocityAvg[bnd_id]=Eigen::Vector3d::Zero();
			}
			regionArea[bnd_id] += 3.0*areaOver3;
			for(int i=0; i<3; ++i){ // for all nodes of triangle k
				regionRestAvg[bnd_id] += areaOver3*fem.getRestCoord( fem.getBoundaryElement(k)(i) );
				regionDefdAvg[bnd_id] += areaOver3*fem.getDeformedCoord( fem.getBoundaryElement(k)(i) );
				if( targetMode == TARGET_VELOCITY ) regionVelocityAvg[bnd_id] += areaOver3*fem.getVelocity( fem.getBoundaryElement(k)(i) );
			}
		}
	}
	// now compute the objective function
	for(std::map<unsigned int, double>::iterator it=regionArea.begin(); it!=regionArea.end(); ++it){
		unsigned int bnd_id = it->first;
		regionRestAvg[bnd_id] /= regionArea[bnd_id];
		regionDefdAvg[bnd_id] /= regionArea[bnd_id];
		if( targetMode == TARGET_VELOCITY ) regionVelocityAvg[bnd_id] /= regionArea[bnd_id];

		//debug output
		if( trackedLocations.count(bnd_id)==0){
			trackedLocations[bnd_id] = new TemporalInterpolationField;
			((TemporalInterpolationField*)trackedLocations[bnd_id])->addPoint(0.0,regionDefdAvg[bnd_id]); // add the initial point twice so we have enough data to build interpolating splines even if this is the only objective function evaluation
		}
		((TemporalInterpolationField*)trackedLocations[bnd_id])->addPoint(fem.simTime,regionDefdAvg[bnd_id]);

		// evaluate target field ...
		if( targetFields[bnd_id]->isDataAvailable(fem.simTime) ){
			targetFields[bnd_id]->eval(uStar, regionRestAvg[bnd_id], regionDefdAvg[bnd_id], fem.simTime);

			// overwrite regionDefdAvg with (uBar-uStar)
			if( targetMode == TARGET_LOCATION ){
				regionDefdAvg[bnd_id] -= uStar;
			}else
			if( targetMode == TARGET_DISPLACEMENT ){
				regionDefdAvg[bnd_id] = ((regionDefdAvg[bnd_id] - regionRestAvg[bnd_id] ) - uStar).eval();
			}else
			if( targetMode == TARGET_VELOCITY ){
				//printf("\n%% phi(v): "); cout << regionVelocityAvg[bnd_id].transpose() << " - " << uStar.transpose();
				regionDefdAvg[bnd_id] = (regionVelocityAvg[bnd_id] - uStar);
				//printf("  ||.||^2 = %.4lg ", 0.5* regionDefdAvg[bnd_id].squaredNorm());

			}

			if( uStar.hasNaN() ){
				uStar.setZero(); regionDefdAvg[bnd_id].setZero(); // ignore and zero obj fcn contribution
			}
		}else{
			uStar.setZero(); regionDefdAvg[bnd_id].setZero(); // ignore and zero obj fcn contribution
		}

		if( resetOnNextEval && dynamic_cast<TemporalInterpolationField*>(targetFields[bnd_id])!=NULL ){
			printf("\n%% WARNING: DEPRECATED: THIS CODE HAS NOT BEEN TESTED WITH RECENT MODIFICATIONS! ");
			cout << endl << "% p_shift on target " << bnd_id << " was " << ((TemporalInterpolationField*)targetFields[bnd_id])->p_shift.transpose() << ";  ";

			// update p_shift such that objective function evaluates to zero
			((TemporalInterpolationField*)targetFields[bnd_id])->p_shift += regionDefdAvg[bnd_id] - uStar;
			if( targetMode == TARGET_DISPLACEMENT ){
				((TemporalInterpolationField*)targetFields[bnd_id])->p_shift -= regionRestAvg[bnd_id];
			}
			targetFields[bnd_id]->eval(uStar, regionRestAvg[bnd_id], regionDefdAvg[bnd_id], fem.simTime);

			//debug output
			cout << "new p_shift" << ((TemporalInterpolationField*)targetFields[bnd_id])->p_shift.transpose() << "  ";
			//if( targetMode == TARGET_LOCATION ) cout << "obj.fcn. term " << (regionDefdAvg[bnd_id] - uStar).squaredNorm()*0.5;
		}

		phiVal += 0.5*scale* regionDefdAvg[bnd_id].squaredNorm();

	}
	// and objective function derivatives
	for(unsigned int k=0; k<fem.getNumberOfBndryElems(); ++k){
		unsigned int bnd_id = fem.getBoundaryId(k);
		if( targetFields.count(bnd_id)>0 ){ // have a target for this one ...
			double areaOver3 = 1.0/3.0*fem.getArea(k);
			for(int i=0; i<3; ++i){ // for all nodes of triangle k
				for(int dof=0; dof<LinearFEM::N_DOFS; ++dof){ // for all DOFs of node (k,i)
					unsigned int idof=fem.getNodalDof( fem.getBoundaryElement(k)(i) ,dof);

					if( targetMode == TARGET_VELOCITY ){
						phi_v[idof] += areaOver3*scale/regionArea[bnd_id] *regionDefdAvg[bnd_id](dof); // regionVelocityAvg now contains (vBar-vStar)
						phi_vv[idof]+= areaOver3*scale/regionArea[bnd_id];
					}else{ // displacement or location mode ...
						phi_x[idof] += areaOver3*scale/regionArea[bnd_id] *regionDefdAvg[bnd_id](dof); // regionDefdAvg now contains (uBar-uStar)
						phi_xx[idof]+= areaOver3*scale/regionArea[bnd_id];
					}
				}
			}
		}
	}
	resetOnNextEval=false;

	return phiVal;
}

void AverageBoundaryValueObjectiveFunction::reset( LinearFEM& fem ){
	if( resetMode == RESET_FIELD_OFFSETS ) resetOnNextEval=true;
}

void AverageBoundaryValueObjectiveFunction::writeTrackedLocations(std::string fname, unsigned int samples){
	std::vector<vtkSmartPointer<TemporalInterpolationField> > fields;
	std::vector<unsigned int> ids;
	for(std::map<unsigned int,VectorField*>::iterator it=trackedLocations.begin(); it!=trackedLocations.end(); ++it){
		ids.push_back(it->first);
		fields.push_back( vtkSmartPointer<TemporalInterpolationField>((TemporalInterpolationField*)it->second));
	}
	TemporalInterpolationField::writeFieldsToVTU(fname,samples,fields,ids);

	//fields.clear(); ids.clear();
	//for(std::map<unsigned int,VectorField*>::iterator it=trackedEvals.begin(); it!=trackedEvals.end(); ++it){
	//	ids.push_back(it->first);
	//	fields.push_back( vtkSmartPointer<TemporalInterpolationField>((TemporalInterpolationField*)it->second));
	//}
	//TemporalInterpolationField::writeFieldsToVTU(fname.append("_evals"),samples,fields,ids);

	for(std::map<unsigned int,VectorField*>::iterator it=trackedLocations.begin(); it!=trackedLocations.end(); ++it) if(it->second!=NULL) delete it->second;
	for(std::map<unsigned int,VectorField*>::iterator it=trackedEvals.begin();     it!=trackedEvals.end();     ++it) if(it->second!=NULL) delete it->second;
	trackedLocations.clear(); trackedEvals.clear();
}