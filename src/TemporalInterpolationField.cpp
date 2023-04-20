#include "TemporalInterpolationField.h"
#include "DifferentiableSpline.h"

#include <string>
#include <sstream>
#include <fstream>

#include "types.h"

// for output ...
#include "vtkCellArray.h"
#include "vtkIntArray.h"
#include "vtkDoubleArray.h"
#include "vtkCellData.h"
#include "vtkPointData.h"
#include "vtkLine.h"
#include "vtkUnstructuredGrid.h"
#include "vtkXMLUnstructuredGridWriter.h"
#include "vtkPiecewiseFunction.h"

using namespace MyFEM;

TemporalInterpolationField::TemporalInterpolationField(){
	t_shift = 0.0; t_scale = 1.0;
	p_shift.setZero();
	//printf("\n%% TemporalInterpolationField constructed ***\n");
	xSpline = vtkSmartPointer<DifferentiableSpline>::New();
	ySpline = vtkSmartPointer<DifferentiableSpline>::New();
	zSpline = vtkSmartPointer<DifferentiableSpline>::New();
	haveData = vtkSmartPointer<vtkPiecewiseFunction>::New();

	haveData->ClampingOff(); // haveData will return 0 outside of domain

	xSpline->ClosedOff();
	xSpline->SetLeftConstraint(2);  // Set the left and right second derivatives to 0 corresponding to linear interpolation
	xSpline->SetLeftValue(0);
	xSpline->SetRightConstraint(2);
	xSpline->SetRightValue(0);

	ySpline->ClosedOff();
	ySpline->SetLeftConstraint(2);  // Set the left and right second derivatives to 0 corresponding to linear interpolation
	ySpline->SetLeftValue(0);
	ySpline->SetRightConstraint(2);
	ySpline->SetRightValue(0);

	zSpline->ClosedOff();
	zSpline->SetLeftConstraint(2);  // Set the left and right second derivatives to 0 corresponding to linear interpolation
	zSpline->SetLeftValue(0);
	zSpline->SetRightConstraint(2);
	zSpline->SetRightValue(0);
}

void TemporalInterpolationField::setInitialVelocity(const Eigen::Vector3d& v0){
	xSpline->SetLeftConstraint(1); xSpline->SetLeftValue(v0(0));
	ySpline->SetLeftConstraint(1); ySpline->SetLeftValue(v0(1));
	zSpline->SetLeftConstraint(1); zSpline->SetLeftValue(v0(2));
}

void TemporalInterpolationField::addPoint(double t, const Eigen::Vector3d& p){
	xSpline->AddPoint(t,p[0]);
	ySpline->AddPoint(t,p[1]);
	zSpline->AddPoint(t,p[2]);
	haveData->AddPoint(t,1.0,0.5,0.0); // sharpness of 0 yields a piecewise linear function
}

// interpolate stored data for time t and write result to u; x and x0 unused
void TemporalInterpolationField::eval(Eigen::Vector3d& u, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
	if( evalMode == EVAL_VELOCITY ) return evalVelocity(u,x0,x,t);
	t*=t_scale; t+=t_shift;
	u[0] = xSpline->Evaluate(t);
	u[1] = ySpline->Evaluate(t);
	u[2] = zSpline->Evaluate(t);
	u += p_shift;
	if( dataMode == DATA_DISPLACEMENT ) u += x0;
}

void TemporalInterpolationField::evalVelocity(Eigen::Vector3d& u, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const {
	t*=t_scale; t+=t_shift;
	u[0] = xSpline->EvaluateTDerivative(t);
	u[1] = ySpline->EvaluateTDerivative(t);
	u[2] = zSpline->EvaluateTDerivative(t);
}


void TemporalInterpolationField::getRange(double* r/*[2]*/){
	xSpline->GetParametricRange(r);
	r[0]-=t_shift; r[1]-=t_shift;
}

bool TemporalInterpolationField::isDataAvailable(double t){
	t*=t_scale; t+=t_shift;
	return ( haveData->GetValue(t) > (1.0-FLT_EPSILON) );
}
void TemporalInterpolationField::addEmptyPoint(double t){
	haveData->AddPoint(t,0.0,0.5,0.0); // sharpness of 0 yields a piecewise linear function
}
		
int TemporalInterpolationField::buildFieldsFromTextFile(
	const std::string& fname,
	std::vector<vtkSmartPointer<TemporalInterpolationField> >& trackedTargets,
	std::vector<unsigned int>& trackedIDs, double addNoise
){
	trackedIDs.clear();
	trackedTargets.clear();
	std::ifstream in(fname);
	std::stringstream strstream;
	std::string line, token;
	unsigned int tmpUint; double tmpDbl;
	if (!in.is_open()){
		printf("\n%% FAILED TO OPEN FILE \"%s\"\n", fname.c_str());
		return -1;
	}
	bool first=true;
	while (in.good()) {
		getline(in, line);
		strstream.clear();
		strstream.str(line);
		if( first ){ // first line contains IDs of tracked boundaries
			first = false;
			while( getline(strstream, token, ' ') ){
				// add boundary ID from token
				if( sscanf(token.c_str(), "%u", &tmpUint) ==1 ){
					trackedIDs.push_back(tmpUint);
					trackedTargets.push_back(
						vtkSmartPointer<TemporalInterpolationField>::New()
					);
				}
			}
		}else{ // regular line - contains time and triplets of displacements in the same order as trackedIDs
			// read time
			getline(strstream, token, ' ');
			if( sscanf(token.c_str(), "%lf", &tmpDbl) ==1 ){
				double currentTime = tmpDbl;
				unsigned int next = 0, coord = 0;
				Eigen::Vector3d u;
				// read triplets of (x y z) displacements
				while( getline(strstream, token, ' ') && next<trackedIDs.size()){
					if( sscanf(token.c_str(), "%lf", &tmpDbl) !=1 ) tmpDbl=0.0;
					u[coord] = tmpDbl; coord = (coord+1)%3;
					if( coord==0 ){
						if( addNoise>0.0 ) u += addNoise*Eigen::Vector3d::Random(); // uniform random in addNoise*[-1;1]
						trackedTargets[next++]->addPoint(currentTime, u);
					}
				}
			}
		}
	}
	in.close();
	return 0;
}

void TemporalInterpolationField::writeFieldsToVTU(
	const std::string& fname, unsigned int samples,
	std::vector<vtkSmartPointer<TemporalInterpolationField> >& fields,
	std::vector<unsigned int>& fieldIDs, std::vector<Eigen::Vector3d>& fieldOffsets
){
	// create evenly spaced samples of all input fields
	// write to an unstructured grid (connect every two samples by an edge)
	// write fieldID as cell data (so we can color them nicely)
	// write time code as point data

	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New(); points->SetDataTypeToDouble();
	points->SetNumberOfPoints(samples * fields.size());
	unsigned int nextPoint=0;

	vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
	cells->Allocate( cells->EstimateSize( (samples-1) * fields.size(), 2) );
	vtkSmartPointer<vtkLine> line;
	unsigned int nextCell=0;

	vtkSmartPointer<vtkIntArray> cellIDs = vtkSmartPointer<vtkIntArray>::New();
	cellIDs->SetName( "fieldID" );
	cellIDs->SetNumberOfComponents(1); cellIDs->SetNumberOfTuples( (samples-1) * fields.size() );

	vtkSmartPointer<vtkIntArray> dataAvailable = vtkSmartPointer<vtkIntArray>::New();
	dataAvailable->SetName( "dataAvailable" );
	dataAvailable->SetNumberOfComponents(1); dataAvailable->SetNumberOfTuples( samples * fields.size() );

	vtkSmartPointer<vtkDoubleArray> pointTi = vtkSmartPointer<vtkDoubleArray>::New();
	pointTi->SetName( "time" );
	pointTi->SetNumberOfComponents(1); pointTi->SetNumberOfTuples( samples * fields.size() );

	vtkSmartPointer<vtkDoubleArray> velocity = vtkSmartPointer<vtkDoubleArray>::New();
	velocity->SetName( "velocity" );
	velocity->SetNumberOfComponents(3); velocity->SetNumberOfTuples( samples * fields.size() );

	for(int k=0; k<fields.size(); ++k){
		double tRange[2]; fields[k]->getRange(tRange);
		Eigen::Vector3d u,x0; x0.setZero();
		if( fieldOffsets.size()>k ) x0 = fieldOffsets[k];
		for(int i=0; i<samples; ++i){
			double ti = tRange[0]+i*(tRange[1]-tRange[0])/((double)(samples-1));
			TemporalInterpolationField::EVAL_MODE m = fields[k]->evalMode;
			fields[k]->evalMode = TemporalInterpolationField::EVAL_DEFAULT;
			fields[k]->eval(u,x0,x0,ti);
			points->SetPoint(nextPoint, u.data() );
			pointTi->SetTuple1(nextPoint, ti);
			dataAvailable->SetTuple1(nextPoint, fields[k]->isDataAvailable(ti) );
			fields[k]->evalVelocity(u,x0,x0,ti);
			velocity->SetTuple3(nextPoint, u(0),u(1),u(2)); //cout << endl << u.transpose();
			fields[k]->evalMode = m;
			if( i>0 ){
				line = vtkSmartPointer<vtkLine>::New();
				line->GetPointIds()->SetId(0,nextPoint-1);
				line->GetPointIds()->SetId(1,nextPoint  );
				cells->InsertNextCell(line);
				int fieldId=0;
				if( fieldIDs.size()>k ) fieldId =  fieldIDs[k];
				cellIDs->SetTuple1(nextCell++, fieldId);
			}
			++nextPoint;
		}
	}

	vtkSmartPointer<vtkUnstructuredGrid> mesh = vtkSmartPointer<vtkUnstructuredGrid>::New();
	mesh->SetPoints(points);
	mesh->SetCells(VTK_LINE, cells);
	mesh->GetCellData()->AddArray(cellIDs);
	mesh->GetCellData()->SetActiveAttribute(0,vtkDataSetAttributes::SCALARS);
	mesh->GetPointData()->AddArray(pointTi);
	mesh->GetPointData()->AddArray(dataAvailable);
	mesh->GetPointData()->AddArray(velocity);

	vtkSmartPointer<vtkXMLUnstructuredGridWriter> out = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
	out->SetFileName((fname+".vtu").c_str());
	out->SetInputData(mesh);
	out->Write();
}



PositionRotationInterpolationField::PositionRotationInterpolationField(){
	rc.setZero();
	
	rwSpline = vtkSmartPointer<vtkKochanekSpline>::New();
	rxSpline = vtkSmartPointer<vtkKochanekSpline>::New();
	rySpline = vtkSmartPointer<vtkKochanekSpline>::New();
	rzSpline = vtkSmartPointer<vtkKochanekSpline>::New();

	rwSpline->ClosedOff();
	rwSpline->SetLeftConstraint(2);  // Set the left and right second derivatives to 0 corresponding to linear interpolation
	rwSpline->SetLeftValue(0);
	rwSpline->SetRightConstraint(2);
	rwSpline->SetRightValue(0);

	rxSpline->ClosedOff();
	rxSpline->SetLeftConstraint(2);  // Set the left and right second derivatives to 0 corresponding to linear interpolation
	rxSpline->SetLeftValue(0);
	rxSpline->SetRightConstraint(2);
	rxSpline->SetRightValue(0);

	rySpline->ClosedOff();
	rySpline->SetLeftConstraint(2);  // Set the left and right second derivatives to 0 corresponding to linear interpolation
	rySpline->SetLeftValue(0);
	rySpline->SetRightConstraint(2);
	rySpline->SetRightValue(0);

	rzSpline->ClosedOff();
	rzSpline->SetLeftConstraint(2);  // Set the left and right second derivatives to 0 corresponding to linear interpolation
	rzSpline->SetLeftValue(0);
	rzSpline->SetRightConstraint(2);
	rzSpline->SetRightValue(0);
}

void PositionRotationInterpolationField::addPoint(double t, const Eigen::Vector3d& p, const Eigen::Quaterniond& r){
	TemporalInterpolationField::addPoint(t,p);
	rwSpline->AddPoint(t,r.w());
	rxSpline->AddPoint(t,r.x());
	rySpline->AddPoint(t,r.y());
	rzSpline->AddPoint(t,r.z());
}

void PositionRotationInterpolationField::eval(Eigen::Vector3d& u, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const{
	TemporalInterpolationField::eval(u,x0,x,t);
	t*=t_scale; t+=t_shift; //TemporalInterpolationField::eval does the same to t, but not by ref
	Eigen::Quaterniond r;
	r.w() = rwSpline->Evaluate(t);
	r.x() = rxSpline->Evaluate(t);
	r.y() = rySpline->Evaluate(t);
	r.z() = rzSpline->Evaluate(t);
	r.normalize();
	//       rotated rest coord  + displacement
	u = ( r*(x0-rc) + rc + u ).eval();
}

int PositionRotationInterpolationField::LoadDataFromCSV( const std::string& fname, double scale ){
	xSpline->RemoveAllPoints(); ySpline->RemoveAllPoints(); zSpline->RemoveAllPoints();
	rwSpline->RemoveAllPoints(); rxSpline->RemoveAllPoints(); rySpline->RemoveAllPoints(); rzSpline->RemoveAllPoints();

	std::ifstream in(fname);
	std::stringstream strstream;
	std::string line, token;
	unsigned int tmpUint; double tmpDbl;
	double currentTime; Eigen::Vector3d u; Eigen::Quaterniond r;
	if (!in.is_open()){
		printf("\n%% FAILED TO OPEN FILE \"%s\"\n", fname.c_str());
		return -1;
	}
	bool first=true;
	while (in.good()) {
		getline(in, line);
		strstream.clear();
		strstream.str(line);
		// columns per line must be: frame number (ignored), time, rotation quaternion (rx,ry,rz,rw), position (x,y,z) -- other columns are ignored

		//printf("\n%% in: %s",line.c_str());

		char delim = ';';
		getline(strstream, token, delim);
		if( sscanf(token.c_str(), "%u", &tmpUint) ==1 ){ // first column: frame number -- if not skip this line (could be comment or empty line)

			getline(strstream, token, delim);
			if( sscanf(token.c_str(), "%lf", &tmpDbl) ==1 ) currentTime = tmpDbl; // second column: time
			else return -1;

			// next 4 columns: rotation
			getline(strstream, token, delim);
			if( sscanf(token.c_str(), "%lf", &tmpDbl) ==1 ) r.x() = tmpDbl;
			else return -1;
			getline(strstream, token, delim);
			if( sscanf(token.c_str(), "%lf", &tmpDbl) ==1 ) r.y() = tmpDbl;
			else return -1;
			getline(strstream, token, delim);
			if( sscanf(token.c_str(), "%lf", &tmpDbl) ==1 ) r.z() = tmpDbl;
			else return -1;
			getline(strstream, token, delim);
			if( sscanf(token.c_str(), "%lf", &tmpDbl) ==1 ) r.w() = tmpDbl;
			else return -1;

			// next 3 columns: position
			for(int i=0; i<3; ++i){
				getline(strstream, token, delim);
				if( sscanf(token.c_str(), "%lf", &tmpDbl) ==1 ) u[i] = scale*tmpDbl;
				else return -1;
			}

			// done, add data and continue with next line
			addPoint(currentTime, u, r);
			//printf("\n%% add: %.4lg : (%.4lg %.4lg %.4lg %.4lg) : (%.4lg %.4lg %.4lg )",currentTime, r.x(),r.y(),r.z(),r.w(), u[0],u[1],u[2] );
		}
	}
	in.close();
	double range[2];
	getRange(range);
	printf("\n%% CSV file read, time range is (%.4lg, %.4lg) ", range[0],range[1]);

	// set post-translation such that the centre of rotation at t=range[0] matches the model centre of rotation in world space
	Eigen::Vector3d u0;
	eval(u0,rc,rc,range[0]);
	p_shift = rc-u0;

	return 0;
}

int MyFEM::loadMocapDataFromCSV( const std::string& fname,
	std::vector<vtkSmartPointer<PositionRotationInterpolationField> >& rigidBodyFields,
	std::vector<std::string>& rigidBodyNames,
	std::vector<vtkSmartPointer<TemporalInterpolationField> >& markerFields,
	std::vector<std::string>& markerNames, double scale
){	// here we read CSV files exported from OptiTrack Motive directly
	// the files should contain a header that has the following data in the first 7 lines:
	// 1: general data, version, name, etc. -- ignored
	// 2: empty -- ignored
	// 3: type of data per column (either "Rigid Body" or "Marker" -- other values ignored)
	// 4: name of data per column (will be copied to rigidBodyNames or markerNames)
	// 5: ID -- ignored
	// 6: column type (either "Rotation" or "Position" -- other values ignored)
	// 7: column header (first column must be "Frame" - ignored, second must be time, then either "X","Y","Z","W", other values ignored)
	// afterwards each line must contain data for one frame


	char delim = ',';
	std::ifstream in(fname);
	std::stringstream strstream;
	std::string line, token;
	unsigned int lineNr=0, columnNr=0, i;
	unsigned int currentFrame, activeRbField=0, activeMkField=0;
	double currentTime;
	double tmpDbl;
	Eigen::Vector3d u; Eigen::Quaterniond r;
	std::vector<unsigned int> rigidBodyStartColumns, markerStartColumns;

	std::vector<std::string> groupTypes, types, names, directions, *activeList=NULL;

	if (!in.is_open()){
		printf("\n%% FAILED TO OPEN FILE \"%s\"\n", fname.c_str());
		return -1;
	}
	//printf("\n%% Reading \"%s\" ...\n", fname.c_str());

	while (in.good()) {
		getline(in, line); ++lineNr; // count lines 1-based
		strstream.clear();
		strstream.str(line);

		// reading header
		if( lineNr==3 ){ // line 3 must be group types
			activeList = &groupTypes;
			// also check if we use ';' or ',' (default) to separate columns
			if( line.find(';')!=std::string::npos ){
				if( line.find(',')!=std::string::npos ){
					if( line.find(';') < line.find(',') ) delim = ';'; // ';' appears before ','
				}else delim = ';'; // only ';' appears
			}
		}else
		if( lineNr==4 ){ // line 4 must be names
			activeList = &names;
		}else
		if( lineNr==6 ){ // line 6 must be column types
			activeList = &types;
		}else
		if( lineNr==7 ){ // line 7 must be directions
			activeList = &directions;
		}else activeList=NULL;

		if( activeList!=NULL ){
			columnNr=0;
			while( strstream.good() ){
				getline(strstream, token, delim);
				if( columnNr>=2) activeList->push_back(token); // skip the first two columns (must be Frame and Time)
				++columnNr;
			}
		}

		if( lineNr==7 ){ // headers are done - prepare fields now
			for(columnNr=0; columnNr<directions.size(); ++columnNr){
				if( groupTypes[columnNr].find("Rigid Body")!=std::string::npos &&
					types[columnNr].find("Rotation")!=std::string::npos &&
					directions[columnNr].find("X")!=std::string::npos
				){	// "Rigid Body" + "Rotation" + "X" marks the start of a rigid body data block -- must have at least 7 columns, so skip the next 6
					rigidBodyFields.push_back( vtkSmartPointer<PositionRotationInterpolationField>::New() );
					rigidBodyNames.push_back(  names[columnNr] );
					rigidBodyStartColumns.push_back( columnNr  );
					printf("\n%% Rigid body data \"%s\" starting in column %d. ", names[columnNr].c_str(), columnNr);
					columnNr+=6;
				}else
				if( groupTypes[columnNr].find("Marker")!=std::string::npos &&
					types[columnNr].find("Position")!=std::string::npos &&
					directions[columnNr].find("X")!=std::string::npos
				){	// "Rigid Body" + "Rotation" + "X" marks the start of a rigid body data block -- must have at least 3 columns, so skip the next 2
					markerFields.push_back( vtkSmartPointer<TemporalInterpolationField>::New() );
					markerNames.push_back(  names[columnNr] );
					markerStartColumns.push_back( columnNr  );
					printf("\n%% Marker data \"%s\" starting in column %d. ", names[columnNr].c_str(), columnNr);
					columnNr+=2;
				}
			}
		}

		if( lineNr > 7 ){ // data line
			// columns per line must be: frame number (ignored), time, rotation quaternion (rx,ry,rz,rw), position (x,y,z) -- other columns are ignored

			//printf("\n%% in: %s",line.c_str());

			getline(strstream, token, delim);
			if( sscanf(token.c_str(), "%u", &currentFrame) ==1 ){ // first column: frame number -- if not skip this line (could be comment or empty line)

				getline(strstream, token, delim);
				if( sscanf(token.c_str(), "%lf", &tmpDbl) ==1 ) currentTime = tmpDbl; // second column: time
				else return -1;

				// now go through the rest of the columns and construct data ...
				columnNr=0; // start counting now
				activeRbField=0; activeMkField=0;
				i=0; //printf("\n%% t=%.4lg ", currentTime);
				while( strstream.good() ){
					getline(strstream, token, delim); //printf(" (%d) \"%s\" ", columnNr, token.c_str());

					if( groupTypes[columnNr].find("Rigid Body")!=std::string::npos && activeRbField<rigidBodyStartColumns.size() && columnNr>=rigidBodyStartColumns[activeRbField] ){ // processing data for a rigid body
						if( sscanf(token.c_str(), "%lf", &tmpDbl) ==1 ){
							//printf(" rb-%d ", i);
							if( i==0 ) r.x() = tmpDbl; else
							if( i==1 ) r.y() = tmpDbl; else
							if( i==2 ) r.z() = tmpDbl; else
							if( i==3 ) r.w() = tmpDbl; else
							if( i>=4 && i<7 ) u(i-4) = scale*tmpDbl; // position components in order (x,y,z)
							++i;

							if( i==7 ){ // data complete -- write and move to next RB field
								if( std::abs(r.norm()-1.0)>1e-4 ) printf(" ||r||=%.4lg ",r.norm());
								rigidBodyFields[activeRbField]->addPoint(currentTime,u,r); //printf(" add rb %d", activeRbField);
								++activeRbField;
								i=0;
							}
						}
						if( (activeRbField+1)<rigidBodyStartColumns.size() && (columnNr+1)>=rigidBodyStartColumns[activeRbField+1] ){
							++activeRbField; // skip if data was incomplete
							i=0;
						}
					}else
					if( groupTypes[columnNr].find("Marker")!=std::string::npos && activeMkField<markerStartColumns.size() && columnNr>=markerStartColumns[activeMkField]  ){ // processing data for a marker
						if( sscanf(token.c_str(), "%lf", &tmpDbl) ==1 ){
							//printf(" mk-%d ", i);
							if( i<3 ) u(i) = scale*tmpDbl; // position components in order (x,y,z)
							++i;

							if( i==3 ){ // data complete -- write and move to next MK field
								markerFields[activeMkField]->addPoint(currentTime,u); //printf(" add mk %d", activeMkField);
								++activeMkField;
								i=0;
							}
						}
						if( (activeMkField+1)<markerStartColumns.size() && (columnNr+1)>=markerStartColumns[activeMkField+1] ){
							markerFields[activeMkField]->addEmptyPoint(currentTime); // store flag that marker data is NOT available at this time ...
							++activeMkField;
							i=0;
						}

					}

					++columnNr;
				}
			}
		}
	}
	in.close();
	//double range[2];
	//getRange(range);
	//printf("\n%% CSV file read, time range is (%.4lg, %.4lg) ", range[0],range[1]);

	//// set post-translation such that the centre of rotation at t=range[0] matches the model centre of rotation in world space
	//Eigen::Vector3d u0;
	//eval(u0,rc,rc,range[0]);
	//p_shift = rc-u0;


	return 0;
}


