#ifndef TEMPORALINTERPOLATIONFIELD_H
#define TEMPORALINTERPOLATIONFIELD_H

#include "types.h"
#include <string>

#include "vtkObject.h"
#include "vtkSmartPointer.h"

class vtkKochanekSpline;
class vtkPiecewiseFunction;

namespace MyFEM{
	class DifferentiableSpline;

	class TemporalInterpolationField : public VectorField, public vtkObject{
	public:
		vtkTypeMacro(TemporalInterpolationField, vtkObject)
		static TemporalInterpolationField* New(){ return new TemporalInterpolationField(); } // for compatibility with vtkSmartPointer
		virtual void Delete(){/**printf("\n%% TemporalInterpolationField destroyed ***\n");/**/} // for compatibility with vtkSmartPointer
		virtual ~TemporalInterpolationField(){Delete();}
		vtkSmartPointer<DifferentiableSpline> xSpline, ySpline, zSpline;
		vtkSmartPointer<vtkPiecewiseFunction> haveData;
		enum DATA_MODE { DATA_POSITION, DATA_DISPLACEMENT} dataMode = DATA_POSITION; // if dataMode == DATA_POSITION eval() will treat stored point data as absolute position, if dataMode == DATA_DISPLACEMENT, eval() will treat stored point data as displacements u and return u + x0
		enum EVAL_MODE { EVAL_DEFAULT, EVAL_VELOCITY } evalMode = EVAL_DEFAULT; // evalMode = EVAL_VELOCITY all calls to eval() will be forwarded to evalVelocity() internally
		double t_shift, t_scale; // shift time by constant offset (default 0) and/or scale by constant factor (default 1)
		Eigen::Vector3d p_shift; // apply a constant offset to the result of eval

		TemporalInterpolationField();
		virtual void setInitialVelocity(const Eigen::Vector3d& v0);
		virtual void addPoint(double t, const Eigen::Vector3d& p);
		// interpolate stored data for time t and write result to u; x and x0 unused
		virtual void eval(Eigen::Vector3d& u, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const;
		// interpolate velocity for time t and write result to u; x and x0 unused
		virtual void evalVelocity(Eigen::Vector3d& u, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const;
		virtual void getRange(double* r/*[2]*/);
		virtual bool isDataAvailable(double t); // returns true if data is available at t
		virtual void addEmptyPoint(double t); // flags time t as not having data available
		
		static int buildFieldsFromTextFile(
			const std::string& fname,
			std::vector<vtkSmartPointer<TemporalInterpolationField> >& trackedTargets,
			std::vector<unsigned int>& trackedIDs, double addNoise=-1.0
		);
		static void writeFieldsToVTU(
			const std::string& fname, unsigned int samples,
			std::vector<vtkSmartPointer<TemporalInterpolationField> >& fields, std::vector<unsigned int>& fieldIDs
		){
			std::vector<Eigen::Vector3d> empty;
			writeFieldsToVTU(fname, samples, fields, fieldIDs, empty);
		}
		static void writeFieldsToVTU(
			const std::string& fname, unsigned int samples,
			std::vector<vtkSmartPointer<TemporalInterpolationField> >& fields,
			std::vector<unsigned int>& fieldIDs, std::vector<Eigen::Vector3d>& fieldOffsets
		);
	};

	class  PositionRotationInterpolationField : public TemporalInterpolationField{
	public:
		vtkTypeMacro(PositionRotationInterpolationField, TemporalInterpolationField)
		static PositionRotationInterpolationField* New(){ return new PositionRotationInterpolationField(); } // for compatibility with vtkSmartPointer
		virtual void Delete(){} // for compatibility with vtkSmartPointer
		virtual ~PositionRotationInterpolationField(){Delete();}
		// in addition to (x,y,z) position in TemporalInterpolationField, also interpolate a quaternion rotation (rw,rx,ry,rz)
		vtkSmartPointer<vtkKochanekSpline> rwSpline, rxSpline, rySpline, rzSpline;
		Eigen::Vector3d rc; // rc is centre of rotation
		
		PositionRotationInterpolationField();
		virtual void addPoint(double t, const Eigen::Vector3d& p, const Eigen::Quaterniond& r);
		virtual void eval(Eigen::Vector3d& u, const Eigen::Vector3d& x0, const Eigen::Vector3d& x, double t) const;

		int LoadDataFromCSV( const std::string& fname, double scale=1.0 ); // use scale to adjust units of position data in the file
	};

	int loadMocapDataFromCSV( const std::string& fname,
		std::vector<vtkSmartPointer<PositionRotationInterpolationField> >& rigidBodyFields,
		std::vector<std::string>& rigidBodyNames,
		std::vector<vtkSmartPointer<TemporalInterpolationField> >& markerFields,
		std::vector<std::string>& markerNames,
		double scale=1.0 ); // use scale to adjust units of position data in the file)
}

#endif
