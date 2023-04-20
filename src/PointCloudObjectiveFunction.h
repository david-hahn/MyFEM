#ifndef POINTCLOUDOBJECTIVEFUNCTION_H
#define POINTCLOUDOBJECTIVEFUNCTION_H


#include "vtkUnstructuredGrid.h"
#include "vtkSmartPointer.h"

#include "SensitivityAnalysis.h" // contains ObjectiveFunction base class

namespace MyFEM{
	class LinearFEM; // fwd decl

	class PointCloudObjectiveFunction : public ObjectiveFunction {
	public:
		PointCloudObjectiveFunction() : ObjectiveFunction() {}

		virtual void setPointCloud(vtkSmartPointer<vtkUnstructuredGrid> pointCloud_) { pointCloud = pointCloud_; }

		virtual double evaluate(LinearFEM& theFEM,
			Eigen::VectorXd& phi_x, Eigen::VectorXd& phi_v, Eigen::VectorXd& phi_f, Eigen::VectorXd& phi_q
		);

		vtkSmartPointer<vtkUnstructuredGrid> pointCloud;
		vtkSmartPointer<vtkUnstructuredGrid> closestPoints;
		vtkSmartPointer<vtkUnstructuredGrid> closestPointLines; // used only for debug output, only set if storeCPlines==true
		bool storeCPlines = true;
		const char* barycentricName = "barycentric coord";
		const char* closestCellName = "closest cell";
	};

	class TimeDependentPointCloudObjectiveFunction : public PointCloudObjectiveFunction {
	public:
		TimeDependentPointCloudObjectiveFunction() : PointCloudObjectiveFunction() { storeCPlines = false; }

		virtual void loadFromFileSeries(std::string fileNamePrefix, double timestep=0.0); // if file (fileNamePrefix+"frametimes.txt") exists, timestep is not used
		virtual double evaluate(LinearFEM& theFEM,
			Eigen::VectorXd& phi_x, Eigen::VectorXd& phi_v, Eigen::VectorXd& phi_f, Eigen::VectorXd& phi_q
		);
		virtual unsigned int getFrameBefore(double time);
		virtual unsigned int getFrameAfter (double time);

		std::map< unsigned int, vtkSmartPointer<vtkUnstructuredGrid> > pointCloudFrames;
		std::map< unsigned int, double > timeOfFrame;
	};

}

#endif // POINTCLOUDOBJECTIVEFUNCTION_H
