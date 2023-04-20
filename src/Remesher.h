#ifndef REMESHER_H
#define REMESHER_H

#include "LinearFEM.h"

#include <vtkSmartPointer.h>
class vtkUnstructuredGrid;

#include <mmg/mmg3d/libmmgtypes.h>

namespace MyFEM{
	class Remesher{
	public:
		Remesher(){mmgMesh=NULL;mmgSfcn=NULL; setDefaultOptions();}
		virtual ~Remesher(){ cleanup(); }
		void cleanup();

		void setDefaultOptions();
		void setTargetEdgeLengthFromMesh(vtkSmartPointer<vtkUnstructuredGrid> sample); // compute the mean edge length in the given mesh and set this as a target for future remeshing operations (does not change or store the mesh)

		void remesh(LinearFEM& fem);
	
		void buildMMGmesh(LinearFEM& fem);        // Step 1: build a MMG mesh from the data stored in the FEM object
		void buildSizingFunction(LinearFEM& fem); // Step 2: build a MMG mesh from the data stored in the FEM object
		void buildVTKmesh();                      // Step 3: build a VTK mesh from the result of the MMG remeshing
		void buildBoundaryMesh();                 // Step 4: build a VTK mesh for the boundary of the new mesh and interpolate boundary IDs
		void interpolateData(LinearFEM& fem);     // Step 5: interpolate data from the input mesh to the remeshed version


		MMG5_pMesh mmgMesh; // mesh storage in MMG format
		MMG5_pSol  mmgSfcn; // sizing function for remeshing
		vtkSmartPointer<vtkUnstructuredGrid> newMesh; // VTK storage for remeshing result
		vtkSmartPointer<vtkUnstructuredGrid> newMeshBoundary;
		Eigen::MatrixXd                      newMatParams, newViscParams;
		std::map<int,double> dOpt;
		std::map<int,int>    iOpt;
		double targetEdgeLength; // remesh to this target edge length in the deformed configuration
	};

}

#endif // !REMESHER_H
