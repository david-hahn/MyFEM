/*
 * File:   ElmerReader.h
 * Author: David
 *
 * Created on 19. November 2013, 14:16
 */

#ifndef READER_H
#define READER_H

#include <map>
#include <vector>
#include <set>
#include <cstdio>
#include <cfloat>
#include <string>

class vtkUnstructuredGrid;

namespace MyFEM{

	enum ELEM_TYPE {
	  NODE = 101,
	  LINE = 202,
	  TRI = 303,
	  TET = 504
	}; // element type constants used in Elmer meshes

	/* Depending on which kind of file we are reading, the element type is
	 * found in a different column.
	 * The first column is always the ID of the element, and
	 * the second column is the body ID to which the element belongs.
	 * In .element files the element type is in the 3rd column, whereas
	 * in .boundary files, we first have two columns for the parent element and
	 * the element type in the 5th column.
	 * All columns after the element type are the node IDs forming the element.
	 */
	enum TYPE_COLUMN { ELEMENTS_FILE = 3, BOUNDARY_FILE = 5 };

	typedef std::map<unsigned int, double> value_map;
	typedef std::map<unsigned int, std::vector<double> > node_map;
	typedef std::map<unsigned int, std::vector<unsigned int> > elem_map;
	typedef std::map<unsigned int, unsigned int> id_map;
	typedef std::set<unsigned int> id_set;
    
	inline void printMap(elem_map elems){
		for(elem_map::iterator i = elems.begin(); i != elems.end(); ++i){
			for(unsigned int j=0; j<i->second.size(); ++j){
				printf("%u\t ", i->second[j]);
			}
			printf("%% %u\n", i->first);
		}
	}
    
	inline void printMap(node_map nodes){
		for(node_map::iterator i = nodes.begin(); i != nodes.end(); ++i){
			for(unsigned int j=0; j<i->second.size(); ++j){
				printf("%lf\t ", i->second[j]);
			}
			printf("%% %u\n", i->first);
		}
	}

	inline void printMap(id_map data){
		for(id_map::iterator i = data.begin(); i != data.end(); ++i){
			printf("%u\t %% %u\n", i->second, i->first);
		}
	}

	inline id_set nodeSet(elem_map elems){
		id_set nodes;
		for(elem_map::iterator i = elems.begin(); i != elems.end(); ++i)
			for(elem_map::mapped_type::iterator j = i->second.begin();
				j != i->second.end(); ++j)
				nodes.insert(*j);
		return nodes;
	}
	/* Provides general reading methods for Elmer mesh files.
	 * These files are text based and space-separated with each line
	 * representing one node/element in the mesh.
	 */
	class ElmerReader{
	public:
	  ElmerReader(std::string meshName = ""){
		meshFile = meshName;
		nodeFile = meshName;
		elemFile = meshName;
		nodeFile.append(".nodes");
		elemFile.append(".elements");
	  }
	  ElmerReader(const ElmerReader &ori){
		nodeFile = ori.nodeFile;
		elemFile = ori.elemFile;
	  }
	  virtual ~ElmerReader(){}

	  void setNodeFile(std::string fname){ nodeFile = fname; }
	  void setElemFile(std::string fname){ elemFile = fname; }

	  /* Read elements from the input file
	   * The output parameters are elems and bodyId (optional).
	   * Previous contents will be deleted.
	   * elems contains the node IDs of nodes forming each element
	   * bodyIDs contains the body ID of each element
	   *
	   * Only elements whose body ID is in the bodies set will be read.
	   * If bodies is an empty set, all body IDs will be accepted.
	   * Returns the number of elements read successfully or -1 on error
	   *
	   * If strictlyListedBodies=false and the bodies set is empty, ALL bodies
	   * will be read from the mesh.
	   */
	  int readElems(elem_map &elems, ELEM_TYPE elemType,
					std::set<unsigned int> bodies, int typeColumn = ELEMENTS_FILE,
					id_map *bodyIDs = NULL, id_map *parentIDs = NULL,
					bool strictlyListedBodies = false);

	  /* Read nodes from the input file
	   * The output parameter is nodes. Previous contents will be deleted.
	   * Only elements whose body ID is in the bodies set will be read.
	   * Returns the number of nodes read successfully or -1 on error
	   */
	  int readNodes(node_map &nodes);

  
	  /* Reads elems from .elements file matching elemType and elemBodies
	   * (the body ID of each element is stored in bodyIDs),
	   * bndrys from .boundary file matching bndryType and bndryBodies and
	   * nodes from .nodes file
	   */
	  int readModel(node_map& nodes, elem_map& elems, id_map& bodyIDs,
		elem_map& bndrys, id_map& bndryParents, id_map& bndryIDs,
		const ELEM_TYPE elemType, const ELEM_TYPE bndryType,
		const std::set<unsigned int> elemBodies, const std::set<unsigned int> bndryBodies);

	  /* Reads all tetrahedral elements from .elements file,
	   * and all nodes from .nodes file
	   * stores the data as a VTK unstructured grid
	   * and boundary data in another VTK unstructured grid (ignored if NULL)
	   */
	  int readModel(vtkUnstructuredGrid* mesh, vtkUnstructuredGrid* bndMesh);

	protected:
	  std::string meshFile;
	  std::string nodeFile;
	  std::string elemFile;
	};

}
#endif /* READER_H */

