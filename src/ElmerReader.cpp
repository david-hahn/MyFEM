/*
 * File:   ElmerReader.cpp
 * Author: David
 *
 * Created on 19. November 2013, 14:16
 */

#include "ElmerReader.h"
#include "fieldNames.h"

// VTK includes for reading directly into VTK data structures
#include <vtkUnstructuredGrid.h>
#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkTetra.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkIntArray.h>
#include <vtkTriangle.h>

#include <fstream>
#include <sstream>
#include <cstdio>

using namespace MyFEM;
using namespace std;

int ElmerReader::readElems(elem_map &elems, ELEM_TYPE elemType,
							std::set<unsigned int> bodies, int typeColumn,
							id_map *bodyIDs, id_map *parentIDs,
							bool strictlyListedBodies) {
	unsigned int elemId, bodyId, buffer;
	int column;
	bool flag;
	istringstream strstream;
	string line, token;
	std::vector<unsigned int> nodes; // store node IDs
	unsigned int parent; // store parents when reading boundary elements
	elems.clear();
	if (bodyIDs != NULL)
	bodyIDs->clear();

	ifstream in(elemFile.c_str());
	if (!in.is_open())
	return -1;
	while (in.good()) {
	getline(in, line);
	strstream.clear();
	strstream.str(line);
	nodes.clear();
	parent=0;
	//            printf("parsing line \"%s\"\n", line.c_str());
	//            printf("strstream holds \"%s\"\n", strstream.str().c_str());
	// first column is element ID
	getline(strstream, token, ' ');
	sscanf(token.c_str(), "%u", &elemId);
	//            printf(" token 1 is \"%s\"\n",token.c_str());
	// second column is body id
	getline(strstream, token, ' ');
	sscanf(token.c_str(), "%u", &bodyId);
	//            printf(" token 2 is \"%s\"\n",token.c_str());
	//            printf("body id is %u", bodyId);
	flag = (!strictlyListedBodies && bodies.empty()) ||
			(bodies.count(bodyId) > 0); // only read further if body ID matches
	//            printf(" -- %s found\n", flag?"":"not");
	column = 3;
	while (getline(strstream, token, ' ') && flag) {
		sscanf(token.c_str(), "%u", &buffer);
		//                printf(" token %u is \"%s\"\n",column, token.c_str());
		if ((parentIDs != NULL) && (column < typeColumn)) { // columns 3 and 4 in a boundary file are the parent elements of the boundary element (face) - usually only one of them is non-zero (each boundary element should be the face of exactly one higher dimensional element)
		if( parent==0 ) parent=buffer;
		} else if (column == typeColumn) {
		// check the element type
		flag = (buffer == elemType); // stop reading if the elment type is wrong
		} else if (column > typeColumn) {
		// store node ID
		nodes.push_back(buffer);
		}
		++column;
	}
	if (flag && !nodes.empty()) {
		elems[elemId] = nodes;
		if (bodyIDs != NULL)
		(*bodyIDs)[elemId] = bodyId;
		if (parentIDs != NULL)
		(*parentIDs)[elemId] = parent;
		//                printf("added element %u\n", elemId);
	}
	}
	in.close();
	return elems.size();
}

int ElmerReader::readNodes(node_map &nodes) {
	string line;
	double coords[3]; // store node coords
	int id, buffer, tokens;
	char test;
	nodes.clear();

	// printf("reading nodes from \"%s\"\n",nodeFile.c_str());
	ifstream in(nodeFile.c_str());
	if (!in.is_open())
	return -1;
	// printf("file opened successfully\n");
	while (in.good()) {
	getline(in, line);
	test = 0;
	// format for nodes is "node-id unused x y z"
	tokens = sscanf(line.c_str(), "%d %d %lf %lf %lf", &id, &buffer, coords,
					coords + 1, coords + 2);
	// printf("read %d tokens: %d %d %lf %lf %lf, test is %d\n",
	//        tokens, id, buffer, coords[0], coords[1], coords[2], test);
	if (tokens == 5 && test == 0) { // line format is correct
		nodes[id] = vector<double>(coords, coords + 3);
	}
	}
	in.close();
	// printf("read %d nodes from %s\n", nodes.size(),nodeFile.c_str());
	return nodes.size();
}

int ElmerReader::readModel(node_map& nodes, elem_map& elems, id_map& bodyIDs,
	elem_map& bndrys, id_map& bndryParents, id_map& bndryIDs,
	const ELEM_TYPE elemType, const ELEM_TYPE bndryType,
	const std::set<unsigned int> elemBodies, const std::set<unsigned int> bndryBodies
){
	//node_map nodes_in;
	//elem_map elems_in;
	int ret;
	// read elements and nodes
	ret=readElems(elems, elemType, elemBodies,ELEMENTS_FILE,&bodyIDs,NULL, !elemBodies.empty());
	if(ret<0) return ret;
	ret=readNodes(nodes);
	if(ret<0) return ret;
	// now switch to .boundary file
	string tmp=elemFile;
	elemFile=meshFile;
	elemFile.append(".boundary");
	ret=readElems(bndrys,bndryType,bndryBodies,BOUNDARY_FILE,&bndryIDs,&bndryParents, !bndryBodies.empty());
	// restore state of the reader object
	elemFile=tmp;	

	//// reading is complete, but for HyENA-BEM compatibility the nodes need to be numbered
	//// in the same order as they appear in the element map.
	//
	//// run through the element map and decide new node numbers
	//id_map fwd;// bkwd; // fwd[old_id]=new_id, bkwd[new_id]=old_id
	//unsigned int new_id=0;
	//for(elem_map::iterator i = elems.begin(); i!=elems.end(); ++i){
	//	//run through all nodes of the element
	//	for(elem_map::mapped_type::iterator j = i->second.begin(); j!=i->second.end(); ++j){
	//		if(fwd.count(*j)==0){ // assing new number at first occurence of node
	//			fwd[*j]=new_id; //bkwd[new_id]=*j;
	//			new_id++;
	//		}
	//		(*j)=fwd[*j]; //update element
	//	}
	//}
	//nodes.clear();
	//// copy from nodes_in to nodes while applying new numbering
	//for(node_map::iterator i = nodes_in.begin(); i!= nodes_in.end(); ++i){
	//	nodes[fwd[i->first]] = i->second;
	//}
	//// apply new numbering to bndry elements
	//for(elem_map::iterator i = bndrys.begin(); i!=bndrys.end(); ++i){
	//	for(elem_map::mapped_type::iterator j = i->second.begin(); j!=i->second.end(); ++j){
	//		(*j)=fwd[*j]; //update element
	//	}
	//}

	return elems.size();
}


int ElmerReader::readModel(vtkUnstructuredGrid* mesh, vtkUnstructuredGrid* bndMesh){
	node_map nodes; elem_map elems, bndrys; id_map bodyIDs, bndryIDs, bndryParents;
	set<unsigned int> emptyForAll; // pass empty sets to read all bodies/boundaries from the mesh
	mesh->Initialize();

	int n=readModel(nodes, elems, bodyIDs, bndrys, bndryParents, bndryIDs, TET, TRI, emptyForAll, emptyForAll);

	if( n>0 ){ // n is the number of elements read

		// nodes
		//printf("\n adding nodes ...");
		vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New(); points->SetDataTypeToDouble();
		points->SetNumberOfPoints(nodes.size());
		for(node_map::iterator i = nodes.begin(); i!= nodes.end(); ++i){
			points->SetPoint(i->first-nodes.begin()->first, i->second.data()); // convert to 0-based node index
		}
		mesh->SetPoints(points);

		// elements
		//printf("\n adding elems ...");
		vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
		cells->Allocate( cells->EstimateSize(elems.size(), 4) );
		vtkSmartPointer<vtkTetra> tetra = vtkSmartPointer<vtkTetra>::New();
		for(elem_map::iterator i = elems.begin(); i!=elems.end(); ++i){			
			tetra->GetPointIds()->SetId(0, i->second[0]-nodes.begin()->first ); // convert to 0-based node index
			tetra->GetPointIds()->SetId(1, i->second[1]-nodes.begin()->first );
			tetra->GetPointIds()->SetId(2, i->second[2]-nodes.begin()->first );
			tetra->GetPointIds()->SetId(3, i->second[3]-nodes.begin()->first );
			//printf("\n adding elem %d %d %d %d ",tetra->GetPointId(0),tetra->GetPointId(1),tetra->GetPointId(2),tetra->GetPointId(3));
			cells->InsertNextCell(tetra);
		}
		mesh->SetCells(VTK_TETRA, cells);

		// body IDs per element
		vtkSmartPointer<vtkIntArray> data = vtkSmartPointer<vtkIntArray>::New();
		data->SetName( fieldNames[BODY_NAME].c_str() );
		data->SetNumberOfComponents(1); data->SetNumberOfTuples(cells->GetNumberOfCells());
		for(elem_map::iterator i = elems.begin(); i!=elems.end(); ++i){
			//printf("\n body of elem %d is %d", i->first, bodyIDs[i->first]);
			data->SetTuple1(i->first - elems.begin()->first, bodyIDs[i->first]); // convert to 0-based element index
		}
		mesh->GetCellData()->AddArray(data);

		if( bndMesh != NULL ){
			bndMesh->Initialize();
			// nodes
			points = vtkSmartPointer<vtkPoints>::New(); points->SetDataTypeToDouble();
			points->SetNumberOfPoints(nodes.size());
			for(node_map::iterator i = nodes.begin(); i!= nodes.end(); ++i){
				points->SetPoint(i->first-nodes.begin()->first, i->second.data()); // convert to 0-based node index
			}
			bndMesh->SetPoints(points);

			cells = vtkSmartPointer<vtkCellArray>::New();
			cells->Allocate( cells->EstimateSize(bndrys.size(), 3) );
			for(elem_map::iterator i = bndrys.begin(); i!=bndrys.end(); ++i){
				vtkSmartPointer<vtkTriangle> tri = vtkSmartPointer<vtkTriangle>::New();
				tri->GetPointIds()->SetId(0, i->second[0]-nodes.begin()->first ); // convert to 0-based node index
				tri->GetPointIds()->SetId(1, i->second[1]-nodes.begin()->first );
				tri->GetPointIds()->SetId(2, i->second[2]-nodes.begin()->first );
				cells->InsertNextCell(tri);
			}
			bndMesh->SetCells(VTK_TRIANGLE, cells);

			data = vtkSmartPointer<vtkIntArray>::New();
			data->SetName( fieldNames[BOUNDARY_NAME].c_str() );
			data->SetNumberOfComponents(1); data->SetNumberOfTuples(cells->GetNumberOfCells());
			for(elem_map::iterator i = bndrys.begin(); i!=bndrys.end(); ++i){
				data->SetTuple1(i->first - bndrys.begin()->first, bndryIDs[i->first] ); // convert to 0-based element index
			}
			bndMesh->GetCellData()->AddArray(data);

			data = vtkSmartPointer<vtkIntArray>::New();
			data->SetName( fieldNames[PARENT_NAME].c_str() );
			data->SetNumberOfComponents(1); data->SetNumberOfTuples(cells->GetNumberOfCells());
			for(elem_map::iterator i = bndrys.begin(); i!=bndrys.end(); ++i){
				data->SetTuple1(i->first - bndrys.begin()->first, bndryParents[i->first] ); // convert to 0-based element index
			}
			bndMesh->GetCellData()->AddArray(data);
		}
	}
	return n;
}

//const string fieldNames[] = {"body", "boundary", "parent"};
