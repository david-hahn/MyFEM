#ifndef FIELD_NAMES_H
#define FIELD_NAMES_H

#include <string>

namespace MyFEM{	
	// names for data fields stored in VTK objects by readModel(vtk ...) and by LinearFEM, access as fieldNames[PARENT_NAME] etc.
	enum FIELD_NAMES{ BODY_NAME, BOUNDARY_NAME, PARENT_NAME, VELOCITY_NAME, FORCE_NAME, FORCE_EXT_NAME, VOLUME_NAME, AREA_NAME, CONTACT_FORCE_NAME, CONTACT_STATE_NAME };
	const std::string fieldNames[] = {"body", "boundary", "parent", "velocity", "force", "load", "volume", "area", "contactForce", "contactState"};

}

#endif // FIELD_NAMES_H
