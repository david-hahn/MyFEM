project (MyFEM)
cmake_minimum_required (VERSION 3.0)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	set(CMAKE_INSTALL_PREFIX ".." CACHE PATH "comment" FORCE)
endif()

if(MSVC)
	# stuff for Visual Studio compiler
	add_definitions(-TP) # force MSVC to use C++ compiler, otherwise it falls back to ANSI C
	add_definitions(-D_USE_MATH_DEFINES) # make sure we have M_PI available
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp") # enable OpenMP
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W0") # get rid of warnings
else()
	# stuff for GNU compiler
	add_definitions(-D_USE_MATH_DEFINES) # make sure we have M_PI available
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp") # enable OpenMP
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w") # get rid of warnings
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS             "${CMAKE_C_FLAGS}             ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS           "${CMAKE_CXX_FLAGS}           ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS}    ${OpenMP_EXE_LINKER_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${OpenMP_MODULE_LINKER_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS}")
    set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} ${OpenMP_STATIC_LINKER_FLAGS}")
endif()

# using VTK for mesh and data storage, interpolation and output
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

# using Eigen for linear algebra
#find_path(EIGEN_INC Eigen/Eigen)
#include_directories (${EIGEN_INC})
find_package(Eigen3 REQUIRED NO_MODULE)

OPTION(USE_OPTIMLIB "Enable optimlib support (only used for tests and comparisons)" OFF)
if(USE_OPTIMLIB)
 # optimlib header-only version
 include_directories(${PROJECT_SOURCE_DIR}/optimlib)
 find_library(BLAS openblas libopenblas PATH_SUFFIXES OpenBLAS/lib)
 add_definitions(-D_USE_OPTIMLIB)
endif()

# using MMGtools for tet-remeshing
find_library(MMG_LIB mmg PATH_SUFFIXES MMG-5-3/lib)
find_path(MMG_INC "mmg/libmmg.h"  PATH_SUFFIXES MMG-5-3/include)
include_directories (${MMG_INC})

OPTION(USE_OSQP "Enable OSQP support (only used QP contact method)" OFF)
if(USE_OSQP)
 # using OSQP and corresponding Eigen interface to solve quadratic optimization problems
 find_path(OSQP_EIGEN_INC "OsqpEigen/OsqpEigen.h")
 include_directories (${OSQP_EIGEN_INC})
 find_library(OSQP_EIGEN_LIB OsqpEigen)
 find_path(OSQP_INC "osqp/osqp.h")
 include_directories (${OSQP_INC}/osqp)
 find_library(OSQP_LIB osqp)
 add_definitions(-D_USE_OSQP)
endif()

# try mosek for QP solving instead ...
# find_path(MOSEK_INC "mosek.h")
# include_directories (${MOSEK_INC})
# find_library(MOSEK_LIB mosek64_8_1)

# also try MatLab Engine API for QP solving ...
# find_path(MATLAB_INC "MatlabEngine.hpp")
# include_directories (${MATLAB_INC})
# find_library(MATLAB_ENGINE_LIB libMatlabEngine)
# find_library(MATLAB_DATA_LIB libMatlabDataArray)


# local includes
include_directories(${PROJECT_SOURCE_DIR}/src)

FILE(GLOB SRC_HEADERS ${PROJECT_SOURCE_DIR}/src/*.h) # for convenience -> puts the headers into the MSVC project's file list
add_executable(${PROJECT_NAME} ${SRC_HEADERS}
  "src/main.cpp"
  "src/main_contact.cpp"
  "src/main_YuMiData.cpp"
  "src/main_Xworm.cpp"
  "src/main_CylinderMarkers.cpp"
  "src/main_CylinderMarkers_dualPhi.cpp"
  "src/main_TPU_lattices.cpp"
  "src/main_PointCloudDemo.cpp"
  "src/ElmerReader.cpp"
  "src/LinearFEM.cpp"
  "src/ContactFEM.cpp"
  "src/Materials.cpp"
  "src/Remesher.cpp"
  "src/AdjointSensitivity.cpp"
  "src/DirectSensitivity.cpp"
  "src/BoundaryFieldObjectiveFunction.cpp"
  "src/ElasticMaterialParameterHandler.cpp"
  "src/ViscousMaterialParameterHandler.cpp"
  "src/PrincipalStretchMaterial.cpp"
  "src/EigenmodeViscosityModel.cpp"
  "src/TemporalInterpolationField.cpp"
  "src/DifferentiableSpline.cpp"
  "src/PointCloudObjectiveFunction.cpp"
  "src/InitialConditionParameterHandler.cpp"
  "src/FrictionParameterHandler.cpp"
  "src/cmaes/cmaes.c"
  "src/cmaes/CMAMinimizer.cpp"
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17) # mostly for std::filesystem access

target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME} ${VTK_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${MMG_LIB})
if(USE_OSQP)
 target_link_libraries(${PROJECT_NAME} ${OSQP_EIGEN_LIB} ${OSQP_LIB})
endif()
if(USE_OPTIMLIB)
 target_link_libraries(${PROJECT_NAME} ${BLAS})
endif()
# target_link_libraries(${PROJECT_NAME} ${MOSEK_LIB})
# target_link_libraries(${PROJECT_NAME} ${MATLAB_ENGINE_LIB} ${MATLAB_DATA_LIB})
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
