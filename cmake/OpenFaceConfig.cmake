include(CMakeFindDependencyMacro)
find_dependency(dlib 18.18)
find_dependency(OpenCV 3.0)
find_dependency(Boost)

include("${CMAKE_CURRENT_LIST_DIR}/OpenFaceTargets.cmake")

# Compute paths
get_filename_component(OpenFace_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

if(${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} VERSION_LESS 2.8)
    get_filename_component(OpenFace_INSTALL_PATH "${OpenFace_CMAKE_DIR}/../../../" ABSOLUTE)
else()
    get_filename_component(OpenFace_INSTALL_PATH "${OpenFace_CMAKE_DIR}/../../../" REALPATH)
endif()

set(OpenFace_INCLUDE_DIRS "${OpenFace_INSTALL_PATH}/include/OpenFace;${dlib_INCLUDE_DIRS};${Boost_INCLUDE_DIRS};${Boost_INCLUDE_DIRS}/boost")

