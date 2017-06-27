# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")

find_package( Boost 1.5.9 REQUIRED COMPONENTS filesystem system python)

# Python stuff.
find_package(PythonLibs 2.7 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# TBB library.
include_directories(${TBB_ROOT_DIR}/include)

# Local libraries.
include_directories(${LandmarkDetector_SOURCE_DIR}/include)
  
include_directories(../lib/local/LandmarkDetector/include)
include_directories(../lib/local/FaceAnalyser/include)  
      
add_library(pyopenface SHARED pyopenface.cpp detector.cpp tracker.cpp)
target_link_libraries(pyopenface LandmarkDetectorShared)
target_link_libraries(pyopenface ${OpenCV_LIBS} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${TBB_LIBRARIES})
target_link_libraries(pyopenface dlib_shared)

set_property(TARGET pyopenface PROPERTY CXX_STANDARD 11)
set_property(TARGET pyopenface PROPERTY SUFFIX .so)
set_property(TARGET pyopenface PROPERTY PREFIX "")

install(TARGETS pyopenface LIBRARY DESTINATION /usr/lib/python2.7/dist-packages PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ GROUP_WRITE WORLD_WRITE WORLD_EXECUTE WORLD_READ)

