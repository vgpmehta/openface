#TBB library
include_directories(${TBB_ROOT_DIR}/include)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt5 COMPONENTS Core Widgets REQUIRED)

if(WIN32)
  set(CMAKE_WIN32_EXECUTABLE ON)
endif()


#OpenBlas library
include_directories(../../lib/3rdParty/OpenBLAS/include)

# Local libraries
include_directories(${LandmarkDetector_SOURCE_DIR}/include)
	
include_directories(../../lib/local/LandmarkDetector/include)
include_directories(../../lib/local/FaceAnalyser/include)	
include_directories(../../lib/local/GazeAnalyser/include)	
include_directories(../../lib/local/Utilities/include)	
			
add_executable(openface-cross-platform-gui main.cpp render-thread.cpp OpenfaceWidget.cpp)

target_link_libraries(openface-cross-platform-gui LandmarkDetector)
target_link_libraries(openface-cross-platform-gui FaceAnalyser)
target_link_libraries(openface-cross-platform-gui GazeAnalyser)
target_link_libraries(openface-cross-platform-gui Utilities)
target_link_libraries(openface-cross-platform-gui dlib)
target_link_libraries(openface-cross-platform-gui ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TBB_LIBRARIES} ${BLAS_LIBRARIES})
target_link_libraries(openface-cross-platform-gui Qt5::Widgets)
install (TARGETS openface-cross-platform-gui DESTINATION bin)
