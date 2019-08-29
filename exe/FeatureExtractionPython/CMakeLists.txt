# Local libraries
include_directories(${Python3_INCLUDE_DIRS})
link_directories(${Python3_LIBRARIES})
include_directories(${LandmarkDetector_SOURCE_DIR}/include)

# EXEC
#add_executable(OpenfaceVideo FeatureExtractionPython.cpp)
#target_link_libraries(OpenfaceVideo LandmarkDetector)
#target_link_libraries(OpenfaceVideo FaceAnalyser)
#target_link_libraries(OpenfaceVideo GazeAnalyser)
#target_link_libraries(OpenfaceVideo Utilities)
#target_link_libraries(OpenfaceVideo ${Boost_LIBRARIES})
#target_link_libraries(OpenfaceVideo ${Python3_LIBRARIES})
#install (TARGETS OpenfaceVideo DESTINATION bin)

# LIBRARY
add_library(PyOpenfaceVideo SHARED FeatureExtractionPython.cpp)
# set_property(TARGET PyOpenfaceVideo PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(PyOpenfaceVideo LandmarkDetector)
target_link_libraries(PyOpenfaceVideo FaceAnalyser)
target_link_libraries(PyOpenfaceVideo GazeAnalyser)
target_link_libraries(PyOpenfaceVideo Utilities)
target_link_libraries(PyOpenfaceVideo ${Boost_LIBRARIES})
target_link_libraries(PyOpenfaceVideo ${Python3_LIBRARIES})
set_target_properties(PyOpenfaceVideo PROPERTIES PREFIX "")
install (TARGETS PyOpenfaceVideo DESTINATION bin)