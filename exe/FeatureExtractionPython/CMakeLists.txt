# Local libraries
include_directories(${PYTHON_INCLUDE_DIRS})
link_directories(${PYTHON_LIBRARIES})
include_directories(${LandmarkDetector_SOURCE_DIR}/include)

# EXEC
add_executable(OpenfaceVideo FeatureExtractionPython.cpp)
target_link_libraries(OpenfaceVideo LandmarkDetector)
target_link_libraries(OpenfaceVideo FaceAnalyser)
target_link_libraries(OpenfaceVideo GazeAnalyser)
target_link_libraries(OpenfaceVideo Utilities)
target_link_libraries(OpenfaceVideo ${PYTHON_LIBRARIES})
install (TARGETS OpenfaceVideo DESTINATION bin)

# LIBRARY
add_library(PyOpenfaceVideo SHARED FeatureExtractionPython.cpp)
target_link_libraries(PyOpenfaceVideo LandmarkDetector)
target_link_libraries(PyOpenfaceVideo FaceAnalyser)
target_link_libraries(PyOpenfaceVideo GazeAnalyser)
target_link_libraries(PyOpenfaceVideo Utilities)
target_link_libraries(PyOpenfaceVideo ${PYTHON_LIBRARIES})
install (TARGETS PyOpenfaceVideo DESTINATION bin)