# Local libraries
include_directories(${LandmarkDetector_SOURCE_DIR}/include)
include_directories(${ZeroMQ_INCLUDE_DIR})

add_executable(AuExtractionMQ AuExtractionMQ.cpp)
target_link_libraries(AuExtractionMQ LandmarkDetector)
target_link_libraries(AuExtractionMQ FaceAnalyser)
target_link_libraries(AuExtractionMQ GazeAnalyser)
target_link_libraries(AuExtractionMQ Utilities)
target_link_libraries(AuExtractionMQ  ${ZeroMQ_LIBRARY})

message(STATUS "AuExtractionMQ :: C++ZMQ Library: " ${ZeroMQ_LIBRARY} )
install (TARGETS AuExtractionMQ DESTINATION bin)
