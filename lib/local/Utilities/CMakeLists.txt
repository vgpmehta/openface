
SET(SOURCE
    src/ImageCapture.cpp
	src/RecorderCSV.cpp
    src/RecorderHOG.cpp
	src/RecorderOpenFace.cpp
    src/RecorderOpenFaceParameters.cpp
	src/SequenceCapture.cpp
	src/VisualizationUtils.cpp
	src/Visualizer.cpp
)

SET(HEADERS
    include/ImageCapture.h	
    include/RecorderCSV.h
	include/RecorderHOG.h
    include/RecorderOpenFace.h
	include/RecorderOpenFaceParameters.h
	include/SequenceCapture.h
	include/VisualizationUtils.h
	include/Visualizer.h	
)

add_library( Utilities ${SOURCE} ${HEADERS})
add_library( OpenFace::Utilities ALIAS Utilities)

target_include_directories(Utilities PUBLIC ./include)
target_include_directories(Utilities PUBLIC ../../3rdParty/dlib/include)
target_include_directories(Utilities PRIVATE ${UTILITIES_SOURCE_DIR}/include)
target_include_directories(Utilities PRIVATE ${BOOST_INCLUDE_DIR})

target_link_libraries(Utilities INTERFACE dlib)
target_link_libraries(Utilities INTERFACE ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TBB_LIBRARIES} ${BLAS_LIBRARIES})

install (TARGETS Utilities DESTINATION lib)
install (FILES ${HEADERS} DESTINATION include/OpenFace)
