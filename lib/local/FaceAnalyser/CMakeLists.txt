#TBB library
include_directories(${TBB_ROOT_DIR}/include)

#LandmarkDetector library
include_directories(../../local/LandmarkDetector/include)

#Utilities library
include_directories(../../local/Utilities/include)

SET(SOURCE
    src/Face_utils.cpp
	src/FaceAnalyser.cpp
	src/FaceAnalyserParameters.cpp
	src/PAW.cpp
	src/PDM.cpp		
	src/SVM_dynamic_lin.cpp
	src/SVM_static_lin.cpp
	src/SVR_dynamic_lin_regressors.cpp
	src/SVR_static_lin_regressors.cpp
)

SET(HEADERS
    include/Face_utils.h	
	include/FaceAnalyser.h
	include/FaceAnalyserParameters.h
	include/PAW.h
	include/PDM.h		
	include/SVM_dynamic_lin.h
	include/SVM_static_lin.h
	include/SVR_dynamic_lin_regressors.h
	include/SVR_static_lin_regressors.h
)

add_library( FaceAnalyser ${SOURCE} ${HEADERS})
add_library( OpenFace::FaceAnalyser ALIAS FaceAnalyser)

target_include_directories(FaceAnalyser PUBLIC ./include)
target_include_directories(FaceAnalyser PUBLIC ../../3rdParty/dlib/include)
target_include_directories(FaceAnalyser PRIVATE ../../3rdParty/OpenBLAS/include)
target_include_directories(FaceAnalyser PRIVATE ${FACEANALYSER_SOURCE_DIR}/include)
target_include_directories(FaceAnalyser PRIVATE ${BLAS_ROOT_DIR})
target_include_directories(FaceAnalyser PRIVATE ${BOOST_INCLUDE_DIR})

target_link_libraries(FaceAnalyser INTERFACE dlib)
target_link_libraries(FaceAnalyser INTERFACE ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TBB_LIBRARIES} ${BLAS_LIBRARIES})

install (TARGETS FaceAnalyser DESTINATION lib)
install (FILES ${HEADERS} DESTINATION include/OpenFace)
