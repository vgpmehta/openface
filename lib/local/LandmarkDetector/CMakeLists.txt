#TBB library
include_directories(${TBB_ROOT_DIR}/include)

#Utilities library
include_directories(../../local/Utilities/include)

SET(SOURCE
    src/CCNF_patch_expert.cpp
	src/LandmarkDetectionValidator.cpp
    src/LandmarkDetectorFunc.cpp
	src/LandmarkDetectorModel.cpp
    src/LandmarkDetectorUtils.cpp
	src/LandmarkDetectorParameters.cpp
	src/Patch_experts.cpp
	src/PAW.cpp
    src/PDM.cpp
	src/SVR_patch_expert.cpp
	src/stdafx.cpp
)

SET(HEADERS
    include/CCNF_patch_expert.h	
    include/LandmarkCoreIncludes.h
	include/LandmarkDetectionValidator.h
    include/LandmarkDetectorFunc.h
	include/LandmarkDetectorModel.h
	include/LandmarkDetectorParameters.h
	include/LandmarkDetectorUtils.h
	include/Patch_experts.h	
    include/PAW.h
	include/PDM.h
	include/SVR_patch_expert.h		
	include/stdafx.h
)

add_library( LandmarkDetector ${SOURCE} ${HEADERS} )
add_library( OpenFace::LandmarkDetector ALIAS LandmarkDetector)

target_include_directories(LandmarkDetector PUBLIC ./include)
target_include_directories(LandmarkDetector PUBLIC ../../3rdParty/dlib/include)
target_include_directories(LandmarkDetector PRIVATE ${LandmarkDetector_SOURCE_DIR}/include)
target_include_directories(LandmarkDetector PRIVATE ${BOOST_INCLUDE_DIR})
target_include_directories(LandmarkDetector PRIVATE ../../3rdParty/OpenBLAS/include)

# suppress auto_ptr deprecation warnings
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(LandmarkDetector PUBLIC "-Wno-deprecated-declarations")
endif()

target_link_libraries(LandmarkDetector INTERFACE dlib)
target_link_libraries(LandmarkDetector INTERFACE ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TBB_LIBRARIES} ${BLAS_LIBRARIES})

install (TARGETS LandmarkDetector DESTINATION lib)
install (FILES ${HEADERS} DESTINATION include/OpenFace)
