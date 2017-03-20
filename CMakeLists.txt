project (bohme_shading_constaint_filter)

cmake_minimum_required(VERSION 3.0)
cmake_policy(VERSION 3.0)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE )

set(TIFF_INCLUDE_DIR "include" CACHE PATH "include dir for tiff library")
set(TIFF_LIBRARY_RELEASE "lib/tiff.lib" CACHE FILEPATH "release lib for tiff library")
set(TIFF_LIBRARY_DEBUG "lib/tiffd.lib" CACHE  FILEPATH "debug lib for tiff library")
find_package(OpenCV 3 REQUIRED)

# Source files
SET(FAST_BIL_SOURCES
    src/bohme_shading_constaint_filter/main.cpp
	src/bohme_shading_constaint_filter/bohme_shading_constraint_filter.h
	src/bohme_shading_constaint_filter/bohme_shading_constraint_filter.cpp
	src/bohme_shading_constaint_filter/normal_computer_grad.h
	src/bohme_shading_constaint_filter/normal_computer_grad.cpp
	src/bohme_shading_constaint_filter/single_frame.h
	src/bohme_shading_constaint_filter/single_frame.cpp
	src/bohme_shading_constaint_filter/stdafx.h
	src/bohme_shading_constaint_filter/stdafx.cpp
	src/bohme_shading_constaint_filter/targetver.h
)

include_directories(${OpenCV_INCLUDE_DIRS} ${TIFF_INCLUDE_DIR})

# Create the executable
add_executable(bohme_shading_constaint_filter ${FAST_BIL_SOURCES})
set(LINK_LIBRARY debug ${TIFF_LIBRARY_DEBUG} optimized ${TIFF_LIBRARY_RELEASE} ${OpenCV_LIBS})

target_link_libraries(bohme_shading_constaint_filter ${LINK_LIBRARY})