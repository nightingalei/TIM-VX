cmake_minimum_required (VERSION 3.14)
project(tim-vx)

OPTION(TIM_VX_ENABLE_TEST           "Build the unit test"               ON)
OPTION(TIM_VX_ENABLE_LAYOUT_INFER   "Enable layout inference support"   ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_FLAGS "-Wall -Wextra -Wno-unused-parameter -Wno-sign-compare -Werror -Wno-enum-conversion")
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-parameter -Wno-sign-compare -Werror")

if(NOT DEFINED CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "..." FORCE)
endif()

set(CMAKE_C_VISIBILITY_PRESET hidden)

if(EXTERNAL_VIV_SDK AND EXISTS ${EXTERNAL_VIV_SDK})
    # this is for internal development purpose
    include(cmake/local_sdk.cmake)
else()
    if("${CONFIG}" STREQUAL "A311D")
        include(cmake/A311D.cmake)
    elseif("${CONFIG}" STREQUAL "S905D3")
        include(cmake/S905D3.cmake)
    elseif("${CONFIG}" STREQUAL "vim3_android")
        include(cmake/vim3_android.cmake)
    else()
        include(cmake/X86_64_linux.cmake)
    endif()
endif()

if(TIM_VX_ENABLE_TEST)
    include(FetchContent)
    FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.10.0
    )

    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include/tim/vx)
include_directories(${OVXDRV_INCLUDE_DIRS})

add_subdirectory("src/tim/")
add_subdirectory("samples/lenet")
