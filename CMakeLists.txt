# CMakeLists.txt for CIS6007 Parallel Image Processing Project
# Student ID: ST20290339

cmake_minimum_required(VERSION 3.16)
project(CIS6007_ParallelImageProcessing VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set build type (Release for optimization, Debug for debugging)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags (platform-specific)
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od /Zi")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()

# Find OpenCV
# Accept OpenCV_DIR as a command-line argument (-DOpenCV_DIR=...)
# If not set, CMake will search standard system paths.
if(NOT DEFINED OpenCV_DIR)
    # Provide a hint for common install locations (override with -DOpenCV_DIR=...)
    # On Windows with vcpkg or manual installs, you may need:
    #   cmake -B build -DOpenCV_DIR="C:/path/to/opencv/build"
    message(STATUS "OpenCV_DIR not set. CMake will search default paths.")
endif()
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV.")
endif()

# Optional: Find MPI (uncomment to enable MPI support)
# find_package(MPI REQUIRED)
# if(MPI_FOUND)
#     message(STATUS "MPI found")
#     add_definitions(-DUSE_MPI)
# else()
#     message(WARNING "MPI not found. MPI support will be disabled.")
# endif()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})

# Source files
set(COMMON_SOURCES
    ${PROJECT_SOURCE_DIR}/src/common.cpp
    ${PROJECT_SOURCE_DIR}/src/algorithm1.cpp
    ${PROJECT_SOURCE_DIR}/src/algorithm2.cpp
    ${PROJECT_SOURCE_DIR}/src/algorithm3.cpp
    ${PROJECT_SOURCE_DIR}/src/algorithm4.cpp
    ${PROJECT_SOURCE_DIR}/src/algorithm5.cpp
    ${PROJECT_SOURCE_DIR}/src/parallel_utils.cpp
    ${PROJECT_SOURCE_DIR}/src/knn_classifier.cpp
)

# Create common library
add_library(common_lib STATIC ${COMMON_SOURCES})
target_link_libraries(common_lib ${OpenCV_LIBS})
# if(MPI_FOUND)
#     target_link_libraries(common_lib ${OpenCV_LIBS} ${MPI_LIBRARIES})
# endif()

# Main executable
add_executable(main ${PROJECT_SOURCE_DIR}/src/main.cpp)
target_link_libraries(main common_lib ${OpenCV_LIBS})

# KNN Demo executable
add_executable(knn_demo ${PROJECT_SOURCE_DIR}/src/knn_demo.cpp)
target_link_libraries(knn_demo common_lib ${OpenCV_LIBS})
# if(MPI_FOUND)
#     target_link_libraries(main common_lib ${OpenCV_LIBS} ${MPI_LIBRARIES} pthread)
# endif()

# Test executables
add_executable(test_algorithm1 ${PROJECT_SOURCE_DIR}/tests/test_algorithm1.cpp)
target_link_libraries(test_algorithm1 common_lib ${OpenCV_LIBS})

add_executable(test_algorithm2 ${PROJECT_SOURCE_DIR}/tests/test_algorithm2.cpp)
target_link_libraries(test_algorithm2 common_lib ${OpenCV_LIBS})

add_executable(test_algorithm3 ${PROJECT_SOURCE_DIR}/tests/test_algorithm3.cpp)
target_link_libraries(test_algorithm3 common_lib ${OpenCV_LIBS})

add_executable(test_algorithm4 ${PROJECT_SOURCE_DIR}/tests/test_algorithm4.cpp)
target_link_libraries(test_algorithm4 common_lib ${OpenCV_LIBS})

add_executable(test_algorithm5 ${PROJECT_SOURCE_DIR}/tests/test_algorithm5.cpp)
target_link_libraries(test_algorithm5 common_lib ${OpenCV_LIBS})

add_executable(test_knn ${PROJECT_SOURCE_DIR}/tests/test_knn.cpp)
target_link_libraries(test_knn common_lib ${OpenCV_LIBS})

# Enable testing
enable_testing()

# Add tests
add_test(NAME test_algorithm1 COMMAND test_algorithm1)
add_test(NAME test_algorithm2 COMMAND test_algorithm2)
add_test(NAME test_algorithm3 COMMAND test_algorithm3)
add_test(NAME test_algorithm4 COMMAND test_algorithm4)
add_test(NAME test_algorithm5 COMMAND test_algorithm5)
add_test(NAME test_knn COMMAND test_knn)
set_tests_properties(test_knn PROPERTIES WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

# Installation rules
install(TARGETS main DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

# Print configuration summary
message(STATUS "")
message(STATUS "========================================")
message(STATUS "Configuration Summary")
message(STATUS "========================================")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
# if(MPI_FOUND)
#     message(STATUS "MPI support: ENABLED")
# else()
#     message(STATUS "MPI support: DISABLED")
# endif()
message(STATUS "========================================")
message(STATUS "")
