cmake_minimum_required(VERSION 3.10)
project(flow_imbalance LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(flow_imbalance
    src/main/main.cpp
    src/main/OrderBook.cpp
    src/main/OFI.h
    src/main/predictor/Predictor.cpp
)

target_include_directories(flow_imbalance PRIVATE src)
target_compile_options(flow_imbalance PRIVATE -Wall -Wextra -Wpedantic -Werror)
find_package(Threads REQUIRED)
target_link_libraries(flow_imbalance PRIVATE Threads::Threads)

# Optional OpenCL GPU support. Enable with -DBUILD_WITH_OPENCL=ON
option(BUILD_WITH_OPENCL "Enable OpenCL GPU support for Predictor (optional)" OFF)
if(BUILD_WITH_OPENCL)
    find_package(OpenCL REQUIRED)
    if(OpenCL_FOUND)
        target_compile_definitions(flow_imbalance PRIVATE BUILD_WITH_OPENCL=1)
        target_include_directories(flow_imbalance PRIVATE ${OpenCL_INCLUDE_DIRS})
        target_link_libraries(flow_imbalance PRIVATE ${OpenCL_LIBRARIES})
    else()
        message(WARNING "BUILD_WITH_OPENCL requested but OpenCL not found. Falling back to CPU-only build.")
    endif()
endif()