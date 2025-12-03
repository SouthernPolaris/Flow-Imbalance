cmake_minimum_required(VERSION 3.10)
project(flow_imbalance LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(flow_imbalance
    src/main/main.cpp
    src/main/OrderBook.cpp
    src/main/OFI.h
    src/main/Predictor.cpp
)

target_include_directories(flow_imbalance PRIVATE src)
target_compile_options(flow_imbalance PRIVATE -Wall -Wextra -Wpedantic -Werror)
find_package(Threads REQUIRED)
target_link_libraries(flow_imbalance PRIVATE Threads::Threads)