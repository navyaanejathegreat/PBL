cmake_minimum_required(VERSION 3.16)
project(ewaste_cpp_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---- Eigen (header-only). Uses a local copy if EIGEN_DIR is given, else downloads 3.4.0.
set(EIGEN_DIR "" CACHE PATH "Folder containing Eigen/ (optional)")
if(EIGEN_DIR)
  set(EIGEN_INCLUDE ${EIGEN_DIR})
else()
  include(FetchContent)
  FetchContent_Declare(eigen
    GIT_REPOSITORY https://github.com/eigen-mirror/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE)
  FetchContent_GetProperties(eigen)
  if(NOT eigen_POPULATED)
    FetchContent_Populate(eigen)   # headers only, skip Eigen's own build/tests
  endif()
  set(EIGEN_INCLUDE ${eigen_SOURCE_DIR})
endif()

if(MSVC)
  add_compile_options(/O2 /EHsc /bigobj)
else()
  add_compile_options(-O2 -Wall)
endif()

add_library(nncore STATIC src/mlp.cpp src/data.cpp src/neurons.cpp src/learning_rules.cpp src/qc_metrics.cpp)
target_include_directories(nncore PUBLIC include ${EIGEN_INCLUDE})

add_executable(sandbox    src/sandbox_main.cpp)
add_executable(neurons    src/neurons_main.cpp)
add_executable(mlp_train  src/mlp_train_main.cpp)
add_executable(learning_rules src/learning_rules_main.cpp)
add_executable(qc_app     src/qc_app_main.cpp)
foreach(t sandbox neurons mlp_train learning_rules qc_app)
  target_link_libraries(${t} PRIVATE nncore)
endforeach()

enable_testing()
add_executable(unit_tests tests/unit_tests.cpp)
target_link_libraries(unit_tests PRIVATE nncore)
add_test(NAME unit_tests COMMAND unit_tests)
