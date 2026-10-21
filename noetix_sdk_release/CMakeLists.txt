cmake_minimum_required( VERSION 3.16.3 )

project( noetix-sdk )
set(USER_NAME $ENV{USER})
message(STATUS "environment user : ${USER_NAME}")
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_PREFIX_PATH /home/${USER_NAME}/.local/lib/python3.8/site-packages/pybind11)
set(pybind11_DIR /home/${USER_NAME}/.local/lib/python3.8/site-packages/pybind11/share/cmake/pybind11/pybind11Config.cmake)
# 找到 Python3 的库文件路径
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm.*|arm|aarch64")
  set(PYTHON_LIB_PATH /usr/lib/python3.8/config-3.8-aarch64-linux-gnu)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64)|(^i686)")
  set(PYTHON_LIB_PATH /usr/lib/python3.8/config-3.8-x86_64-linux-gnu)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # global PIC

find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(pybind11 REQUIRED)
find_package(spdlog REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(onnxruntime REQUIRED)
add_compile_definitions(SDK_ROOT="${CMAKE_BINARY_DIR}/..")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm.*|arm|aarch64")
    set(LIB_DIR ${PROJECT_SOURCE_DIR}/lib/aarch64)
    # The prebuilt SDK still refers to RKNN symbols; policy inference uses ONNX Runtime.
    set(RKNN_COMPAT_LIBRARY
        "${PROJECT_SOURCE_DIR}/../rknpu2v2.1.0/runtime/Linux/librknn_api/aarch64/librknnrt.so")
    if(NOT EXISTS "${RKNN_COMPAT_LIBRARY}")
      unset(RKNN_COMPAT_LIBRARY)
      find_library(RKNN_COMPAT_LIBRARY NAMES rknnrt)
    endif()
    if(NOT RKNN_COMPAT_LIBRARY OR NOT EXISTS "${RKNN_COMPAT_LIBRARY}")
      message(FATAL_ERROR "ARM64 SDK requires librknnrt.so for linking; install it or keep the bundled rknpu2v2.1.0 directory")
    endif()

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64)|(^i686)")
    set(LIB_DIR ${PROJECT_SOURCE_DIR}/lib/x86_64)
endif()
include_directories(/usr/include/python3.8)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(/home/${USER_NAME}/.local/lib/python3.8/site-packages/pybind11/include)

MESSAGE("LIB_DIR:${LIB_DIR}")


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
add_executable(noetix_joint_controller src/controller_example.cpp src/usercontroller.cpp)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm.*|arm|aarch64")
  target_link_libraries(noetix_joint_controller 
      PRIVATE pybind11::module 
      Python3::Python
      ${LIB_DIR}/librlcontrol.a
      ${LIB_DIR}/libhardware.a
      ${LIB_DIR}/libsoem.a
      ${LIB_DIR}/libyaml-cppd.a
      onnxruntime
      Threads::Threads
      spdlog::spdlog
      ${RKNN_COMPAT_LIBRARY}
  )
  set_target_properties(noetix_joint_controller PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
  add_custom_command(TARGET noetix_joint_controller POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${RKNN_COMPAT_LIBRARY}" "$<TARGET_FILE_DIR:noetix_joint_controller>/librknnrt.so")
else()
  target_link_libraries(noetix_joint_controller 
      PRIVATE pybind11::module 
      Python3::Python
      ${LIB_DIR}/librlcontrol.a
      ${LIB_DIR}/libhardware.a
      ${LIB_DIR}/libsoem.a
      ${LIB_DIR}/libyaml-cppd.a
      onnxruntime
      Threads::Threads
      spdlog::spdlog
  )
endif()



