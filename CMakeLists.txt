cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(gdn_qsa_backends LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)

option(GDN_QSA_ENABLE_PPU "Build the native PPU GDN backend" OFF)

set(
  PPU_SDK_ROOT
  "$ENV{PPU_SDK}"
  CACHE PATH "PPU SDK root used by PPU compile and ownership gates")
if(NOT PPU_SDK_ROOT AND
   EXISTS "/root/autodl-tmp/root-relocated/ppu-sdk/2.1.1")
  set(PPU_SDK_ROOT "/root/autodl-tmp/root-relocated/ppu-sdk/2.1.1")
endif()

add_library(gdn_qsa_backend_headers INTERFACE)
target_include_directories(
  gdn_qsa_backend_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(
  l001_ppu_superchunk_schedule dev/ppu/l001_superchunk_schedule.cpp)
target_link_libraries(
  l001_ppu_superchunk_schedule PRIVATE gdn_qsa_backend_headers)

add_executable(l002_ppu_warp_mma_ownership dev/ppu/l002_warp_mma_ownership.cpp)
target_include_directories(
  l002_ppu_warp_mma_ownership PRIVATE
  ${PPU_SDK_ROOT}/include
  ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include
  ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include/cccl
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/actlize/include)
target_link_libraries(
  l002_ppu_warp_mma_ownership PRIVATE gdn_qsa_backend_headers)

add_executable(l004_ppu_affine_pipeline dev/ppu/l004_affine_pipeline.cpp)
add_executable(l005_ppu_backend_c_abi dev/ppu/l005_backend_c_abi.c)
target_link_libraries(l005_ppu_backend_c_abi PRIVATE gdn_qsa_backend_headers)

if(BUILD_TESTING)
  add_test(NAME ppu_superchunk_schedule COMMAND l001_ppu_superchunk_schedule)
  add_test(NAME ppu_warp_mma_ownership COMMAND l002_ppu_warp_mma_ownership)
  add_test(NAME ppu_affine_pipeline COMMAND l004_ppu_affine_pipeline)
  add_test(NAME ppu_backend_c_abi COMMAND l005_ppu_backend_c_abi)
endif()

if(GDN_QSA_ENABLE_PPU)
  if(NOT PPU_SDK_ROOT)
    message(FATAL_ERROR "GDN_QSA_ENABLE_PPU requires PPU_SDK_ROOT or PPU_SDK")
  endif()
  set(CUTLASS_PPU_ARCHS ppu0010 CACHE STRING "PPU architectures" FORCE)
  set(CUTLASS_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(CUTLASS_ENABLE_TOOLS OFF CACHE BOOL "" FORCE)
  set(CUTLASS_ENABLE_LIBRARY OFF CACHE BOOL "" FORCE)
  set(CUTLASS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  set(CUTLASS_ENABLE_GTEST_UNIT_TESTS OFF CACHE BOOL "" FORCE)
  set(PPU_SDK_ROOT "${PPU_SDK_ROOT}" CACHE PATH "" FORCE)

  add_subdirectory(third_party/actlize actlize)
  get_directory_property(
    _actlize_hgcc_flags
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/actlize
    DEFINITION CUTLASS_PPU_EXTRA_HGCC_FLAGS)
  get_directory_property(
    _actlize_device_includes
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/actlize
    DEFINITION CUTLASS_PPU_DEV_INCLUDE_FLAGS)
  get_directory_property(
    _actlize_hgcc
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/actlize
    DEFINITION PPU_DEVICE_HGCC_REAL)
  set(CUTLASS_PPU_EXTRA_HGCC_FLAGS ${_actlize_hgcc_flags})
  set(CUTLASS_PPU_DEV_INCLUDE_FLAGS ${_actlize_device_includes})
  set(PPU_DEVICE_HGCC_REAL ${_actlize_hgcc})
  list(APPEND CUTLASS_PPU_DEV_INCLUDE_FLAGS
       "-I${CMAKE_CURRENT_SOURCE_DIR}/include")

  cutlass_add_library(
    gdn_qsa_ppu SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gdn_chunk_ppu/gdn_pipeline_ppu.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gdn_chunk_ppu/gdn_scan_stage2_ppu.cu)
  target_include_directories(
    gdn_qsa_ppu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
  target_link_libraries(gdn_qsa_ppu PRIVATE CUTLASS ppu::runtime)
  set_target_properties(gdn_qsa_ppu PROPERTIES OUTPUT_NAME gdn_qsa_ppu)
endif()
