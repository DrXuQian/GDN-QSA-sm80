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

add_executable(l004_ppu_affine_pipeline dev/ppu/l004_affine_pipeline.cpp)
add_executable(l006_ppu_original_delivery dev/ppu/l006_original_kernel_delivery.cpp)
target_include_directories(l006_ppu_original_delivery PRIVATE
  ${PPU_SDK_ROOT}/include
  ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include
  ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include/cccl
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/actlize/include)
target_link_libraries(l006_ppu_original_delivery PRIVATE gdn_qsa_backend_headers)
add_executable(l007_ppu_wy_ownership dev/ppu/l007_wy_ownership.cpp)
target_include_directories(l007_ppu_wy_ownership PRIVATE
  ${PPU_SDK_ROOT}/include ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include
  ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include/cccl
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/actlize/include)
target_link_libraries(l007_ppu_wy_ownership PRIVATE gdn_qsa_backend_headers)
add_executable(l008_ppu_wy_delivery dev/ppu/l008_wy_delivery.cpp)
target_include_directories(l008_ppu_wy_delivery PRIVATE
  ${PPU_SDK_ROOT}/include ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include
  ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include/cccl
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/actlize/include)
target_link_libraries(l008_ppu_wy_delivery PRIVATE gdn_qsa_backend_headers)
add_executable(l009_ppu_wy_tiles dev/ppu/l009_wy_tiles.cpp)
target_include_directories(l009_ppu_wy_tiles PRIVATE
  ${PPU_SDK_ROOT}/include ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include
  ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include/cccl
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/actlize/include)
target_link_libraries(l009_ppu_wy_tiles PRIVATE gdn_qsa_backend_headers)
add_executable(l010_ppu_wy_state_address dev/ppu/l010_wy_state_address.cpp)
target_include_directories(l010_ppu_wy_state_address PRIVATE
  ${PPU_SDK_ROOT}/include ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include
  ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include/cccl
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/actlize/include)
target_link_libraries(l010_ppu_wy_state_address PRIVATE gdn_qsa_backend_headers)

if(BUILD_TESTING)
  add_test(NAME ppu_affine_pipeline COMMAND l004_ppu_affine_pipeline)
  add_test(NAME ppu_original_delivery COMMAND l006_ppu_original_delivery)
  add_test(NAME ppu_wy_ownership COMMAND l007_ppu_wy_ownership)
  add_test(NAME ppu_wy_delivery COMMAND l008_ppu_wy_delivery)
  add_test(NAME ppu_wy_tiles COMMAND l009_ppu_wy_tiles)
  add_test(NAME ppu_wy_state_address COMMAND l010_ppu_wy_state_address)
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
  include(cmake/GdnHgccArch.cmake)
  gdn_hgcc_arch_flags(CUTLASS_PPU_EXTRA_HGCC_FLAGS _gdn_arch_contract
    "${PPU_DEVICE_HGCC_REAL}" ${CUTLASS_PPU_EXTRA_HGCC_FLAGS})
  list(APPEND CUTLASS_PPU_DEV_INCLUDE_FLAGS
       "-I${CMAKE_CURRENT_SOURCE_DIR}/include")
  list(APPEND CUTLASS_PPU_EXTRA_HGCC_FLAGS "-DGDN_QSA_PPU=1")

  cutlass_add_library(
    gdn_qsa_ppu SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gdn_chunk/gdn_kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gdn_chunk/gdn_scan_stage1.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gdn_chunk/gdn_scan_stage1_reset.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gdn_chunk/gdn_scan_stage2.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gdn_chunk/gdn_scan_stage2_blelloch.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gdn_chunk/gdn_scan_stage3_reset.cu)
  # actlize's hgcc custom rule tracks its .cu only (no header depfile).
  # Track our target adapter explicitly: a header-only correctness repair
  # must not leave an old device object in an otherwise rebuilt library.
  get_target_property(_gdn_objects gdn_qsa_ppu SOURCES)
  foreach(_object IN LISTS _gdn_objects)
    if(_object MATCHES "[.]o$")
      add_custom_command(OUTPUT "${_object}" APPEND DEPENDS
        "${_gdn_arch_contract}"
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gdn_chunk/gdn_target.cuh
        ${CMAKE_CURRENT_SOURCE_DIR}/include/gdn_qsa/ppu/fragment.cuh
        ${CMAKE_CURRENT_SOURCE_DIR}/include/gdn_qsa/ppu/shared_copy.cuh)
    endif()
  endforeach()
  target_include_directories(
    gdn_qsa_ppu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
  target_link_libraries(gdn_qsa_ppu PRIVATE CUTLASS ppu::runtime)
  set_target_properties(gdn_qsa_ppu PROPERTIES OUTPUT_NAME gdn_qsa_ppu)

  # Reuse the ORIGINAL host dispatch as well as its kernels. No Python copy of
  # reset admission / scan selection / auto-dispatch is maintained for PPU.
  # This TU contains no device code: compile as C++ against the box's PyTorch.
  # Do not use TorchConfig's NVIDIA CUDA compiler discovery for this PPU build.
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
      "import json,torch; from torch.utils.cpp_extension import include_paths,library_paths; print(json.dumps({'includes':include_paths(),'libraries':library_paths(),'abi':int(torch._C._GLIBCXX_USE_CXX11_ABI)}))"
    OUTPUT_VARIABLE _torch_json OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _torch_status)
  if(NOT _torch_status EQUAL 0)
    message(FATAL_ERROR "PPU Python binding requires the installed PyTorch")
  endif()
  string(JSON _torch_abi GET "${_torch_json}" abi)
  foreach(_kind includes libraries)
    string(JSON _count LENGTH "${_torch_json}" ${_kind})
    math(EXPR _last "${_count}-1")
    foreach(_index RANGE 0 ${_last})
      string(JSON _entry GET "${_torch_json}" ${_kind} ${_index})
      list(APPEND _torch_${_kind} "${_entry}")
    endforeach()
  endforeach()
  set(_host_ops ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gdn_chunk/gdn_ops.cu)
  set_source_files_properties(${_host_ops} PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-x c++")
  Python3_add_library(_gdn_chunk_ppu MODULE WITH_SOABI ${_host_ops})
  target_compile_definitions(_gdn_chunk_ppu PRIVATE
    TORCH_EXTENSION_NAME=_gdn_chunk_ppu
    _GLIBCXX_USE_CXX11_ABI=${_torch_abi})
  target_include_directories(_gdn_chunk_ppu PRIVATE
    ${_torch_includes}
    ${PPU_SDK_ROOT}/include
    ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include
    ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include/cccl
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/actlize/include)
  target_link_directories(_gdn_chunk_ppu PRIVATE ${_torch_libraries})
  target_link_libraries(_gdn_chunk_ppu PRIVATE
    gdn_qsa_ppu ppu::runtime torch_python torch torch_cpu torch_cuda c10 c10_cuda)
  set_target_properties(_gdn_chunk_ppu PROPERTIES
    BUILD_RPATH "${_torch_libraries};$ORIGIN"
    INSTALL_RPATH "$ORIGIN")

  # An opt-in algorithm candidate, not a rewrite of upstream dispatch. Keep
  # its device/host symbols and resource audit separate from the control.
  cutlass_add_library(gdn_wy_ppu SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gdn_chunk/gdn_wy_ppu.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gdn_chunk/gdn_wy_tiles_ppu.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gdn_chunk/gdn_wy_state_ab_ppu.cu)
  target_include_directories(gdn_wy_ppu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(gdn_wy_ppu PRIVATE CUTLASS ppu::runtime)
  get_target_property(_wy_objects gdn_wy_ppu SOURCES)
  foreach(_object IN LISTS _wy_objects)
    if(_object MATCHES "[.]o$")
      add_custom_command(OUTPUT "${_object}" APPEND DEPENDS
        "${_gdn_arch_contract}"
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gdn_chunk/gdn_target.cuh
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gdn_chunk/gdn_wy_common.cuh
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gdn_chunk/gdn_wy_prepare.cuh
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gdn_chunk/gdn_wy_state_copy.cuh
        ${CMAKE_CURRENT_SOURCE_DIR}/include/gdn_qsa/wy_contract.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/gdn_qsa/ppu/wy_mma.cuh
        ${CMAKE_CURRENT_SOURCE_DIR}/include/gdn_qsa/ppu/wy_delivery.cuh
        ${CMAKE_CURRENT_SOURCE_DIR}/include/gdn_qsa/ppu/wy_tiles.cuh
        ${CMAKE_CURRENT_SOURCE_DIR}/include/gdn_qsa/ppu/wy_state_address.cuh
        ${CMAKE_CURRENT_SOURCE_DIR}/include/gdn_qsa/ppu/shared_copy.cuh
        ${CMAKE_CURRENT_SOURCE_DIR}/include/gdn_qsa/ppu/fragment.cuh)
    endif()
  endforeach()
  Python3_add_library(_gdn_wy_ppu MODULE WITH_SOABI csrc/gdn_chunk/gdn_wy_ops.cpp)
  target_compile_definitions(_gdn_wy_ppu PRIVATE TORCH_EXTENSION_NAME=_gdn_wy_ppu
    _GLIBCXX_USE_CXX11_ABI=${_torch_abi})
  target_include_directories(_gdn_wy_ppu PRIVATE ${_torch_includes}
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${PPU_SDK_ROOT}/include
    ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include
    ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include/cccl)
  target_link_directories(_gdn_wy_ppu PRIVATE ${_torch_libraries})
  target_link_libraries(_gdn_wy_ppu PRIVATE gdn_wy_ppu ppu::runtime
    torch_python torch torch_cpu torch_cuda c10 c10_cuda)
  set_target_properties(_gdn_wy_ppu PROPERTIES
    BUILD_RPATH "${_torch_libraries};$ORIGIN" INSTALL_RPATH "$ORIGIN")
endif()
