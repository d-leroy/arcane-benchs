cmake_minimum_required(VERSION 3.21)
project(MicroHydro LANGUAGES C CXX)

find_package(Arcane REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

add_executable(MicroHydro MicroHydroModule.cc main.cc MicroHydro_axl.h AdditionalVariablesModule.cc AdditionalVariables_axl.h
  connectivix/CSR.cc
  connectivix/Metadata.cc
  connectivix/ConnectivityMatMul.cc
  connectivix/ConnectivityMatMulBinning.cc
  connectivix/ConnectivityMatMulSymbolic.cc
  connectivix/ConnectivityMatMulNumeric.cc)

arcane_generate_axl(MicroHydro)
arcane_generate_axl(AdditionalVariables)
arcane_add_arcane_libraries_to_target(MicroHydro)
target_include_directories(MicroHydro PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(MicroHydro PRIVATE ../include/)
configure_file(MicroHydro.config ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file(MicroHydro.arc ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

# Partie spécifique accélérateur
arcane_accelerator_enable()
arcane_accelerator_add_source_files(MicroHydroModule.cc)
arcane_accelerator_add_source_files(MicroHydroModule.cc)
arcane_accelerator_add_to_target(MicroHydro)

target_link_libraries(MicroHydro PRIVATE /home/leroyd/git/cuBool/build/cubool/libcubool.so)

enable_testing()
# Ajout des tests
add_test(NAME microhydro COMMAND ./MicroHydro -A,MaxIteration=50 MicroHydro.arc)
if (ARCANE_HAS_CUDA)
  add_test(NAME microhydro_cuda COMMAND ./MicroHydro -A,MaxIteration=50 -A,AcceleratorRuntime=cuda MicroHydro.arc)
endif()
