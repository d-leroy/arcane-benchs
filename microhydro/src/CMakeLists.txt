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
  connectivix/ConnectivityMatMulNumeric.cc
  connectivix/ConnectivityEWiseMatMul.cc
  connectivix/ConnectivityEWiseMatSub.cc
  connectivix/ConnectivityTranspose.cc)

arcane_generate_axl(MicroHydro)
arcane_generate_axl(AdditionalVariables)
arcane_add_arcane_libraries_to_target(MicroHydro)
target_include_directories(MicroHydro PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(MicroHydro PRIVATE ../include/)
configure_file(MicroHydro.config ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

# Partie spécifique accélérateur
arcane_accelerator_enable()
arcane_accelerator_add_source_files(connectivix/ConnectivityMatMul.cc)
arcane_accelerator_add_source_files(connectivix/ConnectivityMatMulBinning.cc)
arcane_accelerator_add_source_files(connectivix/ConnectivityMatMulSymbolic.cc)
arcane_accelerator_add_source_files(connectivix/ConnectivityMatMulNumeric.cc)
arcane_accelerator_add_source_files(connectivix/ConnectivityEWiseMatMul.cc)
arcane_accelerator_add_source_files(connectivix/ConnectivityEWiseMatSub.cc)
arcane_accelerator_add_source_files(connectivix/ConnectivityTranspose.cc)
arcane_accelerator_add_source_files(MicroHydroModule.cc)
arcane_accelerator_add_source_files(MicroHydroModule.cc)
arcane_accelerator_add_to_target(MicroHydro)

target_compile_options(MicroHydro PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -G >)
