cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(ArcaneBenchs VERSION 1.0 LANGUAGES C CXX)

# Il ne faut pas compiler la ou se trouve les sources
if(CMAKE_BINARY_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message(FATAL_ERROR "You can not do in-source compilation. You have to build in a directory distinct from the source directory")
endif()

enable_testing()

# ----------------------------------------------------------------------------

# Shared libraries are mandatory
set (BUILD_SHARED_LIBS TRUE)

# ----------------------------------------------------------------------------

find_package(Arcane REQUIRED)
message(STATUS "Arcane Version = ${Arcane_VERSION}")

# ----------------------------------------------------------------------------

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# http://www.kitware.com/blog/home/post/63
if(NOT DEFINED PROCESSOR_COUNT)
  # Unknown:
  set(PROCESSOR_COUNT 0)

  # Linux:
  set(cpuinfo_file "/proc/cpuinfo")

  if(EXISTS "${cpuinfo_file}")
    file(STRINGS "${cpuinfo_file}" procs REGEX "^processor.: [0-9]+$")
    list(LENGTH procs PROCESSOR_COUNT)
  endif()

  # Windows:
  if(WIN32)
    set(PROCESSOR_COUNT "$ENV{NUMBER_OF_PROCESSORS}")
  endif()
endif()

message(STATUS "Number of processors available: ${PROCESSOR_COUNT}")


# TODO: vérifier que 'framework' existe



# ----------------------------------------------------------------------------
# Add microhydro and tests for microhydro

set(MICROHYDRO_ROOT microhydro/src)
add_subdirectory(${MICROHYDRO_ROOT} microhydro)
set(MICROHYDRO_EXE "${MicroHydro_BINARY_DIR}/MicroHydro")

set(MICROHYDRO_REF_DIR "${CMAKE_CURRENT_SOURCE_DIR}/microhydro/reference_files")
set(MICROHYDRO_DATADIR "${CMAKE_CURRENT_SOURCE_DIR}/microhydro/data")

# Get test names and number of cores from JSON file
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/microhydro/test_list.json" TESTLIST_JSON)

string(JSON NB_TEST LENGTH "${TESTLIST_JSON}" "test_list")
math(EXPR NB_TEST "${NB_TEST} - 1")
message(STATUS "Number of potential MicroHydro test = ${NB_TEST}")

foreach(X RANGE ${NB_TEST})
  string(JSON TEST_BASE_NAME GET "${TESTLIST_JSON}" "test_list" ${X} "name")
  string(JSON NB_PROC GET "${TESTLIST_JSON}" "test_list" ${X} "nb-proc")
  string(JSON MAX_ITER GET "${TESTLIST_JSON}" "test_list" ${X} "max-iter")
  message(STATUS "Try adding microhydro test nb_proc=${NB_PROC}")
  if(PROCESSOR_COUNT GREATER_EQUAL NB_PROC)
    set(TEST_NAME "microhydro_${TEST_BASE_NAME}_${NB_PROC}proc")
    message(STATUS "MicroHydro: TEST_BASE_NAME=${TEST_BASE_NAME} NB_PROC=${NB_PROC}")

    configure_file(microhydro/launch_test.sh.in ${CMAKE_CURRENT_BINARY_DIR}/test_scripts/microhydro/launch_${TEST_NAME}.sh @ONLY)
    add_test(NAME ${TEST_NAME} COMMAND /bin/sh ${CMAKE_CURRENT_BINARY_DIR}/test_scripts/microhydro/launch_${TEST_NAME}.sh)
  endif()
endforeach()



# # ----------------------------------------------------------------------------
# # Add MaHyCo and tests for MaHyCo

add_subdirectory(mahyco)
set(MAHYCO_EXE "${Mahyco_BINARY_DIR}/Mahyco")
set(MAHYCO_DATADIR "${CMAKE_CURRENT_SOURCE_DIR}/mahyco/data")
set(MAHYCO_CPULIST 1 4 8 16 32 128 256 512 1024 2048 4096 8192)
foreach(X ${MAHYCO_CPULIST})
  # Used by 'mahyco/launch_test'
  set(NB_CPU ${X})
  configure_file(mahyco/launch_test.sh.in ${CMAKE_CURRENT_BINARY_DIR}/test_scripts/mahyco/launch_mahyco_${X}.sh @ONLY)
endforeach()

add_test(NAME mahyco_1proc COMMAND /bin/sh ${CMAKE_CURRENT_BINARY_DIR}/test_scripts/mahyco/launch_mahyco_1.sh)
add_test(NAME mahyco_4proc COMMAND /bin/sh ${CMAKE_CURRENT_BINARY_DIR}/test_scripts/mahyco/launch_mahyco_4.sh)

# ----------------------------------------------------------------------------
# Add QAMA and tests for QAMA

add_subdirectory(qama)
set(QAMA_EXE "${QAMA_BINARY_DIR}/qama")

set(QAMA_REF_DIR "${CMAKE_CURRENT_SOURCE_DIR}/qama/reference_files")
set(QAMA_DATADIR "${CMAKE_CURRENT_SOURCE_DIR}/qama/data")

# Get test names and number of cores from JSON file
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/qama/test_list.json" TESTLIST_JSON)

string(JSON NB_TEST LENGTH "${TESTLIST_JSON}" "test_list")
math(EXPR NB_TEST "${NB_TEST} - 1")

set(MAX_ALLOWED_NB_PE ${PROCESSOR_COUNT})
if (MAX_ALLOWED_NB_PE LESS_EQUAL 8)
  set (MAX_ALLOWED_NB_PE 8)
endif()
message(STATUS "Max allowed nb_pe=${MAX_ALLOWED_NB_PE}")

foreach(X RANGE ${NB_TEST})
  string(JSON TEST_BASE_NAME GET "${TESTLIST_JSON}" "test_list" ${X} "name")
  string(JSON NB_PROC GET "${TESTLIST_JSON}" "test_list" ${X} "nb-proc")
  string(JSON NB_THREAD GET "${TESTLIST_JSON}" "test_list" ${X} "nb-thread")
  string(JSON SUBDIR GET "${TESTLIST_JSON}" "test_list" ${X} "subdir")
  string(JSON MAX_ITER GET "${TESTLIST_JSON}" "test_list" ${X} "max-iter")
  math(EXPR TOTAL_NB_PE "${NB_PROC} * ${NB_THREAD}")
  message(STATUS "TOTAL_NB_PE=${TOTAL_NB_PE} MAX_ALLOWED=${MAX_ALLOWED_NB_PE}")

  if (MAX_ALLOWED_NB_PE GREATER_EQUAL TOTAL_NB_PE)
    set(TEST_NAME "qama_${TEST_BASE_NAME}_${NB_PROC}proc_${NB_THREAD}thread")

    message(STATUS "QAMA: TEST_BASE_NAME=${TEST_BASE_NAME} NB_PROC=${NB_PROC} NB_THREAD=${NB_THREAD}")

    configure_file(qama/launch_test.sh.in ${CMAKE_CURRENT_BINARY_DIR}/test_scripts/qama/launch_${TEST_NAME}.sh @ONLY)
    add_test(NAME ${TEST_NAME} COMMAND /bin/sh ${CMAKE_CURRENT_BINARY_DIR}/test_scripts/qama/launch_${TEST_NAME}.sh)
  endif()

endforeach()

# ----------------------------------------------------------------------------
# Add Synchronize
add_subdirectory(synchronize/src)
