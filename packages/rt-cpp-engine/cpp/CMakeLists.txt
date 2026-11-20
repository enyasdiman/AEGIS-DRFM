cmake_minimum_required(VERSION 3.18)
project(aegis_rt_engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(AEGIS_ENABLE_SOAPYSDR "Enable SoapySDR hardware transport (PlutoSDR/USRP/BladeRF)" OFF)

add_library(aegis_rt INTERFACE)
target_include_directories(aegis_rt INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(MSVC)
    target_compile_options(aegis_rt INTERFACE /O2 /arch:AVX2)
else()
    target_compile_options(aegis_rt INTERFACE -O3 -march=native -Wall -Wextra)
endif()

add_executable(drfm_rt_benchmark src/drfm_benchmark.cpp)
target_link_libraries(drfm_rt_benchmark PRIVATE aegis_rt)

add_library(aegis_rt_capi SHARED src/drfm_c_api.cpp)
target_link_libraries(aegis_rt_capi PRIVATE aegis_rt)
set_target_properties(aegis_rt_capi PROPERTIES PREFIX "")

if(MINGW)
    target_link_options(drfm_rt_benchmark PRIVATE -static -static-libgcc -static-libstdc++)
    target_link_options(aegis_rt_capi PRIVATE -static -static-libgcc -static-libstdc++)
endif()

if(AEGIS_ENABLE_SOAPYSDR)
    find_package(SoapySDR REQUIRED)
    target_compile_definitions(drfm_rt_benchmark PRIVATE AEGIS_HAS_SOAPYSDR=1)
    target_link_libraries(drfm_rt_benchmark PRIVATE SoapySDR)
endif()
