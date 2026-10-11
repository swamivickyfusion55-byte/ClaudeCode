cmake_minimum_required(VERSION 3.22.1)
project("phoenix_engine")

add_library(
        phoenix_engine
        SHARED
        native-lib.cpp
)

find_library(log-lib log)

target_link_libraries(
        phoenix_engine
        ${log-lib}
)
