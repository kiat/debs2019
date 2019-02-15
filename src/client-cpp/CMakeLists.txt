cmake_minimum_required(VERSION 3.0)
project (client)

# set the standard
set(CMAKE_CXX_STANDARD 14)

# set clang as the default compiler
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

add_executable(client client_app/main.cc client_app/Client.cc client_app/Client.h client_app/parse-float.h)