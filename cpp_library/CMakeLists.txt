cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 14)

project(sudoku_solver_library)
add_subdirectory(pybind11)
pybind11_add_module(sudoku_solver main.cpp)
