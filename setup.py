"""The C++ backend is optional; installation also supports Python execution."""

import os

from setuptools import Extension, setup


compile_args = ["/std:c++17", "/O2", "/fp:strict"] if os.name == "nt" else [
    "-std=c++17", "-O3", "-ffp-contract=off",
]

setup(ext_modules=[Extension(
    "pedigree_panel_scaling._native",
    sources=["src/pedigree_panel_scaling/_native.cpp"],
    language="c++", optional=True, extra_compile_args=compile_args,
)])
