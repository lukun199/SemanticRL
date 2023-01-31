#!/data/lk/anaconda3/envs/pt16_py37/bin/python

"""
setup.py file for SWIG example
"""
 
from distutils.core import setup, Extension
 
 
example_module = Extension(
    '_CiderDCpp',
    sources=['CiderD_wrap.cxx', 'CiderD.cpp'],
    extra_compile_args=['-std=c++11'])

 
setup(
    name = 'example',
    version = '0.1',
    author      = "lukun199",
    description = "Cpp implementation of Cider",
    ext_modules = [example_module],
    py_modules  = ["CiderDCpp"])