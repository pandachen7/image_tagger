# python build.py build_ext --inplace cleanall

from setuptools import setup
from Cython.Build import cythonize
import shutil
import os
import sys

# Define the .py files you want to compile
py_files = []
for root, dirs, files in os.walk("src"):
    for file in files:
        if file.endswith(".py"):
            py_files.append(os.path.join(root, file))

# Cythonize the .py files
ext_modules = cythonize(py_files, language_level="3")

setup(
    name='image_tagger',
    ext_modules=ext_modules,
)

# Clean up build files
if "cleanall" in sys.argv:
    print("Deleting Cython files...")
    try:
        shutil.rmtree("build")
    except FileNotFoundError:
        pass
    for py in py_files:
        c_file = py.replace(".py", ".c")
        so_file = py.replace(".py", ".so")  # For Linux
        pyd_file = py.replace(".py", ".pyd")  # For Windows
        try:
            os.remove(c_file)
        except FileNotFoundError:
            pass
        try:
            os.remove(so_file)
        except FileNotFoundError:
            pass
        try:
            os.remove(pyd_file)
        except FileNotFoundError:
            pass
    sys.argv.remove("cleanall")
