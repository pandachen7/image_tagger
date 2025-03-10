"""
# 使用以下cmd
python setup.py build_ext --inplace
# vscode 不可直接用f5跑
"""

from setuptools import setup
from Cython.Build import cythonize
import shutil
import os
# import sys  # Remove unused import
from pathlib import Path

DELETE_BUILD_DIR = True

# Define the .py files you want to compile
py_files = []
for root, dirs, files in os.walk("src"):
    for file in files:
        if file.endswith(".py") and file not in ["__init__.py", "node.py"]:
            py_files.append(os.path.join(root, file))

# Cythonize the .py files
ext_modules = cythonize(py_files, language_level="3")

setup(
    name='image_tagger',
    ext_modules=ext_modules,
)

#  Clean up build files and move .pyd files
print("Moving and cleaning up build files...")

build_dir = Path("build")
lib_dir = Path("lib")

# Create lib directory if it doesn't exist
lib_dir.mkdir(exist_ok=True)

# Find all .pyd files in the build directory
for pyd_file in build_dir.glob("**/*.pyd"):
    # Construct the new path in the lib directory
    relative_path = pyd_file.relative_to(build_dir)

    # Find "lib.win-amd64-cpython-311" in parents and replace it with "lib"
    new_path_parts = []
    for part in relative_path.parts:
        if "lib.win-amd64-cpython-311" in part:
            # new_path_parts.append("lib")  # Replace with "lib"
            pass
        else:
            new_path_parts.append(part)

    new_path = lib_dir.joinpath(*new_path_parts)
    # new_path = lib_dir / relative_path

    # Remove the platform-specific suffix from the filename
    new_path = new_path.with_name(pyd_file.stem.split('.')[0] + pyd_file.suffix)

    # Create parent directories if they don't exist
    new_path.parent.mkdir(parents=True, exist_ok=True)

    # Move and rename the file
    pyd_file.replace(new_path)

# Remove the build directory
if DELETE_BUILD_DIR:
    try:
        shutil.rmtree(build_dir)
    except FileNotFoundError:
        pass
