import os
import shutil

# import sys  # Remove unused import
from pathlib import Path

from Cython.Build import cythonize
from setuptools import setup

DELETE_BUILD_DIR = True

# Define the .py files you want to compile
py_files = []
for root, dirs, files in os.walk("src"):
    for file in files:
        if file.endswith(".py") and file not in ["__init__.py", "node.py"]:
            py_files.append(os.path.join(root, file))

# Cythonize the .py files
ext_modules = cythonize(py_files, language_level="3", build_dir="build")

setup(
    name="image_tagger",
    ext_modules=ext_modules,
    script_args=['build_ext'],
    options={"build_ext": {"inplace": True}},
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
    # relative_path = pyd_file.relative_to(build_dir) # 錯誤的作法

    # 從 py_files 中找到對應的 .py 檔案路徑
    py_file_path = None
    for py_file in py_files:
        if pyd_file.stem.startswith(Path(py_file).stem):
            py_file_path = py_file
            break

    if py_file_path:
        # 將 .py 檔案路徑的 src 部分替換成 lib
        new_path_str = py_file_path.replace("src", "lib", 1)  # 只替換第一個 "src"
        new_path = Path(new_path_str).with_suffix(".pyd")

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
