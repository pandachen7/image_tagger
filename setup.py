import os
import shutil
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup

DELETE_BUILD_DIR = True
DESTINATION_FOLDER = "compiled"
BUILD_DIR = "build"


def generate_extensions(source_dir="src"):
    extensions = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".py") and file not in ["__init__.py", "node.py"]:
                file_path = os.path.join(root, file)
                # 將路徑轉換為模組名稱 (例如: src/dir1/module1.py → dir1.module1)
                rel_path = os.path.relpath(file_path, source_dir)
                module_name = os.path.splitext(rel_path)[0].replace(os.path.sep, ".")
                extensions.append(Extension(module_name, [file_path]))
    return extensions


setup(
    ext_modules=cythonize(
        generate_extensions(),
        build_dir=BUILD_DIR,  # 中間文件暫存目錄
        compiler_directives={"language_level": "3"},  # 指定Python 3語法
    ),
    script_args=["build_ext", "--build-lib", DESTINATION_FOLDER],  # 編譯後輸出目錄
)

# rename
# e.g. main.sub.cp311-win_amd64.pyd -> main.sub.pyd
for pyd_file in Path(DESTINATION_FOLDER).glob("**/*.pyd"):
    name_parts = pyd_file.name.split(".")  # 將檔名分割成多個部分
    new_name_parts = name_parts[:-2] + [name_parts[-1]]  # 保留主檔名和最後一個副檔名
    new_name = ".".join(new_name_parts)  # 重新組合檔名
    new_path = pyd_file.with_name(new_name)  # 建立新的路徑
    pyd_file.replace(new_path)  # 執行更名

# Remove the build directory
if DELETE_BUILD_DIR:
    try:
        shutil.rmtree(BUILD_DIR)
    except FileNotFoundError:
        pass
