
import os
from glob import glob
from setuptools import find_packages, setup
import pybind11; print(pybind11.__file__)
from pybind11.setup_helpers import Pybind11Extension


__version__ = "0.0.2"

ext_modules = [
    Pybind11Extension(
        "_stabletrees",
        sorted(glob("src/*.cpp")),
        cxx_std=14,
        include_dirs=[
            os.path.join(os.path.dirname(__file__), "../include/stabletrees"),
            os.path.join(os.path.dirname(__file__), "../include/stabletrees/trees"),
            os.path.join(os.path.dirname(__file__), "../include/stabletrees/splitters"),
            os.path.join(os.path.dirname(__file__), "../include/stabletrees/criterions"),
            os.path.join(os.path.dirname(__file__), "../include/stabletrees/optimism"),
            os.path.join(os.path.dirname(__file__), "../include/thirdparty/eigen"),
            "/usr/include/eigen3",
            "/usr/local/include/eigen3",
            "/usr/local/include/thirdparty/eigen",
            "../eigen",
        ],
        define_macros=[("VERSION_INFO", __version__)],
    ),
]
from setuptools.command.build_ext import build_ext
class BuildExt(build_ext):
    def build_extensions(self):
        for ext in self.extensions:
            # add /openmp and /openmp:llvm to extra_compile_args
            ext.extra_compile_args += ['/openmp', '/openmp:llvm']
        build_ext.build_extensions(self)

setup(
    name='stabletrees',
    version='0.1.1',    
    description='Regression tree with stable update',
    author='Morten Blørstad',
    author_email='mblorstad@email.com',
    url="https://github.com/MortenBlorstad/StableTrees",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=ext_modules,
    cmdclass = {'build_ext': BuildExt},
    install_requires=['numpy', "scikit-learn", "matplotlib"],
    python_requires=">=3.6",

    
)