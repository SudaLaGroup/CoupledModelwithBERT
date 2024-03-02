from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [Extension("sf",["sf.pyx"])]
setup(
    name = "sf pyx",
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
