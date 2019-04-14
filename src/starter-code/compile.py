from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension("cnn2d",["cnn2d.py"]),
    Extension("data",["data.py"]),
    Extension("segmentation",["segmentation.py"]),
    Extension("grouping",["grouping.py"]),
    Extension("train_2d",["train_2d.py"]),
    Extension("test_2d",["test_2d.py"]),
    Extension("cnn3d",["cnn3d.py"]),
    Extension("client_helper",["client_helper.py"])
]

setup(
    name = 'Multi Label Object Detector',
    cmdclass = {'build_ext':build_ext},
    ext_modules = ext_modules
)