import os
import re
import sys
import platform
import subprocess
from zipfile import ZipFile

try:
    from urllib.request import urlopen # Python 3.x
    from urllib.error import URLError, HTTPError
except ImportError:
    from urllib2 import urlopen # Python 2
    from urllib2 import URLError, HTTPError

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


# Prior to installation, make sure pybind11 folder exists. If not, download it.

pybind_version = "2.4.3"
pybind_url = "https://github.com/pybind/pybind11/archive/v{0}.zip".format(pybind_version)
if not os.path.isdir("pybind11"):
    print("\n=== Downloading pybind11 ===")
    try:
        pybind_http = urlopen(pybind_url)
    except (URLError, HTTPError):
        raise RuntimeError("Failed to download pybind.")
    with open("pybind.zip", 'wb') as out:
        out.write(pybind_http.read())
    with ZipFile("pybind.zip", "r") as zf:
        zf.extractall()
    os.rename("pybind11-{0}".format(pybind_version), "pybind11")
    os.remove("pybind.zip")
    print("=== pybind11 downloaded sucessfully ===\n")


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup(
    name='pyopenface',
    version='0.0.1',
    author='Austin Hurst',
    author_email='mynameisaustinhurst@gmail.com',
    description='Experimental Python bindings for the OpenFace library.',
    long_description='',
    ext_modules=[CMakeExtension('_openface')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    packages=['pyopenface']
)
