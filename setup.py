from __future__ import absolute_import, print_function

import multiprocessing
import os
import shutil
import subprocess
import zipfile
import platform
from distutils.sysconfig import get_config_var
from distutils.version import LooseVersion

try:
    # For Python 3
    from urllib.request import urlopen
    import io

    def get_zipfile(url):
        """Returns a ZipFile constructed from the file at the given URL."""
        r = urlopen(url)
        return zipfile.ZipFile(io.BytesIO(r.read()))
except ImportError:
    # Python 2
    from urllib2 import urlopen
    import StringIO

    def get_zipfile(url):
        """Returns a ZipFile constructed from the file at the given URL."""
        r = urlopen(url)
        return zipfile.ZipFile(StringIO.StringIO(r.read()))

from setuptools import setup, Extension, find_packages
from pkg_resources import resource_filename

import sys
from sys import version_info as ver

# Target branch
TILEDB_VERSION = "1.4.2"

# Use `setup.py [] --debug` for a debug build of libtiledb
TILEDB_DEBUG_BUILD = False

# Directory containing this file
CONTAINING_DIR = os.path.abspath(os.path.dirname(__file__))

# Build directory path
BUILD_DIR = os.path.join(CONTAINING_DIR, "build")

# TileDB package source directory
TILEDB_PKG_DIR = os.path.join(CONTAINING_DIR, "tiledb")

def is_windows():
    return os.name == 'nt'

def libtiledb_exists(library_dirs):
    """
    Checks the given list of paths and returns true if any contain the TileDB library.
    :return: The path to the TileDB library, or None.
    """

    print("libtiledb_exists checking 'library_dirs': {}".format(library_dirs))

    if len(library_dirs) > 0:
        names = libtiledb_library_names()
        paths = [os.path.join(d, n) for d in library_dirs for n in names]
        for p in paths:
            if os.path.exists(p):
                return p
        raise RuntimeError("Could not find given --tiledb library path(s):\n{}"
                .format("\n".join(paths)))
    # If no explicit path is given check to see if TileDB is globally installed.
    import ctypes
    if os.name == "posix":
        if sys.platform == "darwin":
            lib_name = "libtiledb.dylib"
        else:
            lib_name = "libtiledb.so"
    elif os.name == "nt":
        lib_name = "tiledb.dll"
    try:
        ctypes.CDLL(lib_name)
        return lib_name
    except:
        pass

    return None


def libtiledb_library_names():
    """
    :return: List of TileDB shared library names.
    """
    if os.name == "posix":
        if sys.platform == "darwin":
            return ["libtiledb.dylib"]
        else:
            return ["libtiledb.so"]
    elif os.name == "nt":
            return ["tiledb.dll"]
    else:
        raise RuntimeError("Unsupported OS name " + os.name)


def download_libtiledb():
    """
    Downloads the native TileDB source.
    :return: Path to extracted source directory.
    """
    dest_name = "TileDB-{}".format(TILEDB_VERSION)
    dest = os.path.join(BUILD_DIR, dest_name)
    if not os.path.exists(dest):
        url = "https://github.com/TileDB-Inc/TileDB/archive/{}.zip".format(TILEDB_VERSION)
        print("Downloading TileDB package from {}...".format(TILEDB_VERSION))
        with get_zipfile(url) as z:
            z.extractall(BUILD_DIR)
    return dest


def build_libtiledb(src_dir):
    """
    Builds and installs the native TileDB library.
    :param src_dir: Path to libtiledb source directory.
    :return: Path to the directory where the library was installed.
    """
    # From https://github.com/pandas-dev/pandas/pull/24274
    # 3-Clause BSD License: https://github.com/pandas-dev/pandas/blob/master/LICENSE
    # For mac, ensure extensions are built for macos 10.9 when compiling on a
    # 10.9 system or above, overriding distutils behaviour which is to target
    # the version that python was built for. This may be overridden by setting
    # MACOSX_DEPLOYMENT_TARGET before calling setup.py
    if sys.platform == 'darwin':
      if 'MACOSX_DEPLOYMENT_TARGET' not in os.environ:
          current_system = LooseVersion(platform.mac_ver()[0])
          python_target = LooseVersion(
              get_config_var('MACOSX_DEPLOYMENT_TARGET'))
          if python_target < '10.9' and current_system >= '10.9':
              os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'

    libtiledb_build_dir = os.path.join(src_dir, "build")
    libtiledb_install_dir = os.path.join(src_dir, "dist")
    if not os.path.exists(libtiledb_build_dir):
        os.makedirs(libtiledb_build_dir)

    print("Building libtiledb in directory {}...".format(libtiledb_build_dir))
    cmake_cmd = ["cmake",
                    "-DCMAKE_INSTALL_PREFIX={}".format(libtiledb_install_dir),
                    "-DTILEDB_TESTS=OFF",
                    "-DTILEDB_S3=ON",
                    "-DTILEDB_HDFS={}".format("ON" if os.name == "posix" else "OFF"),
                    ]

    if TILEDB_DEBUG_BUILD:
        build_type = "Debug"
    else:
        build_type = "Release"

    cmake_cmd.append("-DCMAKE_BUILD_TYPE={}".format(build_type))

    if os.name == 'nt':
        cmake_cmd.extend(['-A', 'x64', "-DMSVC_MP_FLAG=/MP4"])

    # cmake target directory -- important
    cmake_cmd.append(src_dir)

    print("CMake configure command: {}".format(cmake_cmd))

    have_make = True
    try:
        subprocess.check_call(["make", "-v"])
    except:
        have_make = False

    if have_make and not os.name == 'nt':
        njobs = multiprocessing.cpu_count() or 2
        build_cmd = ["make", "-j{:d}".format(njobs)]
        install_cmd = ["make", "install-tiledb"]
    else:
        build_cmd = ["cmake", "--build", ".", "--config", "Release"]
        install_cmd = ["cmake", "--build", ".", "--config", "Release", "--target", "install-tiledb"]

    # Build and install libtiledb
    subprocess.check_call(cmake_cmd, cwd=libtiledb_build_dir)
    subprocess.check_call(build_cmd, cwd=libtiledb_build_dir)
    subprocess.check_call(install_cmd, cwd=libtiledb_build_dir)

    if not 'TILEDB_PATH' in os.environ:
        os.environ['TILEDB_PATH'] = libtiledb_install_dir
    return libtiledb_install_dir


def find_or_install_libtiledb(setuptools_cmd):
    """
    Find the TileDB library required for building the Cython extension. If not found,
    download, build and install TileDB, copying the resulting shared libraries
    into a path where they will be found by package_data.

    :param setuptools_cmd: The setuptools command instance.
    """
    tiledb_ext = None
    for ext in setuptools_cmd.distribution.ext_modules:
        if ext.name == "tiledb.libtiledb":
            tiledb_ext = ext
            break

    # Download, build and locally install TileDB if needed.
    if not libtiledb_exists(tiledb_ext.library_dirs):
        src_dir = download_libtiledb()
        install_dir = build_libtiledb(src_dir)
        lib_subdir = 'bin' if os.name=='nt' else 'lib'
        native_subdir = '' if is_windows() else 'native'
        # Copy libtiledb shared object(s) to the package directory so they can be found
        # with package_data.
        dest_dir = os.path.join(TILEDB_PKG_DIR, native_subdir)
        for libname in libtiledb_library_names():
            src = os.path.join(install_dir, lib_subdir, libname)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            dest = os.path.join(dest_dir, libname)
            print("Copying file {0} to {1}".format(src, dest))
            shutil.copy(src, dest)

        # TODO hack
        # also copy the lib file for dependees
        # this needs to come before
        if is_windows():
            def do_copy(src, dest):
                print("Copying file {0} to {1}".format(src, dest))
                shutil.copy(src, dest)

            # lib files for linking
            src = os.path.join(install_dir, "lib", "tiledb.lib")
            dest = os.path.join(dest_dir, "tiledb.lib")
            do_copy(src, dest)

            # tbb
            src = os.path.join(install_dir, "bin", "tbb.dll")
            dest = os.path.join(dest_dir, "tbb.dll")
            do_copy(src, dest)
            src = os.path.join(install_dir, "lib", "tbb.lib")
            dest = os.path.join(dest_dir, "tbb.lib")
            do_copy(src, dest)

            #
            tiledb_ext.library_dirs += [os.path.join(install_dir, "lib")]

        # Update the TileDB Extension instance with correct paths.
        tiledb_ext.library_dirs += [os.path.join(install_dir, lib_subdir)]
        tiledb_ext.include_dirs += [os.path.join(install_dir, "include")]
        # Update package_data so the shared object gets installed with the Python module.
        libtiledb_objects = [os.path.join(native_subdir, libname) for libname in libtiledb_library_names()]
        if is_windows():
            libtiledb_objects.extend(
                [os.path.join(native_subdir, libname) for libname in
                              ["tiledb.lib", "tbb.dll", "tbb.lib"]])

        setuptools_cmd.distribution.package_data.update({"tiledb": libtiledb_objects})


class LazyCommandClass(dict):
    """
    Lazy command class that defers operations requiring Cython and numpy until
    they've actually been downloaded and installed by setup_requires.
    """

    def __contains__(self, key):
        return (
                key in ['build_ext', 'bdist_wheel', 'bdist_egg']
                or super(LazyCommandClass, self).__contains__(key)
        )

    def __setitem__(self, key, value):
        if key == 'build_ext':
            raise AssertionError("build_ext overridden!")
        super(LazyCommandClass, self).__setitem__(key, value)

    def __getitem__(self, key):
        if key == 'build_ext':
            return self.make_build_ext_cmd()
        elif key == 'bdist_wheel':
            return self.make_bdist_wheel_cmd()
        elif key == 'bdist_egg':
            return self.make_bdist_egg_cmd()
        else:
            return super(LazyCommandClass, self).__getitem__(key)

    def make_build_ext_cmd(self):
        """
        :return: A command class implementing 'build_ext'.
        """
        from Cython.Distutils import build_ext as cython_build_ext

        class build_ext(cython_build_ext):
            """
            Custom build_ext command that lazily adds numpy's include_dir to
            extensions.
            """

            def build_extensions(self):
                """
                Lazily append numpy's include directory to Extension includes.

                This is done here rather than at module scope because setup.py
                may be run before numpy has been installed, in which case
                importing numpy and calling `numpy.get_include()` will fail.
                """
                numpy_incl = resource_filename('numpy', 'core/include')
                for ext in self.extensions:
                    ext.include_dirs.append(numpy_incl)

                find_or_install_libtiledb(self)

                # This explicitly calls the superclass method rather than the
                # usual super() invocation because distutils' build_class, of
                # which Cython's build_ext is a subclass, is an old-style class
                # in Python 2, which doesn't support `super`.
                cython_build_ext.build_extensions(self)

        return build_ext

    def make_bdist_wheel_cmd(self):
        """
        :return: A command class implementing 'bdist_wheel'.
        """
        from wheel.bdist_wheel import bdist_wheel

        class bdist_wheel_cmd(bdist_wheel):
            def run(self):
                # This may modify package_data:
                find_or_install_libtiledb(self)
                bdist_wheel.run(self)

        return bdist_wheel_cmd

    def make_bdist_egg_cmd(self):
        """
        :return: A command class implementing 'bdist_egg'.
        """
        from setuptools.command.bdist_egg import bdist_egg

        class bdist_egg_cmd(bdist_egg):
            def run(self):
                # This may modify package_data:
                find_or_install_libtiledb(self)
                bdist_egg.run(self)

        return bdist_egg_cmd


def cmake_available():
    """
    Checks whether CMake command is available and >= version 3.3.
    :return:
    """
    try:
        output = subprocess.check_output(['cmake', '--version']).split()
        version = output[2].decode('utf-8').split('.')
        return int(version[0]) >= 3 and int(version[1]) >= 3
    except:
        return False


def setup_requires():
    req = ['cython>=0.27',
           'numpy>=1.7',
           'setuptools>=18.0',
           'setuptools_scm>=1.5.4',
           'wheel>=0.30']
    # Add cmake requirement if libtiledb is not found and cmake is not available.
    if not libtiledb_exists(LIB_DIRS) and not cmake_available():
        req.append('cmake>=3.11.0')
    return req


TESTS_REQUIRE = []
if ver < (3,):
    TESTS_REQUIRE.extend(["unittest2", "mock"])

# Globals variables
CXXFLAGS = os.environ.get("CXXFLAGS", "-std=c++11" if not is_windows() else "").split()
LFLAGS = os.environ.get("LFLAGS", "").split()

# Allow setting (lib) TileDB directory if it is installed on the system
TILEDB_PATH = os.environ.get("TILEDB_PATH", "")

# Sources & libraries
INC_DIRS = []
LIB_DIRS = []
LIBS = ["tiledb"]
DEF_MACROS = []
SOURCES = ["tiledb/libtiledb.pyx"]

# Pass command line flags to setup.py script
# handle --tiledb=[PATH] --lflags=[FLAGS] --cxxflags=[FLAGS]
args = sys.argv[:]
for arg in args:
    if arg.find('--tiledb=') == 0:
        TILEDB_PATH = os.path.expanduser(arg.split('=')[1])
        sys.argv.remove(arg)
    if arg.find('--lflags=') == 0:
        LFLAGS = arg.split('=')[1].split()
        sys.argv.remove(arg)
    if arg.find('--cxxflags=') == 0:
        CXXFLAGS = arg.split('=')[1].split()
        sys.argv.remove(arg)
    if arg.find('--debug') == 0:
        TILEDB_DEBUG_BUILD = True
        sys.argv.remove(arg)


if TILEDB_PATH != '':
    LIB_DIRS += [os.path.join(TILEDB_PATH, 'lib')]
    if sys.platform.startswith("linux"):
        LIB_DIRS += [os.path.join(TILEDB_PATH, 'lib64'),
                     os.path.join(TILEDB_PATH, 'lib', 'x86_64-linux-gnu')]
    elif os.name == 'nt':
        LIB_DIRS += [os.path.join(TILEDB_PATH, 'bin')]
    INC_DIRS += [os.path.join(TILEDB_PATH, 'include')]

with open('README.rst') as f:
    README_RST = f.read()

setup(
    name='tiledb',
    description="Pythonic interface to the TileDB array storage manager",
    long_description=README_RST,
    author='TileDB, Inc.',
    author_email='help@tiledb.io',
    maintainer='TileDB, Inc.',
    maintainer_email='help@tiledb.io',
    url='https://github.com/TileDB-Inc/TileDB-Py',
    license='MIT',
    platforms=['any'],
    use_scm_version={
        'version_scheme': 'guess-next-dev',
        'local_scheme': 'dirty-tag',
        'write_to': 'tiledb/version.py'
    },
    ext_modules=[
        Extension(
            "tiledb.libtiledb",
            include_dirs=INC_DIRS,
            define_macros=DEF_MACROS,
            sources=SOURCES,
            library_dirs=LIB_DIRS,
            libraries=LIBS,
            extra_link_args=LFLAGS,
            extra_compile_args=CXXFLAGS,
            language="c++"
        )
    ],
    setup_requires=setup_requires(),
    install_requires=[
        'numpy>=1.7',
        'wheel>=0.30'
    ],
    tests_require=TESTS_REQUIRE,
    packages=find_packages(),
    cmdclass=LazyCommandClass(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Operating System :: Unix',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
