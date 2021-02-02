from __future__ import absolute_import, print_function

import ctypes
import multiprocessing
import os
import sys
import urllib
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

print("setup.py sys.argv is: ", sys.argv)

# Target branch
TILEDB_VERSION = "2.2.3"
# allow overriding w/ environment variable
TILEDB_VERSION = os.environ.get("TILEDB_VERSION") or TILEDB_VERSION

# Use `setup.py [] --debug` for a debug build of libtiledb
TILEDB_DEBUG_BUILD = False
# Use `setup.py [] --release-symbols` for a release build with symbols libtiledb
TILEDB_SYMBOLS_BUILD = False

# Use `setup.py [] --modular` for a modular build of libtiledb_py
#   Each .pyx file will be built as a separate shared library for faster
#   compilation. This is disabled by default to avoid distributing multiple
#   shared libraries.
TILEDBPY_MODULAR = False

# Allow to override TILEDB_FORCE_ALL_DEPS with environment variable
TILEDB_FORCE_ALL_DEPS = "TILEDB_FORCE_ALL_DEPS" in os.environ
TILEDB_SERIALIZATION = "TILEDB_SERIALIZATION" in os.environ
CMAKE_GENERATOR = os.environ.get("CMAKE_GENERATOR", None)

# Directory containing this file
CONTAINING_DIR = os.path.abspath(os.path.dirname(__file__))

# Build directory path
BUILD_DIR = os.path.join(CONTAINING_DIR, "build")

# TileDB package source directory
TILEDB_PKG_DIR = os.path.join(CONTAINING_DIR, "tiledb")

# Set deployment target for mac
#
# Need to ensure thatextensions are built for macos 10.9 when compiling on a
# 10.9 system or above, overriding distutils behaviour which is to target
# the version used to build the current python binary.
#
# TO OVERRIDE:
#   set MACOSX_DEPLOYMENT_TARGET before calling setup.py
#
# From https://github.com/pandas-dev/pandas/pull/24274
# 3-Clause BSD License: https://github.com/pandas-dev/pandas/blob/master/LICENSE
if sys.platform == 'darwin':
  if 'MACOSX_DEPLOYMENT_TARGET' not in os.environ:
      current_system = LooseVersion(platform.mac_ver()[0])
      python_target = LooseVersion(
          get_config_var('MACOSX_DEPLOYMENT_TARGET'))
      if python_target < '10.9' and current_system >= '10.9':
          os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'

def is_windows():
    return os.name == 'nt'

def _libtiledb_exists(library_dirs):
    """
    Checks the given list of paths and returns true if any contain the TileDB library.
    :return: The path to the TileDB library, or None.
    """

    print("libtiledb_exists checking 'library_dirs': {}".format(library_dirs))

    names = libtiledb_library_names()
    if len(library_dirs) > 0:
        paths = [os.path.join(d, n) for d in library_dirs for n in names]
        for p in paths:
            if os.path.exists(p):
                return p
        raise RuntimeError("Could not find given --tiledb library path(s):\n{}"
                .format("\n".join(paths)))
    # If no explicit path is given check to see if TileDB is globally installed.
    import ctypes
    lib_name = names[0]

    try:
        # note: this is a relative path on linux
        #       https://bugs.python.org/issue21042
        ctypes.CDLL(lib_name)
        return lib_name
    except:
        pass

    return None

def libtiledb_exists(library_dirs):
    lib = _libtiledb_exists(library_dirs)
    print("libtiledb_exists found: '{}'".format(lib))
    return lib


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
        try:
            with get_zipfile(url) as z:
                z.extractall(BUILD_DIR)
        except urllib.error.URLError:
            # try falling back to wget, maybe SSL is broken
            subprocess.check_call(['wget', url], shell=True)
            with zipfile.ZipFile("{}.zip".format(TILEDB_VERSION)) as z:
                z.extractall(BUILD_DIR)
    return dest


def build_libtiledb(src_dir):
    """
    Builds and installs the native TileDB library.
    :param src_dir: Path to libtiledb source directory.
    :return: Path to the directory where the library was installed.
    """
    libtiledb_build_dir = os.path.join(src_dir, "build")
    libtiledb_install_dir = os.path.join(src_dir, "dist")
    if not os.path.exists(libtiledb_build_dir):
        os.makedirs(libtiledb_build_dir)

    print("Building libtiledb in directory {}...".format(libtiledb_build_dir))
    cmake = os.environ.get("CMAKE", "cmake")
    cmake_cmd = [cmake,
                    "-DCMAKE_INSTALL_PREFIX={}".format(libtiledb_install_dir),
                    "-DTILEDB_TESTS=OFF",
                    "-DTILEDB_S3=ON",
                    "-DTILEDB_HDFS={}".format("ON" if os.name == "posix" else "OFF"),
                    "-DTILEDB_INSTALL_LIBDIR=lib",
                    "-DTILEDB_CPP_API=ON",
                    "-DTILEDB_LOG_OUTPUT_ON_FAILURE=ON",
                    "-DTILEDB_FORCE_ALL_DEPS:BOOL={}".format("ON" if TILEDB_FORCE_ALL_DEPS else "OFF"),
                    "-DTILEDB_SERIALIZATION:BOOL={}".format("ON" if TILEDB_SERIALIZATION else "OFF")
                    ]

    extra_cmake_args = os.environ.get("CMAKE_ARGS", [])
    if extra_cmake_args:
        cmake_cmd.extend(extra_cmake_args.split())

    if TILEDB_DEBUG_BUILD:
        build_type = "Debug"
    elif TILEDB_SYMBOLS_BUILD:
        build_type = "RelWithDebInfo"
    else:
        build_type = "Release"

    cmake_cmd.append("-DCMAKE_BUILD_TYPE={}".format(build_type))

    if os.name == 'nt':
        cmake_cmd.extend(['-A', 'x64', "-DMSVC_MP_FLAG=/MP4"])

    if CMAKE_GENERATOR:
        cmake_cmd.extend(['-G', CMAKE_GENERATOR])

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
        build_cmd = ["cmake", "--build", ".", "--config", build_type]
        install_cmd = ["cmake", "--build", ".", "--config", build_type, "--target", "install-tiledb"]

    # Build and install libtiledb
    # - run cmake
    # - run build via 'cmake --build'
    # - run install-tiledb
    subprocess.check_call(cmake_cmd, cwd=libtiledb_build_dir)
    subprocess.check_call(build_cmd, cwd=libtiledb_build_dir)
    subprocess.check_call(install_cmd, cwd=libtiledb_build_dir)

    return libtiledb_install_dir


def find_or_install_libtiledb(setuptools_cmd):
    """
    Find the TileDB library required for building the Cython extension. If not found,
    download, build and install TileDB, copying the resulting shared libraries
    into a path where they will be found by package_data or the build process.

    :param setuptools_cmd: The setuptools command instance.
    """
    tiledb_ext = None
    core_ext = None
    print("ext_modules: ", setuptools_cmd.distribution.ext_modules)
    for ext in setuptools_cmd.distribution.ext_modules:
        if ext.name == "tiledb.libtiledb":
            tiledb_ext = ext
        elif ext.name == "tiledb.core":
            core_ext = ext
        elif ext.name == "tiledb.fragment":
            fragment_ext = ext

    print("tiledb_ext: ", tiledb_ext)
    print("core_ext: ", core_ext)
    print("fragment_ext: ", fragment_ext)
    print("tiledb_ext.library_dirs: ", tiledb_ext.library_dirs)
    wheel_build = getattr(tiledb_ext, 'tiledb_wheel_build', False)
    from_source = getattr(tiledb_ext, 'tiledb_from_source', False)
    lib_exists = libtiledb_exists(tiledb_ext.library_dirs)
    do_install = False

    # Download, build and locally install TileDB if needed.
    if from_source or not lib_exists:
        src_dir = download_libtiledb()
        prefix_dir = build_libtiledb(src_dir)
        do_install = True
    elif lib_exists:
        prefix_dir = os.path.abspath(os.path.join(os.path.split(lib_exists)[0], ".."))
    elif hasattr(tiledb_ext, 'tiledb_path'):
        prefix_dir = getattr(tiledb_ext, 'tiledb_path')


    if wheel_build and is_windows() and lib_exists:
        do_install = True

    print("prefix_dir: ", core_ext)
    print("do_install: ", do_install)

    if do_install:
        lib_subdir = 'bin' if os.name=='nt' else 'lib'
        native_subdir = '' if is_windows() else 'native'
        # Copy libtiledb shared object(s) to the package directory so they can be found
        # with package_data.
        dest_dir = os.path.join(TILEDB_PKG_DIR, native_subdir)
        for libname in libtiledb_library_names():
            src = os.path.join(prefix_dir, lib_subdir, libname)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            dest = os.path.join(dest_dir, libname)
            print("Copying file {0} to {1}".format(src, dest))
            shutil.copy(src, dest)

        # Copy dependencies
        if is_windows():
            def do_copy(src, dest):
                print("Copying file {0} to {1}".format(src, dest))
                shutil.copy(src, dest)

            # lib files for linking
            src = os.path.join(prefix_dir, "lib", "tiledb.lib")
            dest = os.path.join(dest_dir, "tiledb.lib")
            do_copy(src, dest)

            # tbb
            has_tbb = False
            if os.path.isdir(os.path.join(prefix_dir, "bin", "tbb.dll")):
                has_tbb = True
                src = os.path.join(prefix_dir, "bin", "tbb.dll")
                dest = os.path.join(dest_dir, "tbb.dll")
                do_copy(src, dest)
                src = os.path.join(prefix_dir, "lib", "tbb.lib")
                dest = os.path.join(dest_dir, "tbb.lib")
                do_copy(src, dest)

            #
            tiledb_ext.library_dirs += [os.path.join(prefix_dir, "lib")]
            core_ext.library_dirs += [os.path.join(prefix_dir, "lib")]
            fragment_ext.library_dirs += [os.path.join(prefix_dir, "lib")]

        # Update the TileDB Extension instance with correct build-time paths.
        tiledb_ext.library_dirs += [os.path.join(prefix_dir, lib_subdir)]
        tiledb_ext.include_dirs += [os.path.join(prefix_dir, "include")]
        core_ext.library_dirs += [os.path.join(prefix_dir, lib_subdir)]
        core_ext.include_dirs += [os.path.join(prefix_dir, "include")]
        fragment_ext.library_dirs += [os.path.join(prefix_dir, lib_subdir)]
        fragment_ext.include_dirs += [os.path.join(prefix_dir, "include")]

        # Update package_data so the shared object gets installed with the Python module.
        libtiledb_objects = [os.path.join(native_subdir, libname)
                             for libname in libtiledb_library_names()]

        # Make sure the built library is usable
        for shared_obj in libtiledb_objects:
            if is_windows(): continue
            test_path = os.path.join(TILEDB_PKG_DIR, shared_obj)
            # should only ever be 1, not sure why libtiledb_library_names -> List
            try:
                ctypes.CDLL(test_path)
            except:
                print("\n-------------------")
                print("Failed to load shared library: {}".format(test_path))
                print("-------------------\n")
                raise

        # This needs to be a glob in order to pick up the versioned SO
        libtiledb_objects = [so + '*' for so in libtiledb_objects]

        if is_windows():
            libtiledb_objects.extend(
                [os.path.join(native_subdir, libname) for libname in
                              ["tiledb.lib"]])
            if has_tbb:
                libtiledb_objects.extend(
                  [os.path.join(native_subdir, libname) for libname in
                                 ["tbb.lib", "tbb.dll"]])

        print("\n-------------------")
        print("libtiledb_objects: ", libtiledb_objects)
        print("-------------------\n")
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

class get_pybind_include(object):
  """Helper class to determine the pybind11 include path
  The purpose of this class is to postpone importing pybind11
  until it is actually installed, so that the ``get_include()``
  method can be invoked. """

  def __init__(self, user=False):
    self.user = user

  def __str__(self):
    import pybind11
    return pybind11.get_include(self.user)

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

numpy_required_version = 'numpy>=1.16'
def setup_requires():
    req = ['cython>=0.27',
           numpy_required_version,
           'setuptools>=18.0',
           'setuptools_scm>=1.5.4',
           'wheel>=0.30',
           'pybind11']
    # Add cmake requirement if libtiledb is not found and cmake is not available.
    if not libtiledb_exists(LIB_DIRS) and not cmake_available():
        req.append('cmake>=3.11.0')
    return req


TESTS_REQUIRE = []
if ver < (3,):
    TESTS_REQUIRE.extend(["unittest2", "mock"])

# Allow setting (lib) TileDB directory if it is installed on the system
TILEDB_PATH = os.environ.get("TILEDB_PATH", "")
print("TILEDB_PATH from env: '{}'".format(TILEDB_PATH))
# Sources & libraries
INC_DIRS = []
LIB_DIRS = []
LIBS = ["tiledb"]
DEF_MACROS = []

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
    if arg.find('--release-symbols') == 0:
        TILEDB_SYMBOLS_BUILD = True
        sys.argv.remove(arg)
    if arg.find('--modular') == 0:
        TILEDBPY_MODULAR = True
        sys.argv.remove(arg)

# Global variables
CXXFLAGS = os.environ.get("CXXFLAGS", "").split()
if not is_windows():
  CXXFLAGS.append("-std=c++11")
  if not TILEDB_DEBUG_BUILD:
    CXXFLAGS.append("-Wno-deprecated-declarations")
  elif TILEDB_DEBUG_BUILD:
    CXXFLAGS.append("-g")
    CXXFLAGS.append("-O0")
    CXXFLAGS.append("-UNDEBUG") # defined by distutils
  if TILEDB_SYMBOLS_BUILD:
    CXXFLAGS.append("-g")
LFLAGS = os.environ.get("LFLAGS", "").split()

if TILEDB_PATH != '' and TILEDB_PATH != 'source':
    print("TILEDB_PATH in block before: '{}'".format(TILEDB_PATH))
    TILEDB_PATH=os.path.normpath(TILEDB_PATH)
    print("TILEDB_PATH in block after: '{}'".format(TILEDB_PATH))
    LIB_DIRS += [os.path.join(os.path.normpath(TILEDB_PATH), 'lib')]
    if sys.platform.startswith("linux"):
        LIB_DIRS += [os.path.join(TILEDB_PATH, 'lib64'),
                     os.path.join(TILEDB_PATH, 'lib', 'x86_64-linux-gnu')]
    elif os.name == 'nt':
        LIB_DIRS += [os.path.join(TILEDB_PATH, 'bin')]
    INC_DIRS += [os.path.join(TILEDB_PATH, 'include')]
    if sys.platform == 'darwin':
        LFLAGS += ['-Wl,-rpath,{}'.format(p) for p in LIB_DIRS]

with open('README.md') as f:
    README_MD = f.read()

# Source files for build
MODULAR_SOURCES = [
  'tiledb/np2buf.pyx',
  'tiledb/indexing.pyx',
  'tiledb/libmetadata.pyx',
  ]
MODULAR_HEADERS = [
  'tiledb/libtiledb.pxd',
  'tiledb/indexing.pxd'
  ]

__extensions = [
  Extension(
    "tiledb.libtiledb",
    include_dirs=INC_DIRS,
    define_macros=DEF_MACROS,
    sources=["tiledb/libtiledb.pyx"],
    depends=MODULAR_HEADERS,
    library_dirs=LIB_DIRS,
    libraries=LIBS,
    extra_link_args=LFLAGS,
    extra_compile_args=CXXFLAGS,
    language="c++"
    ),
  Extension(
    "tiledb.core",
    ["tiledb/core.cc", "tiledb/npbuffer.cc"],
    include_dirs = INC_DIRS + [
        get_pybind_include(),
        get_pybind_include(user=True)
    ],
    language="c++",
    library_dirs=LIB_DIRS,
    libraries=LIBS,
    extra_link_args=LFLAGS,
    extra_compile_args=CXXFLAGS,
    ),
  Extension(
    "tiledb.fragment",
    ["tiledb/fragment.cc"],
    include_dirs = INC_DIRS + [
        get_pybind_include(),
        get_pybind_include(user=True)
    ],
    language="c++",
    library_dirs=LIB_DIRS,
    libraries=LIBS,
    extra_link_args=LFLAGS,
    extra_compile_args=CXXFLAGS,
    ),
]

if TILEDBPY_MODULAR:
  for source in MODULAR_SOURCES:
    module_name = os.path.splitext(os.path.split(source)[-1])[0]
    if module_name+'.pxd' in MODULAR_HEADERS:
      deps = module_name+'.pxd'
    else:
      deps = None
    ext = Extension(
        "tiledb.{}".format(module_name),
        include_dirs=INC_DIRS,
        define_macros=DEF_MACROS,
        sources=[source],
        depends=[deps] if deps else [],
        library_dirs=LIB_DIRS,
        libraries=LIBS,
        extra_link_args=LFLAGS,
        extra_compile_args=CXXFLAGS,
        language="c++"
        )
    __extensions.append(ext)
else:
  __extensions[0].depends += MODULAR_SOURCES

# Helper to set Extension attributes correctly based on python version
def ext_attr_update(attr, value):
  for x in __extensions:
    if sys.version_info < (3,0):
        x.__dict__[attr] = value
    else:
        x.__setattr__(attr, value)

# Monkey patches to be forwarded to cythonize
# some of these will error out if passed directly
# to Extension(..) above

if (('bdist_wheel' in sys.argv) or ('TILEDB_WHEEL_BUILD' in os.environ)):
    ext_attr_update('tiledb_wheel_build', True)


# - build with `#line` directive annotations
# (equivalent to `emit_linenums` command line directive)
ext_attr_update('cython_line_directives', 1)

# - generate XML debug mapping file (`cython_debug`)
if TILEDB_DEBUG_BUILD:
  ext_attr_update('cython_gdb', True)
  __extensions[1].depends += "tiledb/debug.cc"

# - set rt lib dirs to get correct RPATH on unixy platforms
#   note that we set rpath for darwin separately above.
if not is_windows():
  ext_attr_update('runtime_library_dirs', LIB_DIRS)

if TILEDB_PATH == 'source':
  ext_attr_update('tiledb_from_source', True)
elif TILEDB_PATH != '':
  ext_attr_update('tiledb_path', TILEDB_PATH)

# This must always be set so the compile-time conditional has a value
ext_attr_update('cython_compile_time_env', {'TILEDBPY_MODULAR': TILEDBPY_MODULAR})

setup(
    name='tiledb',
    description="Pythonic interface to the TileDB array storage manager",
    long_description=README_MD,
    long_description_content_type='text/markdown',
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
    ext_modules=__extensions,
    setup_requires=setup_requires(),
    install_requires=[
        numpy_required_version,
        'wheel>=0.30'
    ],
    tests_require=TESTS_REQUIRE,
    packages=find_packages(),
    cmdclass=LazyCommandClass(),
    zip_safe=False,
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
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
