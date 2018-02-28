from __future__ import absolute_import, print_function

import os
from setuptools import setup, Extension, find_packages
from pkg_resources import resource_filename

import sys
from sys import version_info as ver

# Check if Python version is supported
if any([ver < (3, 4)]):
    raise Exception("Unsupported Python version %d.%d.  Requires Python >= 3.4")


class LazyCommandClass(dict):
    """
    Lazy command class that defers operations requiring Cython and numpy until
    they've actually been downloaded and installed by setup_requires.
    """
    def __contains__(self, key):
        return (
            key == 'build_ext'
            or super(LazyCommandClass, self).__contains__(key)
        )

    def __setitem__(self, key, value):
        if key == 'build_ext':
            raise AssertionError("build_ext overridden!")
        super(LazyCommandClass, self).__setitem__(key, value)

    def __getitem__(self, key):
        if key != 'build_ext':
            return super(LazyCommandClass, self).__getitem__(key)

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

                # This explicitly calls the superclass method rather than the
                # usual super() invocation because distutils' build_class, of
                # which Cython's build_ext is a subclass, is an old-style class
                # in Python 2, which doesn't support `super`.
                cython_build_ext.build_extensions(self)
        return build_ext


tests_require = []
if ver < (3,):
    tests_require.extend(["unittest2", "mock"])

# Globals variables
CXXFLAGS = os.environ.get("CXXFLAGS", "-std=c++11").split()
LFLAGS = os.environ.get("LFLAGS", "").split()

# Allow setting (lib) TileDB directory if it is installed on the system
TILEDB_DIR = os.environ.get("TILEDB_DIR", "")

# Sources & libraries
inc_dirs = []
lib_dirs = []
libs = ["tiledb"]
def_macros = []
sources = ["tiledb/libtiledb.pyx"]
optional_libs = []

# Pass command line flags to setup.py script
# handle --tiledb=[PATH] --lflags=[FLAGS] --cxxflags=[FLAGS]
args = sys.argv[:]
for arg in args:
    if arg.find('--tiledb=') == 0:
        TILEDB_DIR = os.path.expanduser(arg.split('=')[1])
        sys.argv.remove(arg)
    if arg.find('--lflags=') == 0:
        LFLAGS = arg.split('=')[1].split()
        sys.argv.remove(arg)
    if arg.find('--cxxflags=') == 0:
        CXXFLAGS = arg.split('=')[1].split()
        sys.argv.remove(arg)

if TILEDB_DIR != '':
    lib_dirs += [os.path.join(TILEDB_DIR, 'lib')]
    inc_dirs += [os.path.join(TILEDB_DIR, 'include')]

setup(
    name='tiledb',
    description="Pythonic interface to the TileDB array storage manager",
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
          include_dirs=inc_dirs,
          define_macros=def_macros,
          sources=sources,
          library_dirs=lib_dirs,
          libraries=libs,
          extra_link_args=LFLAGS,
          extra_compile_args=CXXFLAGS,
          language="c++"
        )
    ],
    setup_requires=[
        'cython>=0.22',
        'numpy>=1.7',
        'setuptools>18.0',
        'setuptools-scm>1.5.4'
    ],
    install_requires=[
        'numpy>=1.7',
    ],
    tests_require=tests_require,
    packages=find_packages(),
    cmdclass=LazyCommandClass(),
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

)
