# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import glob
import hashlib
import importlib
import os
import pathlib
import re
import shutil
import subprocess
import sys
import uuid

import torch
import torch.utils.cpp_extension
from torch.utils.file_baton import FileBaton

allow_module_rebuild = True
#allow_module_rebuild = False  # useful for profiling

if allow_module_rebuild:
    print('allow_module_rebuild =', allow_module_rebuild, '-> this would significantly slow down or freeze profiling!')
else:
    print('allow_module_rebuild =', allow_module_rebuild, '-> if not profiling, re-enable it!')


try:
    import intel_extension_for_pytorch as ipex
    using_xpu = True
    cpp_extension = torch.xpu.cpp_extension
    torch_device_specific = torch.xpu
    device_str = 'xpu'

    def build_and_load_module(name, build_directory,  *args, **kwargs):
        if allow_module_rebuild:
            cpp_extension.load(name=name, build_directory=build_directory, *args, **kwargs)
        else:
            print('allow_module_rebuild == False: only loading a module that was built previously')
            ipex.xpu.cpp_extension._import_module_from_library(module_name=name, path=build_directory, is_python_module=True)

except:
    print('Warning: intel_extension_for_pytorch not loaded')
    using_xpu = False
    cpp_extension = torch.utils.cpp_extension
    torch_device_specific = torch.cuda
    device_str = 'cuda'
    build_and_load_module = cpp_extension.load

#----------------------------------------------------------------------------
# Global options.

verbosity = 'brief' # Verbosity level: 'none', 'brief', 'full'

#----------------------------------------------------------------------------
# Internal helper funcs.

def _find_compiler_bindir():
    patterns = [
        'C:/Program Files*/Microsoft Visual Studio/*/Professional/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files*/Microsoft Visual Studio/*/BuildTools/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files*/Microsoft Visual Studio/*/Community/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files*/Microsoft Visual Studio */vc/bin',
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if len(matches):
            return matches[-1]
    return None

#----------------------------------------------------------------------------

def _get_mangled_gpu_name():
    name = torch_device_specific.get_device_name(0).lower()
    out = []
    for c in name:
        if re.match('[a-z0-9_-]+', c):
            out.append(c)
        else:
            out.append('-')
    return ''.join(out)

ori_subprocess_run = subprocess.run
def hooked_subprocess_run(command, **kwargs):
    #print('hooked_subprocess_run', command, kwargs)
    if command == ['ninja', '-v']:
        #print('MATCHED!!!!!!!!!!!!!!!!!!!')
        ori_subprocess_run(['bash', '-c', 'ninja -t compdb > compile_commands.json'], **kwargs)
        print('CHECK THE BUILD DIR FOR compile_commands.json:', kwargs['cwd'])
    return ori_subprocess_run(command, **kwargs)
subprocess.run = hooked_subprocess_run

#----------------------------------------------------------------------------
# Main entry point for compiling and loading C++/XPU plugins.

_cached_plugins = dict()

def get_plugin(module_name, sources, headers=None, source_dir=None, **build_kwargs):
    assert verbosity in ['none', 'brief', 'full']
    if headers is None:
        headers = []
    if source_dir is not None:
        sources = [os.path.join(source_dir, fname) for fname in sources]
        headers = [os.path.join(source_dir, fname) for fname in headers]

    # Already cached?
    if module_name in _cached_plugins:
        return _cached_plugins[module_name]

    # Print status.
    if verbosity == 'full':
        print(f'Setting up PyTorch plugin "{module_name}"...')
    elif verbosity == 'brief':
        print(f'Setting up PyTorch plugin "{module_name}"... ', end='', flush=True)
    verbose_build = (verbosity == 'full')

    # Compile and load.
    try: # pylint: disable=too-many-nested-blocks
        # Make sure we can find the necessary compiler binaries.
        if os.name == 'nt' and os.system("where cl.exe >nul 2>nul") != 0:
            compiler_bindir = _find_compiler_bindir()
            if compiler_bindir is None:
                raise RuntimeError(f'Could not find MSVC/GCC/CLANG installation on this computer. Check _find_compiler_bindir() in "{__file__}".')
            os.environ['PATH'] += ';' + compiler_bindir

        # Incremental build md5sum trickery.  Copies all the input source files
        # into a cached build directory under a combined md5 digest of the input
        # source files.  Copying is done only if the combined digest has changed.
        # This keeps input file timestamps and filenames the same as in previous
        # extension builds, allowing for fast incremental rebuilds.
        #
        # This optimization is done only in case all the source files reside in
        # a single directory (just for simplicity) and if the TORCH_EXTENSIONS_DIR
        # environment variable is set (we take this as a signal that the user
        # actually cares about this.)
        #
        # EDIT: We now do it regardless of TORCH_EXTENSIOS_DIR, in order to work
        # around the *.cu dependency bug in ninja config.
        #
        all_source_files = sorted(sources + headers)
        all_source_dirs = set(os.path.dirname(fname) for fname in all_source_files)
        if len(all_source_dirs) == 1: # and ('TORCH_EXTENSIONS_DIR' in os.environ):

            # Compute combined hash digest for all source files.
            hash_md5 = hashlib.md5()
            for src in all_source_files:
                with open(src, 'rb') as f:
                    hash_md5.update(f.read())

            # Select cached build directory name.
            source_digest = hash_md5.hexdigest()
            build_top_dir = cpp_extension._get_build_directory(module_name, verbose=verbose_build) # pylint: disable=protected-access
            cached_build_dir = os.path.join(build_top_dir, f'{source_digest}-{_get_mangled_gpu_name()}')

            if not os.path.isdir(cached_build_dir):
                tmpdir = f'{build_top_dir}/srctmp-{uuid.uuid4().hex}'
                os.makedirs(tmpdir)
                for src in all_source_files:
                    shutil.copyfile(src, os.path.join(tmpdir, os.path.basename(src)))
                try:
                    os.replace(tmpdir, cached_build_dir) # atomic
                except OSError:
                    # source directory already exists, delete tmpdir and its contents.
                    shutil.rmtree(tmpdir)
                    if not os.path.isdir(cached_build_dir): raise

            # Compile.
            cached_sources = [os.path.join(cached_build_dir, os.path.basename(fname)) for fname in sources]

            #subprocess.run(command, stdout=stdout_fileno if verbose else subprocess.PIPE, stderr=subprocess.STDOUT, cwd=build_directory, check=True, env=env)
            build_and_load_module(name=module_name, build_directory=cached_build_dir,
                verbose=verbose_build, sources=cached_sources, **build_kwargs)
        else:
            build_and_load_module(name=module_name, verbose=verbose_build, sources=sources, **build_kwargs)

        # Load.
        module = importlib.import_module(module_name)

    except:
        if verbosity == 'brief':
            print('Failed!')
        raise

    # Print status and add to cache dict.
    if verbosity == 'full':
        print(f'Done setting up PyTorch plugin "{module_name}".')
    elif verbosity == 'brief':
        print('Done.')
    _cached_plugins[module_name] = module
    return module

#----------------------------------------------------------------------------

def get_ops_include_path():
    include_dir_name = 'include_dpct_2024.2'#'include_2023.2'
    return pathlib.Path(__file__).parent.resolve().joinpath('ops').joinpath(include_dir_name)

def get_xpu_cflags():
    return [
        '-ffast-math', '-cl-fast-relaxed-math', '-I'+str(get_ops_include_path()),
        #'-O3',
        '-O2',
        '-g', # enables VTune to show the C++ kernel source and instructions spent on each code line (rather than just assembly instructions)
        #'-gline-tables-only', # seems to only give line info for non-kernel C++ code
        '-fdebug-info-for-profiling', # https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2024-0/gpu-compute-media-hotspots-analysis.html
    ]
