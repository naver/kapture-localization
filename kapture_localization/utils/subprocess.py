# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import subprocess
import sys
import os
import os.path as path
from typing import List, Optional
from kapture_localization.utils.logging import getLogger


def find_in_PATH(filename: str):
    """
    Look for file in current directory and all PATH directories

    :param filename: name of the file to look for
    :type filename: str
    :raises FileNotFoundError: Could not find file in any of the paths
    :return: the path for which path.isfile returned true
    :rtype: str
    """
    if path.isfile(filename):
        return path.normpath(filename)

    os_paths = os.environ['PATH'].split(path.pathsep)
    for os_path in os_paths:
        fullpath_file = path.join(os_path, filename)
        if path.isfile(fullpath_file):
            return path.normpath(fullpath_file)
    raise FileNotFoundError(f'could not find {filename}')


def run_python_command(local_path: str, args: List[str], python_binary: Optional[str] = None):
    """
    run a python subprocess

    :param local_path: path where you expect the file to be
    :type local_path: str
    :param args: the arguments of the python process
    :type args: List[str]
    :param python_binary: path to the python binary, optional, when None, the .py file is called directly
    :type python_binary: Optional[str]
    :raises ValueError: subprocess crashed
    """
    if python_binary is None:
        if path.isfile(local_path):
            compute_image_pairs_bin = path.normpath(local_path)
        else:
            # maybe the script was installed through pip
            compute_image_pairs_bin = path.basename(local_path)
        args.insert(0, compute_image_pairs_bin)
    else:
        if path.isfile(local_path):
            compute_image_pairs_bin = path.normpath(local_path)
        else:
            # maybe the script was installed through pip
            # with the direct binary, we need to get the full path
            compute_image_pairs_bin = find_in_PATH(path.basename(local_path))
        args.insert(0, compute_image_pairs_bin)
        args.insert(0, python_binary)

    getLogger().debug(f'run_python_command : {args}')

    use_shell = sys.platform.startswith("win")
    python_process = subprocess.Popen(args, shell=use_shell)
    python_process.wait()
    if python_process.returncode != 0:
        raise ValueError('\nSubprocess Error (Return code:' f' {python_process.returncode} )')
