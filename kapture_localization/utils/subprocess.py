# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import subprocess
import sys
import os.path as path
from typing import List
from kapture_localization.utils.logging import getLogger


def run_python_command(local_path: str, args: List[str]):
    if path.isfile(local_path):
        compute_image_pairs_bin = path.normpath(local_path)
    else:
        # maybe the script was installed through pip
        compute_image_pairs_bin = path.basename(local_path)

    args.insert(0, compute_image_pairs_bin)
    use_shell = sys.platform.startswith("win")
    python_process = subprocess.Popen(args, shell=use_shell)
    python_process.wait()
    if python_process.returncode != 0:
        getLogger().debug(f'{compute_image_pairs_bin} : {args}')
        raise ValueError('\nSubprocess Error (Return code:' f' {python_process.returncode} )')
