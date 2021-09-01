# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

from enum import auto
import logging
import kapture_localization.utils.path_to_kapture  # noqa: F401
from kapture.utils import AutoEnum


class BenchmarkFormatStyle(AutoEnum):
    Default = auto()
    RobotCar_Seasons = auto()
    Gangnam_Station = auto()
    Hyundai_Department_Store = auto()
    RIO10 = auto()
    ETH_Microsoft = auto()

    def __str__(self):
        return self.value


def get_benchmark_format_command(benchmark_format_style: BenchmarkFormatStyle,
                                 input_path: str,
                                 output_path: str,
                                 force_overwrite_existing: bool,
                                 logger: logging.Logger):
    """
    get script_name and arguments for the export to benchmark format command
    """
    export_LTVL_args = ['-v', str(logger.level),
                        '-i', input_path,
                        '-o', output_path]
    if benchmark_format_style == BenchmarkFormatStyle.ETH_Microsoft:
        script_name = 'kapture_export_ETH_MS_LTVL.py'
    else:
        script_name = 'kapture_export_LTVL2020.py'
        if benchmark_format_style == BenchmarkFormatStyle.RobotCar_Seasons:
            export_LTVL_args.append('-p')
        elif benchmark_format_style == BenchmarkFormatStyle.Gangnam_Station \
                or benchmark_format_style == BenchmarkFormatStyle.Hyundai_Department_Store:
            export_LTVL_args.append('--full_file_name')
        elif benchmark_format_style == BenchmarkFormatStyle.RIO10:
            export_LTVL_args.append('--full_file_name')
            export_LTVL_args.append('--truncate-extensions')
            export_LTVL_args.append('--inverse-pose')
    if force_overwrite_existing:
        export_LTVL_args.append('-f')
    return script_name, export_LTVL_args
