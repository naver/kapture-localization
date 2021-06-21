from enum import auto
import kapture_localization.utils.path_to_kapture  # noqa: F401
from kapture.utils import AutoEnum


class BenchmarkFormatStyle(AutoEnum):
    Default = auto()
    RobotCar_Seasons = auto()
    Gangnam_Station = auto()
    Hyundai_Department_Store = auto()
    RIO10 = auto()

    def __str__(self):
        return self.value
