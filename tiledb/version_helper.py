import dataclasses
from typing import Tuple, Union

from . import _generated_version


@dataclasses.dataclass(frozen=True)
class VersionHelper:
    version: str
    version_tuple: Tuple[Union[str, int], ...]

    def __call__(self) -> Tuple[Union[str, int], ...]:
        return self.version_tuple


version = VersionHelper(_generated_version.version, _generated_version.version_tuple)
