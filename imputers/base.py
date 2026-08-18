from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
import os
from subprocess import list2cmdline

import pandas as pd


class ImputerExecutionError(RuntimeError):
    """Error controlado al ejecutar un imputador externo."""


class BaseImputer(ABC):
    @abstractmethod
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...


def format_command(command: Sequence[str]) -> str:
    if os.name == "nt":
        return list2cmdline(list(command))
    return " ".join(command)
