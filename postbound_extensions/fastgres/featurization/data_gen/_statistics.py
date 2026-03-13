
from __future__ import annotations
from pathlib import Path
from abc import ABC, abstractmethod
from .._dbutil import DatabaseConnection


class StatisticsComponent(ABC):

    @abstractmethod
    def to_dict(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def build(self, dbc: DatabaseConnection, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save(self, dir_path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, dir_path: Path) -> StatisticsComponent:
        raise NotImplementedError
