from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Optional


class UserGroup(IntEnum):
    VISITOR = 0
    WRITER = 1
    FILE_ADMIN = 2
    ADMIN = 3

    @classmethod
    def names(cls):
        return list(cls.__members__.keys())

    @classmethod
    def from_name(cls, name: str):
        return cls[name]


@dataclass
class User:
    name: str
    password: str
    user_group: int
    last_project: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls(**data)


@dataclass
class Project:
    name: str
    owner: str
    last_chat: str
    update_time: float
    create_time: float = 0.
    time_zone: str = 'Asia/Shanghai'

    @classmethod
    def from_list(cls, data: list[Any]):
        return cls(*data[1:])


@dataclass
class ChatHistory:
    session_id: str
    description: str
    owner: str
    project: str
    update_time: float
    create_time: float = 0.

    @classmethod
    def from_list(cls, data: list[Any]):
        return cls(*data[1:])
