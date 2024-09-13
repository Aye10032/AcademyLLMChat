from dataclasses import dataclass, asdict
from enum import IntEnum
from typing import Any


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls(**data)


@dataclass
class Project:
    name: str
    owner: User


def main() -> None:
    user = User.from_dict({
        "name": "aye",
        "password": "114514",
        "user_group": UserGroup.ADMIN
    })


if __name__ == '__main__':
    main()
