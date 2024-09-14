from dataclasses import dataclass, asdict
from datetime import datetime
from enum import IntEnum
from typing import Any
from zoneinfo import ZoneInfo


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
    last_project: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls(**data)


@dataclass
class Project:
    name: str
    owner: str
    create_time: float
    update_time: float
    archived: bool

    @classmethod
    def from_list(cls, data: list[Any]):
        return cls(*data)


def main() -> None:
    user = User.from_dict({
        "name": "aye",
        "password": "114514",
        "user_group": UserGroup.ADMIN
    })

    now_time = datetime.now().timestamp()
    print(now_time)
    print(type(now_time))

    tz = ZoneInfo('Asia/Shanghai')
    print(datetime.fromtimestamp(now_time, tz).strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == '__main__':
    main()
