from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import StrEnum
from typing import Generic, Optional, TypeVar

T = TypeVar('T')


class VidType(StrEnum):
    STRING255 = 'FIXED_STRING(255)'
    INT64 = 'INT[64]'


class PropType(StrEnum):
    INT64 = 'int64'
    INT32 = 'int32'
    INT16 = 'int16'
    INT8 = 'int8'
    FLOAT = 'float'
    DOUBLE = 'double'
    BOOL = 'bool'
    STRING = 'string'
    DATE = 'date'
    TIME = 'time'
    DATETIME = 'datetime'
    TIMESTAMP = 'timestamp'
    DURATION = 'duration'
    GEO = 'geography'
    POINT = 'geography(point)'
    LINESTRING = 'geography(linestring)'
    POLYGON = 'geography(polygon)'


@dataclass
class Prop(Generic[T]):
    prop_name: str
    data_type: str
    not_null: bool = False
    default: Optional[T] = None
    comment: str = ''

    def to_ngql(self):
        prop_str = f'{self.prop_name} {self.data_type}'
        if self.not_null:
            prop_str += ' NOT NULL'

        if self.default is not None:
            if isinstance(self.default, str):
                prop_str += f' DEFAULT {self.default}' if self.default.endswith("()") else f' DEFAULT "{self.default}"'
            else:
                prop_str += f" DEFAULT {self.default}"

        if self.comment:
            prop_str += f' COMMENT "{self.comment}"'

        return prop_str

    def __eq__(self, other):
        if isinstance(other, str):
            return self.prop_name == other
        return False


@dataclass
class NebulaBase(ABC):
    name: str
    props: list[Prop] = field(default_factory=list)
    ttl_duration: int = None
    ttl_col: str = ''
    comment: str = ''

    @abstractmethod
    def to_dict(self):
        pass


class Tag(NebulaBase):
    def to_dict(self):
        return {
            'tag_name': self.name,
            'props': self.props,
            'ttl_duration': self.ttl_duration,
            'ttl_col': self.ttl_col,
            'comment': self.comment
        }


class Edge(NebulaBase):
    def to_dict(self):
        return {
            'edge_name': self.name,
            'props': self.props,
            'ttl_duration': self.ttl_duration,
            'ttl_col': self.ttl_col,
            'comment': self.comment
        }


class NebularBuilder:
    def __init__(self):
        self.name = None
        self.props = []
        self.ttl_duration = None
        self.ttl_col = ''
        self.comment = ''

    def set_name(self, name: str) -> 'NebularBuilder':
        self.name = name
        return self

    def add_prop(self, prop: Prop) -> 'NebularBuilder':
        self.props.append(prop)
        return self

    def set_ttl_duration(self, ttl_duration: int) -> 'NebularBuilder':
        self.ttl_duration = ttl_duration
        return self

    def set_ttl_col(self, ttl_col: str) -> 'NebularBuilder':
        assert ttl_col in self.props
        self.ttl_col = ttl_col
        return self

    def set_comment(self, comment: str) -> 'NebularBuilder':
        self.comment = comment
        return self

    def build_tag(self) -> Tag:
        assert self.name is not None
        return Tag(self.name, self.props, self.ttl_duration, self.ttl_col, self.comment)

    def build_edge(self) -> Edge:
        assert self.name is not None
        return Edge(self.name, self.props, self.ttl_duration, self.ttl_col, self.comment)


def main() -> None:
    paper: Tag = (
        NebularBuilder()
        .set_name('paper')
        .add_prop(Prop('DOI', PropType.STRING, True))
        .add_prop(Prop('Title', PropType.STRING, True))
        .build_tag()
    )

    print(paper.to_dict())


if __name__ == '__main__':
    main()
