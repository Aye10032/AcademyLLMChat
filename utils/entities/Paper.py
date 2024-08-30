from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict


class PaperType(IntEnum):
    PURE_MARKDOWN = 0
    GROBID_PAPER = 1
    PMC_PAPER = 2
    PM_INFO = 3
    NSFC = 4


@dataclass
class PaperInfo:
    author: str = ''
    year: int = -1
    type: int = PaperType.PURE_MARKDOWN
    keywords: str = ''
    ref: bool = False
    doi: str = ''

    @classmethod
    def from_yaml(cls, data: Dict[str, any]):
        return cls(**data)


@dataclass
class Section:
    text: str
    level: int


@dataclass
class Reference:
    source_doi: str
    ref_list: list = field(default_factory=list)


@dataclass
class Paper:
    info: PaperInfo
    sections: list[Section] = field(default_factory=list)
    reference: Reference = field(default_factory=Reference)
