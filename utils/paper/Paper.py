from dataclasses import dataclass, field
from typing import Dict

from utils.paper.PaperEnum import PaperType


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
