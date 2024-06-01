from enum import IntEnum


class PaperType(IntEnum):
    PURE_MARKDOWN = 0
    GROBID_PAPER = 1
    PMC_PAPER = 2
    PM_INFO = 3
    NSFC = 4
