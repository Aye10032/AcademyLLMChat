import re
import string
from dataclasses import dataclass, field
from enum import IntEnum

from langchain_core.documents import Document


class PaperType(IntEnum):
    PURE_MARKDOWN = 0
    GROBID_PAPER = 1
    PMC_PAPER = 2
    NSFC = 3


@dataclass
class Section:
    text: str
    level: int


@dataclass
class PaperInfo:
    author: str
    year: int
    type: int = PaperType.PURE_MARKDOWN
    keywords: str = ''
    ref: bool = False
    doi: str = ''

    def get_section(self) -> Section:
        block_str = (
            f'---\t\n'
            f'author: {self.author}\t\n'
            f'year: {self.year}\t\n'
            f'type: {self.type}\t\n'
            f'keywords: {self.keywords}\t\n'
            f'ref: {self.ref}\t\n'
            f'doi: {self.doi}\t\n'
            f'---\t\n'
        )
        return Section(block_str, 0)


def format_filename(filename):
    """
    格式化文件名，移除无效字符
    :param filename: 需要格式化的文件名
    :return: 格式化后的文件名
    """
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c for c in filename if c in valid_chars)

    filename = filename.replace('\\', ' or ').replace('|', '')

    filename = re.sub(r'_+', '_', filename)

    filename = filename.strip('._')

    return filename


def split_words(text):
    pattern = re.compile(r'\([^()]*\)|\S+')
    words = pattern.findall(text)
    clean_words = [word.replace('(', '').replace(')', '') for word in words]

    return clean_words


def replace_multiple_spaces(text):
    pattern = re.compile(r'\s+')
    clean_text = pattern.sub(' ', text)

    return clean_text


def is_en(text: str):
    # 使用正则判断输入语句是否只含有英文大小写和数字
    pattern = r'[a-zA-Z0-9]+'
    if re.match(pattern, text):
        return True
    else:
        return False


def save_to_md(sections: list[Section], output_path):
    """
    将章节列表保存为Markdown格式的文件。

    :param sections: 包含章节内容的列表，每个章节都由Section类型表示。
    :param output_path: 输出Markdown文件的路径。
    :return: 无返回值。
    """

    with open(output_path, 'w', encoding='utf-8') as f:
        for sec in sections:
            text = sec.text
            level = sec.level
            if level == 0:
                f.write(f'{text}\t\n')
            elif level == 1:
                f.write(f'# {text}\t\n')
            elif level == 2:
                f.write(f'## {text}\t\n')
            elif level == 3:
                f.write(f'### {text}\t\n')
            else:
                f.write(f'#### {text}\t\n')


def section_to_documents(sections: list[Section], author: str, year: int, doi: str) -> list[Document]:
    """
    将章节列表转换为文档列表。

    :param sections: 一个Section类型的列表，表示文档的各个章节。
    :param author: 文档的作者。
    :param year: 文档发表的年份。
    :param doi: 文档的数字对象标识符。

    :return: 一个Document类型的列表，每个元素代表一个章节的内容及其元数据。
    """
    __Title = ''
    __Section = ''
    docs: list[Document] = []

    for section in sections:
        match section.level:
            case 1:
                __Title = section.text
            case 2:
                __Section = section.text
            case 0:
                if __Section == 'References':
                    continue

                docs.append(Document(
                    page_content=section.text,
                    metadata={
                        'title': __Title,
                        'section': __Section,
                        'author': author,
                        'year': int(year),
                        'doi': doi,
                        'ref': section.ref})
                )
            case _:
                pass

    return docs
