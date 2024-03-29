import re
import string
from dataclasses import dataclass, field

from langchain_core.documents import Document


@dataclass
class Section:
    text: str
    level: int
    ref: str = ''


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


def save_to_md(sections: list[Section], output_path, append: bool = False, **kwargs):
    """
    将章节列表保存为Markdown格式的文件。

    :param sections: 包含章节内容的列表，每个章节都由Section类型表示。
    :param output_path: 输出Markdown文件的路径。
    :param append: 是否追加模式，默认为False，即覆盖原有文件；如果为True，则追加到文件末尾。
    :param kwargs: 可选参数，用于在文件开头写入额外的信息，如引用(ref)、作者(author)、年份(year)和DOI(doi)。
    :return: 无返回值。
    """
    # 根据append参数决定文件打开模式
    if append:
        open_type = 'a'
    else:
        open_type = 'w'

    with open(output_path, open_type, encoding='utf-8') as f:
        # 如果非追加模式，写入额外信息（如引用、作者、年份和DOI）
        if not append:
            ref: bool = kwargs.get('ref')
            year: str = kwargs.get('year')
            author: str = kwargs.get('author')
            doi: str = kwargs.get('doi')
            f.write(f'---\nref: {ref}\t\nauthor: {author}\t\nyear: {year}\t\ndoi: {doi}\t\n---\n\n')
        # 遍历章节列表，根据章节级别写入相应格式的文本
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
