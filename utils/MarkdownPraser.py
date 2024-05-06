import io
import os
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from io import StringIO
from typing import Tuple, Dict

import yaml
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from streamlit.runtime.uploaded_file_manager import UploadedFile
from loguru import logger
from yaml import Dumper


class PaperType(IntEnum):
    PURE_MARKDOWN = 0
    GROBID_PAPER = 1
    PMC_PAPER = 2
    PM_INFO = 3
    NSFC = 4


def type_representer(dumper: Dumper, _data):
    return dumper.represent_int(_data.value)


yaml.add_representer(PaperType, type_representer)


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


def split_markdown(document: UploadedFile) -> Tuple[list[Document], Reference]:
    """
    分割Markdown文档为多个部分。

    :param document: 上传的文件对象，预期为一个包含Markdown文本的文件。
    :return: 返回一个分割后的Markdown文档部分的列表。
    """

    stringio = StringIO(document.getvalue().decode("utf-8"))
    string_data = stringio.read()
    md_docs, reference_data = split_markdown_text(string_data)
    return md_docs, reference_data


def split_markdown_text(md_text: str) -> Tuple[list[Document], Reference]:
    """
    分割Markdown文本。

    该函数首先根据Markdown中的标题（#，##，###，####）进行分割，然后将文本进一步拆分为多个文档块，
    每个文档块保持相对的完整性。此外，如果提供了`year`、`doi`和`author`参数，则会将这些信息添加到
    每个拆分后的文档的元数据中；如果没有提供这些参数，则从Markdown文本的元数据部分自动提取。

    :param md_text: 要分割的Markdown文本字符串。
    :return: 返回一个元组，包含解析后的 Document列表和包含引用信息的一个字典
    """

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[('#', 'title'), ('##', 'section'), ('###', 'subtitle'), ('####', 'subtitle')]
    )
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=0,
        separators=['\n\n', '\n', '\t\n'],
        keep_separator=False
    )

    head_split_docs = md_splitter.split_text(md_text)

    if head_split_docs[0].page_content.startswith('---'):
        yaml_text = head_split_docs.pop(0).page_content.replace('---', '')
        data = yaml.load(yaml_text, Loader=yaml.FullLoader)
        paper_info = PaperInfo.from_yaml(data)
    else:
        raise Exception('Markdown miss information!')

    reference_data = []
    if paper_info.ref:
        if head_split_docs[-1].metadata['section'] != 'Reference' or head_split_docs[-1].metadata['section'] != '参考文献':
            raise Exception('Missing "Reference" section')
        else:
            head_split_docs.pop(-1)
            md_text.replace('## 参考文献', '## Reference')
            reference_text = md_text.split('## Reference')[-1].rstrip('\t\n').lstrip('\t\n')
            reference_data = yaml.load(reference_text, Loader=yaml.FullLoader)

    for doc in head_split_docs:
        doc.metadata['author'] = paper_info.author
        doc.metadata['year'] = paper_info.year
        doc.metadata['type'] = paper_info.type
        doc.metadata['keywords'] = paper_info.keywords
        doc.metadata['doi'] = paper_info.doi

    md_docs = r_splitter.split_documents(head_split_docs)

    return md_docs, Reference(paper_info.doi, reference_data)


def split_paper(paper: Paper) -> Tuple[list[Document], Reference]:
    """
    将给定的章节列表转换为文档列表

    :param paper: Paper数据类，包含
    :return: 返回一个元组，包含解析后的 Document列表和包含引用信息的一个字典
    """

    md_stream = io.StringIO()

    md_stream.write('---\t\n')
    yaml.dump(asdict(paper.info), md_stream, sort_keys=False, width=900)
    md_stream.write('---\t\n')

    for sec in paper.sections:
        text = sec.text
        level = sec.level
        if level == 0:
            md_stream.write(f'{text}\t\n')
        elif level == 1:
            md_stream.write(f'# {text}\t\n')
        elif level == 2:
            md_stream.write(f'## {text}\t\n')
        elif level == 3:
            md_stream.write(f'### {text}\t\n')
        else:
            md_stream.write(f'#### {text}\t\n')

    if paper.info.ref:
        md_stream.write('## Reference\t\n')
        yaml.dump(paper.reference.ref_list, md_stream, sort_keys=False, width=900)

    md_text = md_stream.getvalue()

    return split_markdown_text(md_text)


def save_to_md(paper: Paper, output_path) -> None:
    """
    将章节列表保存为Markdown格式的文件。

    :param paper: 具体的文献数据
    :param output_path: 输出Markdown文件的路径。
    :return: 无返回值。
    """

    with open(output_path, 'w', encoding='utf-8') as f:

        f.write('---\t\n')
        yaml.dump(asdict(paper.info), f, sort_keys=False, width=900)
        f.write('---\t\n')

        for sec in paper.sections:
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

        f.write('## Reference\t\n')
        yaml.dump(paper.reference.ref_list, f, sort_keys=False, width=900)


def load_from_md(path: str) -> Tuple[list[Document], Reference]:
    """
    从给定的Markdown文件路径加载内容，并将其分割为文档列表和元数据字典。

    :param path: 指向Markdown文件的路径。
    :return: 一个元组，包含一个文档列表和一个元数据字典。
    """
    # 检查路径是否指向一个文件
    if not os.path.isfile(path):
        logger.error('This is not a file path!')
        return [], {}

    # 检查文件扩展名是否为.md
    if not path.endswith('.md'):
        logger.error('This is not a markdown file!')
        return [], {}

    with open(path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    return split_markdown_text(md_text)
