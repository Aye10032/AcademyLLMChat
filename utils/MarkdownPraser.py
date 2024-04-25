import os
import re
from dataclasses import dataclass, asdict
from enum import IntEnum
from io import StringIO
from typing import List, Tuple, Any, Dict

import yaml
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from streamlit.runtime.uploaded_file_manager import UploadedFile
from loguru import logger


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
    author: str = ''
    year: int = -1
    type: int = PaperType.PURE_MARKDOWN
    keywords: str = ''
    ref: bool = False
    doi: str = ''

    def get_section(self) -> Section:
        info_dict = asdict(self)
        info_dict['type'] = int(self.type)
        block_str = f'---\t\n{info_dict}\t\n---\t\n'
        return Section(block_str, 0)

    @classmethod
    def from_yaml(cls, data: Dict[str, any]):
        return cls(**data)


def split_markdown(document: UploadedFile) -> Tuple[list[Document], Dict[str, Any]]:
    """
    分割Markdown文档为多个部分。

    :param document: 上传的文件对象，预期为一个包含Markdown文本的文件。
    :return: 返回一个分割后的Markdown文档部分的列表。
    """

    stringio = StringIO(document.getvalue().decode("utf-8"))
    string_data = stringio.read()
    md_docs, reference_data = split_markdown_text(string_data)
    return md_docs, reference_data


def split_markdown_text(md_text: str) -> Tuple[list[Document], Dict[str, Any]]:
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
        if head_split_docs[-1].metadata['section'] != 'Reference':
            raise Exception('Missing "Reference" section')
        else:
            head_split_docs.pop(-1)
            reference_text = md_text.split('## Reference')[-1].rstrip('\t\n').lstrip('\t\n')
            reference_data = yaml.load(reference_text, Loader=yaml.FullLoader)

    for doc in head_split_docs:
        doc.metadata['author'] = paper_info.author
        doc.metadata['year'] = paper_info.year
        doc.metadata['type'] = paper_info.type
        doc.metadata['keywords'] = paper_info.keywords
        doc.metadata['doi'] = paper_info.doi

    md_docs = r_splitter.split_documents(head_split_docs)

    return md_docs, {'source_doi': paper_info.doi, 'ref_data': reference_data}


def save_to_md(sections: list[Section], output_path) -> None:
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


def load_from_md(path: str) -> Tuple[list[Document], Dict[str, Any]]:
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
