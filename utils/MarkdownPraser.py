import re
from io import StringIO
from typing import List, Tuple, Any

import yaml
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from streamlit.runtime.uploaded_file_manager import UploadedFile


def split_markdown(document: UploadedFile):
    """
    分割Markdown文档为多个部分。

    :param document: 上传的文件对象，预期为一个包含Markdown文本的文件。
    :return: 返回一个分割后的Markdown文档部分的列表。
    """

    stringio = StringIO(document.getvalue().decode("utf-8"))
    string_data = stringio.read()
    md_docs = split_markdown_text(string_data)
    return md_docs


def split_markdown_text(md_text: str) -> Tuple[list[Document], dict[str, Any]]:
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
        author = data['author']
        year = data['year']
        _type = data['type']
        keywords = data['keywords']
        ref = data['ref'] if 'ref' in data else False
        doi = data['doi'] if 'doi' in data else ''
    else:
        raise Exception('Markdown miss information!')

    reference_data = {}
    if ref:
        if head_split_docs[-1].metadata['section'] != 'Reference':
            raise Exception('Missing "Reference" section')
        else:
            head_split_docs.pop(-1)
            reference_text = md_text.split('## Reference')[-1].rstrip('\t\n').lstrip('\t\n')
            # print('---------------')
            # print(reference_text)
            reference_data = yaml.load(reference_text, Loader=yaml.FullLoader)

    for doc in head_split_docs:
        doc.metadata['author'] = author
        doc.metadata['year'] = int(year)
        doc.metadata['type'] = _type
        doc.metadata['keywords'] = keywords
        doc.metadata['doi'] = doi

    md_docs = r_splitter.split_documents(head_split_docs)

    return md_docs, reference_data
