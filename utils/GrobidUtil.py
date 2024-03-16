import os
from typing import LiteralString, Any, Dict

from grobid_client.grobid_client import GrobidClient
from bs4 import BeautifulSoup

from Config import Config
from utils.FileUtil import *
from loguru import logger

from utils.DecoratorUtil import timer


# TODO 传入配置文件

@timer
def parse_pdf(pdf_path: LiteralString | str, config: Config = None):
    """
    解析PDF文件，将其转换为XML格式。

    :param pdf_path: PDF文件或文件夹的路径，可以是字符串。
    :param config: 配置对象，包含处理PDF所需的配置信息。如果未提供，则创建默认配置。

    :return: 无返回值。
    """
    if config is None:
        config = Config()

    pdf_paths = []
    # 遍历pdf_path中所有包含PDF文件的文件夹，并将其绝对路径收集到pdf_paths列表中
    for root, dirs, files in os.walk(pdf_path):
        if len(files) > 0:
            pdf_paths.append(os.path.abspath(root))

    grobid_cfg = config.grobid_config
    # 初始化Grobid客户端，用于处理PDF
    client = GrobidClient(config_path=grobid_cfg.CONFIG_PATH)

    # 遍历pdf_paths中的每个PDF文件夹，将每个PDF文件转换为XML
    for path in pdf_paths:
        relative_path = os.path.relpath(path, pdf_path)
        xml_path = os.path.join(config.get_xml_path(), relative_path)
        logger.info(f'Parsing {path} to {xml_path}')
        client.process(grobid_cfg.SERVICE, path, output=xml_path, n=grobid_cfg.MULTI_PROCESS)

    # 清空pdf_paths列表
    pdf_paths.clear()


def parse_pdf_to_xml(pdf_path: LiteralString | str | bytes, config: Config = None) -> (Any, int, str):
    """
    将PDF文件解析为XML格式。

    :param pdf_path: PDF文件的路径，可以是字符串、字节序列或LiteralString类型。
    :param config: 用于配置Grobid客户端的Config对象，可选。如果未提供，则使用默认配置。

    :return: 一个元组，包含处理结果、HTTP状态码和响应内容类型。
    """
    if config is None:
        config = Config()

    grobid_cfg = config.grobid_config
    client = GrobidClient(config_path=grobid_cfg.CONFIG_PATH)  # 初始化Grobid客户端
    return client.process_pdf(grobid_cfg.SERVICE, pdf_path, False, True, False, False, False, False, False)


def parse_xml(xml_path: LiteralString | str | bytes) -> Dict:
    """
    解析XML文件，提取相关信息。

    :param xml_path: XML文件的路径，可以是字符串路径、字节序列或LiteralString。
    :return: 包含标题、作者、出版年份、摘要、关键词和章节信息的字典。

    提取的信息包括：
    - 标题：从XML中找到的主要标题。
    - 作者：从XML中提取的所有作者信息。
    - 出版年份：从XML中找到的出版年份。
    - 摘要：从XML中提取的摘要内容。
    - 关键词：从XML中提取的关键词列表。
    - 章节信息：从XML中提取的各章节标题及其级别。
    """

    with open(xml_path, 'r', encoding='utf-8') as f:
        xml_data = f.read()
        soup = BeautifulSoup(xml_data, 'xml')

        # 提取XML中的标题
        title = soup.find('titleStmt').find('title', {'type': 'main'})
        title = format_filename(title.text.strip()) if title is not None else ''

        # 提取作者信息
        authors = []
        for author in soup.find('sourceDesc').find_all('persName'):
            first_name = author.find('forename', {'type': 'first'})
            first_name = first_name.text.strip() if first_name is not None else ''
            middle_name = author.find('forename', {'type': 'middle'})
            middle_name = middle_name.text.strip() if middle_name is not None else ''
            last_name = author.find('surname')
            last_name = last_name.text.strip() if last_name is not None else ''

            if middle_name != '':
                authors.append(__extract_author_name(last_name, f'{first_name} {middle_name}'))
            else:
                authors.append(__extract_author_name(last_name, first_name))

        # 提取出版年份
        pub_date = soup.find('publicationStmt')
        year = pub_date.find('date')
        year = year.attrs.get('when') if year is not None else ''

        # 提取摘要
        abstract = ''
        abstract_list = soup.find('profileDesc').select('abstract p')
        for p in abstract_list:
            abstract += p.text.strip() + ' '

        # 提取关键词
        key_div = soup.find('profileDesc').select('keywords')
        keywords = split_words(key_div[0].get_text()) if len(key_div) != 0 else []

        # 提取章节信息
        sections: list[Section] = []
        for section in soup.find('body').find_all('div'):
            if section.find('head') is None:
                continue
            section_title = section.find('head').text.strip()
            title_level = section.find('head').attrs.get('n')
            level = title_level.count('.') + 1 if title_level else 2

            sections.append(Section(section_title, level))

            for p in section.find_all('p'):
                text = replace_multiple_spaces(p.text.strip())
                if text:
                    sections.append(Section(text, 0))

    return {'title': title,
            'authors': authors,
            'year': year,
            'abstract': abstract,
            'keywords': keywords,
            'sections': sections}


def __extract_author_name(surname, given_names) -> str:
    """
    提取作者的姓名首字母缩写和姓氏。

    :param surname (str): 作者的姓氏。
    :param given_names (str): 作者的名字，可以包含多个名字。

    :return: 格式化后的作者姓名，格式为“姓, 名首字母缩写”。
    """

    initials = ' '.join([name[0] + '.' for name in given_names.split()])

    return f'{surname}, {initials}'
