from grobid_client.grobid_client import GrobidClient
from bs4 import BeautifulSoup

from Config import config
from utils.FileUtil import *
from loguru import logger

from utils.TimeUtil import timer


@timer
def parse_pdf(pdf_path: str):
    """
    批量解析根目录下的PDF文件，并按照原目录结构保存为XML文件
    :param pdf_path: pdf文件的根目录
    :return:
    """
    # 读取根目录下的所有PDF文件
    pdf_paths = []
    for root, dirs, files in os.walk(pdf_path):
        if len(files) > 0:
            pdf_paths.append(os.path.abspath(root))

    # 解析PDF文件
    for path in pdf_paths:
        relative_path = os.path.relpath(path, pdf_path)
        xml_path = os.path.join(config.get_xml_path(), relative_path)
        logger.info(f'Parsing {path} to {xml_path}')
        parse_pdf_to_xml(path, xml_path)
    pdf_paths.clear()


def parse_pdf_to_xml(pdf_path: str, xml_path: str):
    """
    将pdf解析为xml文件
    :param pdf_path: 单个pdf文件或pdf文件所在的目录
    :param xml_path: xml文件输出目录
    :return:
    """
    grobid_cfg = config.grobid_config
    client = GrobidClient(config_path=grobid_cfg.CONFIG_PATH)
    client.process(grobid_cfg.SERVICE, pdf_path, output=xml_path, n=grobid_cfg.MULTI_PROCESS)


def parse_xml(xml_path: str):
    """
    解析XML文件
    :param xml_path: xml文件的路径
    :return:
        title: 论文标题,
        authors: 作者列表,
        year: 论文年份,
        abstract: 摘要,
        keywords: 关键词列表,
        sections: markdown结构段落
    """
    with open(xml_path, 'r', encoding='utf-8') as f:
        xml_data = f.read()
        soup = BeautifulSoup(xml_data, 'xml')

        # Title
        title = soup.find('titleStmt').find('title', {'type': 'main'})
        title = format_filename(title.text.strip()) if title is not None else ''

        # Authors
        authors = []
        for author in soup.find('sourceDesc').find_all('persName'):
            first_name = author.find('forename', {'type': 'first'})
            first_name = first_name.text.strip() if first_name is not None else ''
            middle_name = author.find('forename', {'type': 'middle'})
            middle_name = middle_name.text.strip() if middle_name is not None else ''
            last_name = author.find('surname')
            last_name = last_name.text.strip() if last_name is not None else ''
            if middle_name != '':
                authors.append(f'{first_name} {middle_name} {last_name}')
            else:
                authors.append(f'{first_name} {last_name}')

        # date
        pub_date = soup.find('publicationStmt')
        year = pub_date.find('date')
        year = year.attrs.get('when') if year is not None else ''

        # Abstract
        abstract = ''
        abstract_list = soup.find('profileDesc').select('abstract p')
        for p in abstract_list:
            abstract += p.text.strip() + ' '

        # keywords
        key_div = soup.find('profileDesc').select('keywords')
        keywords = split_words(key_div[0].get_text()) if len(key_div) != 0 else []

        # sections
        sections = []
        for section in soup.find('body').find_all('div', {'xmlns': 'http://www.tei-c.org/ns/1.0'}):
            if section.find('head') is None:
                continue
            section_title = section.find('head').text.strip()
            title_level = section.find('head').attrs.get('n')
            level = title_level.count('.') + 1 if title_level else 2

            text = []
            for p in section.find_all('p'):
                text.append(replace_multiple_spaces(p.text.strip()))

            sections.append({'text': section_title, 'level': level})
            if text:
                sections.append({'text': text, 'level': 0})

    return {'title': title, 'authors': authors, 'year': year, 'abstract': abstract, 'keywords': keywords,
            'sections': sections}


if __name__ == '__main__':
    # parse_pdf('../../../../DATA/document', config.MD_OUTPUT)
    data = parse_xml(config.get_xml_path() + '/2010/10.1016@j.biortech.2010.03.103.grobid.tei.xml')
    save_to_md(data, config.get_md_path() + '/2010/10.1016@j.biortech.2010.03.103.md')
