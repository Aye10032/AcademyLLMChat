from grobid_client.grobid_client import GrobidClient
from bs4 import BeautifulSoup

from Config import config
from utils.FileUtil import *
from loguru import logger

from utils.TimeUtil import timer


@timer
def parse_pdf(pdf_path: str):
    """
    批量解析根目录下的PDF文件，并按照原目录结构保存为MD文件
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
        xml_path = os.path.join(config.XML_OUTPUT, relative_path)
        logger.info(f'Parsing {path} to {xml_path}')
        __parse_pdf_to_xml(path, xml_path)
    pdf_paths.clear()

    # 解析XML文件
    # xml_paths = []
    # for root, dirs, files in os.walk(XML_OUTPUT):
    #     for file in files:
    #         if file.endswith('.xml'):
    #             xml_paths.append(os.path.abspath(os.path.join(root, file)))
    #
    # for xml in xml_paths:
    #     relative_path = os.path.relpath(xml, XML_OUTPUT)
    #     md_file_path = os.path.join(MD_OUTPUT, relative_path).replace('.xml', '.md')
    #     logger.info(f'Parsing {xml} to {md_file_path}')
    #     result = __parse_xml(xml)
    #     save_to_md(result, md_file_path)


def __parse_pdf_to_xml(pdf_path: str, xml_path: str):
    grobid_cfg = config.grobid_config
    client = GrobidClient(config_path=grobid_cfg.CONFIG_PATH)
    client.process(grobid_cfg.SERVICE, pdf_path, output=xml_path, n=grobid_cfg.MULTI_PROCESS)


def parse_xml(xml_path: str):
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
        for section in soup.find('text').find_all('div', {'xmlns': 'http://www.tei-c.org/ns/1.0'}):
            if section.find('head') is None:
                continue
            section_title = section.find('head').text.strip()
            title_level = section.find('head').attrs.get('n')

            text = []
            for p in section.find_all('p'):
                text.append(replace_multiple_spaces(p.text.strip()))

            sections.append({'title': section_title, 'title_level': title_level, 'text': text})

    return {'title': title, 'authors': authors, 'year': year, 'abstract': abstract, 'keywords': keywords,
            'sections': sections}


def save_to_md(_dict: dict, output_path: str, append: bool):
    """
    将解析结果保存为markdown文件
    :param _dict: 解析结果
    :param output_path: 输出MD文件的根目录
    :param append: 是否追加写入，为真则略过标题和摘要
    :return:
    """
    title = _dict['title']
    abstract = _dict['abstract']
    sections = _dict['sections']

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    with open(output_path, 'a', encoding='utf-8') as f:
        if not append:
            f.write(f'# {title}\n\n')

            f.write(f'## Abstract\n\n{abstract}\n\n')

        father_title = ''
        father_level = ''
        check_sub = False
        has_sub = False
        for section in sections:
            section_title = section['title']
            title_level = section['title_level']
            text = section['text']

            if section_title.lower() in ['introduction', 'results', 'discussion']:
                f.write(f'## {section_title}\n\n')

                if title_level and title_level.endswith('.'):
                    father_title = section_title
                    father_level = title_level
                    check_sub = True
                    has_sub = False
                else:
                    check_sub = False
                    logger.warning(f'Section {section_title} has no title level')

                if text:
                    for paragraph in text:
                        f.write(paragraph + '\n\n')

                continue

            if check_sub:
                if title_level and title_level.startswith(father_level):
                    f.write(f'### {section_title}\n\n')
                    for paragraph in text:
                        f.write(paragraph + '\n\n')
                    has_sub = True

                if not has_sub:
                    logger.warning(f'Section {father_title} has no sub section')
                    check_sub = False


if __name__ == '__main__':
    # parse_pdf('../../../../DATA/document', config.MD_OUTPUT)
    data = parse_xml(config.XML_OUTPUT + '/2010/10.1016@j.biortech.2010.03.103.grobid.tei.xml')
    save_to_md(data, config.MD_OUTPUT + '/2010/10.1016@j.biortech.2010.03.103.md', True)
