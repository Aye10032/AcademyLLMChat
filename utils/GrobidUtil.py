import os.path
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import StrEnum
from pathlib import Path
from typing import LiteralString, Any, Tuple

import requests
from bs4 import BeautifulSoup
from requests import RequestException, Response
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3 import Retry

from Config import GrobidConfig

from utils.MarkdownPraser import *
from utils.FileUtil import *


class ConsolidateHeader(StrEnum):
    NO_CONSOLIDATION = '0'
    ALL_METADATA = '1'
    CITATION_AND_DOI = '2'
    DOI_ONLY = '3'


class ConsolidateCitations(StrEnum):
    NO_CONSOLIDATION = '0'
    ALL_METADATA = '1'
    CITATION_AND_DOI = '2'


class ConsolidateFunders(StrEnum):
    NO_CONSOLIDATION = '0'
    ALL_METADATA = '1'
    CITATION_AND_DOI = '2'


class GrobidConnector:
    def __init__(self, config: GrobidConfig):
        self.server_url = f'{config.grobid_server}/api/{config.service}'
        self.check_url = f'{config.grobid_server}/api/isalive'
        self.coordinates = config.coordinates
        self.timeout = config.timeout
        self.batch_size = config.batch_size
        self.max_works = config.multi_process

    def __enter__(self):
        self._check_server_status()
        self.session = requests.Session()

        retries = Retry(total=5, backoff_factor=5, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

        self.session.headers.update({
            'User-Agent': 'GrobidConnector/1.0',
            'Accept': 'application/xml'
        })

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()

    def _check_server_status(self):
        try:
            response = requests.get(self.check_url)
            response.raise_for_status()
        except RequestException as e:
            logger.error(f'[{e}]: Grobid server is unavailable.')
            raise ConnectionError('Grobid server is unavailable.')

    def parse_file(
            self,
            pdf_file: str | bytes,
            *,
            consolidate_header: str = ConsolidateHeader.ALL_METADATA,
            consolidate_citations: str = ConsolidateCitations.ALL_METADATA,
            consolidate_funders: str = ConsolidateFunders.NO_CONSOLIDATION,
            include_raw_citations: bool = True,
            include_raw_affiliations: bool = False,
            include_raw_copyrights: bool = False,
            segment_sentences: bool = False,
            generate_ids: bool = False,
            start: int = -1,
            end: int = -1
    ) -> tuple[str | bytes, int, str]:
        """
        Convert the complete input document into TEI XML format (header, body and bibliographical section).

        :param pdf_file: The PDF file to be parsed. Can be a file path or bytes.
        :param consolidate_header: The level of header consolidation. Default is ALL_METADATA.
        :param consolidate_citations: The level of citation consolidation. Default is ALL_METADATA.
        :param consolidate_funders: The level of funder consolidation. Default is NO_CONSOLIDATION.
        :param include_raw_citations: Whether to include raw citations in the output. Default is False.
        :param include_raw_affiliations: Whether to include raw affiliations in the output. Default is False.
        :param include_raw_copyrights: Whether to include raw copyrights in the output. Default is False.
        :param segment_sentences: Whether to segment sentences in the output. Default is False.
        :param generate_ids: Whether to generate IDs in the output. Default is False.
        :param start: The start page for parsing. Default is -1 (no limit).
        :param end: The end page for parsing. Default is -1 (no limit).
        :return: A tuple containing the HTTP status code and the response text.
        """
        with open(pdf_file, 'rb') as f:
            files = {
                "input": (
                    pdf_file,
                    f,
                    "application/pdf",
                    {"Expires": "0"},
                )
            }

            the_data = {
                "consolidateHeader": consolidate_header,
                "consolidateCitations": consolidate_citations,
                "consolidateFunders": consolidate_funders,
                "teiCoordinates": self.coordinates,
                "start": start,
                "end": end,
                "includeRawCitations": "1" if include_raw_citations else "0",
                "includeRawAffiliations": "1" if include_raw_affiliations else "0",
                "includeRawCopyrights": "1" if include_raw_copyrights else "0",
                "segmentSentences": "1" if segment_sentences else "0",
                "generateIDs": "1" if generate_ids else "0"
            }

            response = self.session.post(self.server_url, files=files, data=the_data, timeout=self.timeout)
            return pdf_file, response.status_code, response.text

    def __default_parse(self, pdf_file: str | bytes):
        return self.parse_file(pdf_file)

    def parse_files(self, pdf_path: str | bytes, output_path: str | bytes, multi_process: bool = False) -> None:
        file_list = [
            os.path.join(dir_path, filename)
            for dir_path, _, filenames in os.walk(pdf_path)
            for filename in filenames
            if filename.lower().endswith('.pdf')
        ]

        with tqdm(total=len(file_list), desc="Processing PDFs", unit="file") as pbar:
            if multi_process:
                for i in range(0, len(file_list), self.batch_size):
                    batch = file_list[i:i + self.batch_size]

                    with ThreadPoolExecutor(max_workers=self.max_works) as executor:
                        responses = [
                            executor.submit(
                                self.__default_parse,
                                file
                            ) for file in batch
                        ]

                        for response in as_completed(responses):
                            input_file, status, text = response.result()

                            if status == 200:
                                xml_file = os.path.join(
                                    output_path,
                                    Path(input_file).name.replace('.pdf', '.grobid.xml')
                                )
                                os.makedirs(output_path, exist_ok=True)
                                with open(xml_file, 'w', encoding='utf8') as f:
                                    f.write(text)
                            else:
                                logger.error(f'Parse {input_file} error.')

                            pbar.update(1)
            else:
                for file in file_list:
                    input_file, status, text = self.parse_file(file)

                    if status == 200:
                        xml_file = os.path.join(
                            output_path,
                            Path(input_file).name.replace('.pdf', '.grobid.xml')
                        )
                        os.makedirs(output_path, exist_ok=True)
                        with open(xml_file, 'w', encoding='utf8') as f:
                            f.write(text)
                    else:
                        logger.error(f'Parse {input_file} error.')

                    pbar.update(1)


def parse_xml(xml_path: LiteralString | str | bytes, sections: list = None) -> Paper:
    """
    解析XML文件，提取相关信息。

    :param xml_path: XML文件的路径，可以是字符串路径、字节序列或LiteralString。
    :param sections:
    :return: 格式化后的段落信息
    """

    if sections is None:
        sections: list[Section] = []
        append = False
    else:
        append = True

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
    match len(year):
        case 4:
            year = int(year)
        case 0:
            year = -1
        case _:
            try:
                year = int(year[:4])
            except ValueError:
                print(len(year))
                exit()

    doi = soup.find('sourceDesc').find('idno', {'type': 'DOI'})
    doi = doi.text if doi else ''

    # 提取关键词
    key_div = soup.find('profileDesc').find('keywords')
    keywords = split_words(key_div) if key_div is not None else ['']

    paper_info = PaperInfo(authors[0], year, PaperType.GROBID_PAPER, ','.join(keywords), True, doi)

    if not append:
        sections.append(Section(title, 1))

        # 提取摘要
        abstract_list = soup.find('profileDesc').select('abstract p')
        if len(abstract_list) > 0:
            sections.append(Section('Abstract', 2))
            for p in abstract_list:
                sections.append(Section(p.text.strip(), 0))

    # 提取章节信息
    for section in soup.find('body').find_all('div'):
        if section.find('head') is None:
            continue
        section_title = section.find('head').text.strip()
        title_level = section.find('head').attrs.get('n')
        if title_level:
            matches = re.findall(r'\d', title_level)
            level = len(matches) + 1
        else:
            level = 2

        sections.append(Section(section_title, level))

        for p in section.find_all('p'):
            ref_tags = p.find_all('ref', {'type': 'bibr'})

            for ref_tag in ref_tags:
                if 'target' in ref_tag:
                    target_info: str = ref_tag['target']
                    target_info = target_info.replace('#b', '')
                    ref_tag.insert_after(f'[^{target_info}]')

            text = replace_multiple_spaces(p.text.strip())
            if text:
                sections.append(Section(text, 0))

    # 提取引用信息
    ref_list = []
    for reference in soup.find('back').find_all('biblStruct'):
        ref_title = reference.find('title').text
        ref_list.append({'title': ref_title, 'pmid': '', 'pmc': '', 'doi': ''})

    return Paper(paper_info, sections, Reference(doi, ref_list))


def __extract_author_name(surname, given_names) -> str:
    """
    提取作者的姓名首字母缩写和姓氏。

    :param surname (str): 作者的姓氏。
    :param given_names (str): 作者的名字，可以包含多个名字。

    :return: 格式化后的作者姓名，格式为“姓, 名首字母缩写”。
    """

    initials = ' '.join([name[0] + '.' for name in given_names.split()])

    return f'{surname}, {initials}'
