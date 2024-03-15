import json
import os.path
import random
import re
from enum import Enum
from typing import Tuple

import requests
import pandas as pd
from bs4 import BeautifulSoup, Tag, NavigableString
from loguru import logger
from pandas import DataFrame
from requests import sessions

from Config import config
from utils.FileUtil import Section, replace_multiple_spaces
from utils.DecoratorUtil import timer, retry


class RefType(Enum):
    SINGLE = 0
    MULTI = 1


class RefIdType(Enum):
    FIXED = 0
    UNFIXED = 1


fix_journal = {'1176-9114'}

id_length = RefIdType.UNFIXED


@timer
@retry(delay=random.uniform(2.0, 5.0))
def get_pmc_id(term: str, file_name: str = 'pmlist.csv') -> None:
    url = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&term={term}'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    response = requests.request("GET", url, headers=headers, timeout=10)

    if response.status_code == 200:
        data = json.loads(response.text)

        pmc_list = data['esearchresult']['idlist']
        df = pd.DataFrame({'title': pd.NA, 'pmc_id': pmc_list, 'doi': pd.NA, 'year': pd.NA})

        df.to_csv(file_name, mode='w', index=False, encoding='utf-8')
    else:
        raise Exception('下载请求失败')


@retry(delay=random.uniform(2.0, 5.0))
def download_paper_data(pmc_id: str) -> Tuple[int, dict]:
    url = (f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmc_id}'
           f'&retmode=xml&api_key={config.pubmed_config.API_KEY}')

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    with sessions.Session() as session:
        if config.pubmed_config.USE_PROXY:
            proxies = {
                'http': config.get_proxy(),
                'https': config.get_proxy()
            }
            response = session.request("GET", url, headers=headers, proxies=proxies, timeout=10)
        else:
            response = session.request("GET", url, headers=headers, timeout=10)

    if response.status_code == 200:

        soup = BeautifulSoup(response.text, 'xml')

        doi = soup.find('article-id', {'pub-id-type': 'doi'}).text \
            if soup.find('article-id', {'pub-id-type': 'doi'}) \
            else None

        year = soup.find('pub-date').find('year').text \
            if soup.find('pub-date') \
            else None

        xml_path = os.path.join(config.get_xml_path(), year, doi.replace('/', '@') + '.xml') if doi else None

        if xml_path:
            os.makedirs(os.path.dirname(xml_path), exist_ok=True)

            with open(xml_path, 'w', encoding='utf-8') as f:
                f.write(response.text)

        return response.status_code, {
            'year': year,
            'doi': doi,
            'output_path': xml_path
        }
    else:
        raise Exception('下载请求失败')


def parse_paper_data(xml_text: str, year: str, doi: str) -> dict:
    soup = BeautifulSoup(xml_text, 'xml')

    try:
        issn_block = soup.find('front').find('journal-meta').find('issn', {'pub-type': 'ppub'})
        if not issn_block:
            issn_block = soup.find('front').find('journal-meta').find('issn', {'pub-type': 'epub'})

        issn = issn_block.text if issn_block else None
    finally:
        issn = None

    if issn in fix_journal:
        global id_length
        id_length = RefIdType.FIXED

    sections: list[Section] = []

    title = soup.find('article-title').text.replace('\n', ' ') \
        if soup.find('article-title') \
        else None
    title = replace_multiple_spaces(title)

    sections.append(Section(title, 1))

    author_block = soup.find('contrib-group').find('name')
    author = __extract_author_name(author_block) if author_block else None

    abs_block = soup.find('abstract')
    main_sections = soup.select_one('body')

    norm = True

    ref_block = soup.select_one('back')
    if ref_block:
        pass
    else:
        logger.warning(f'{doi} has no reference')
        return {
            'title': title,
            'author': author,
            'year': year,
            'doi': doi,
            'sections': sections,
            'norm': False
        }

    if abs_block:
        sections.append(Section('Abstract', 2))
        sections = __solve_section(abs_block, sections, 2, ref_block)
    else:
        logger.warning(f'{doi} has no Abstract')
        return {
            'title': title,
            'author': author,
            'year': year,
            'doi': doi,
            'sections': sections,
            'norm': False
        }

    if main_sections:
        sections = __solve_section(main_sections, sections, 1, ref_block)

    return {
        'title': title,
        'author': author,
        'year': year,
        'doi': doi,
        'sections': sections,
        'norm': norm
    }


def __extract_author_name(xml_block: BeautifulSoup | NavigableString | None) -> str:
    surname = xml_block.find('surname').text
    given_names = xml_block.find('given-names').text if xml_block.find('given-names') else ''

    initials = ' '.join([name[0] + '.' for name in given_names.split()])

    return f'{surname}, {initials}'


def __solve_section(
        soup: BeautifulSoup,
        sections: list[Section],
        title_level: int,
        ref_soup: BeautifulSoup | None
) -> list[Section]:
    """
    解析给定的BeautifulSoup对象，从中提取章节信息，并将其添加到sections列表中。

    :param soup: BeautifulSoup对象，代表待解析的HTML或XML文档的一部分。
    :param sections: Section对象列表，用于收集从文档中解析出的各个章节信息。
    :param title_level: 当前解析标题的层级，用于组织章节结构。
    :param ref_soup: 参考文献的BeautifulSoup对象，用于解析文档中的引用。
    :return: 更新后的Section对象列表。
    """
    # 尝试找到章节标题
    title = soup.find('title', recursive=False)
    if title:
        sections.append(Section(title.text, title_level))

    # 尝试找到所有的sec标签，递归解析它们
    section_list = soup.find_all('sec', recursive=False)
    if section_list:
        for sec in section_list:
            sections = __solve_section(sec, sections, title_level + 1, ref_soup)
    else:
        # 如果没有sec标签，尝试解析段落p标签
        p_tags = soup.select('p')
        for p_tag in p_tags:
            if p_tag and not p_tag.text == '':
                # 提取段落文本，处理换行符，并尝试找到引用信息
                section = p_tag.text.strip().replace('\n', ' ')
                ref_block = p_tag.find_all('xref', {'ref-type': 'bibr'})
                ref = __solve_ref(ref_soup, ref_block) if ref_block else ''
                sections.append(Section(section, 0, ref))

    return sections


def __solve_ref(ref_soup: BeautifulSoup, ref_list: list[Tag]) -> str:
    """
    解析参考文献列表，并根据其类型提取并转换为DOI列表。

    参数:
    - ref_soup: BeautifulSoup对象，包含参考文献的HTML解析树。
    - ref_list: Tag列表，每一个Tag代表一个参考文献的引用。

    返回值:
    - str: 通过逗号分隔的DOI字符串列表。
    """
    global id_length
    rid_list = []
    for ref in ref_list:
        logger.debug(ref)  # 记录调试信息：当前处理的参考文献引用
        content = ref.text  # 获取参考文献的文本内容
        if is_single_reference(content) == RefType.SINGLE:
            # 单个参考文献处理
            rid_list.append(ref['rid'])
        elif is_single_reference(content) == RefType.MULTI:
            # 处理范围引用的参考文献
            if id_length == RefIdType.UNFIXED:
                _r_numbers = parse_range_string(content)  # 解析范围字符串
                _rid = remove_last_digit(ref['rid'])  # 移除最后一个数字
                logger.debug(_rid)  # 记录调试信息：处理后的引用ID
                rid_list.extend([f'{_rid}{i}' for i in _r_numbers])  # 生成范围内的ID列表
            else:
                _r_numbers = parse_range_string(content)  # 解析范围字符串
                _rid, digit = remove_digit_and_return(ref['rid'])  # 移除数字并返回位数
                logger.debug(_rid)  # 记录调试信息：处理后的引用ID
                rid_list.extend([f'{_rid}{str(i).zfill(digit)}' for i in _r_numbers])  # 生成对齐位数的范围ID列表
        else:
            logger.error('unknown type')  # 记录错误：未知的引用类型

    rid_list = sorted(list(set(rid_list)))  # 去重并排序引用ID列表
    logger.debug(rid_list)  # 记录调试信息：处理后的引用ID列表

    doi_list = []
    for ref_id in rid_list:
        ref_block = ref_soup.find(['ref', 'element-citation'], {'id': ref_id})
        if ref_block is None:
            ref_block = ref_soup.find(attrs={'id': ref_id})

        doi_block = ref_block.find('pub-id', {'pub-id-type': 'doi'})
        doi_list.append(doi_block.text) if doi_block else None

    return ','.join([x for x in doi_list])


def is_single_reference(text: str) -> RefType:
    if re.match(r'^\d+$', text):
        return RefType.SINGLE
    elif re.match(r'[0-9]+[,-][0-9]+', text):
        return RefType.MULTI
    else:
        return RefType.SINGLE


def remove_last_digit(input_string: str) -> str:
    return input_string.rstrip('0123456789')


def remove_digit_and_return(input_string: str) -> (str, int):
    result = remove_last_digit(input_string)
    num_removed = len(input_string) - len(result)
    return result, num_removed


def parse_range_string(input_str: str) -> list[int]:
    result = []
    parts = input_str.split(',')
    for part in parts:
        if '–' in part:
            start, end = map(int, part.split('–'))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return result
