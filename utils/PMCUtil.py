import json
import os.path
import random
import re
from enum import Enum
from typing import Tuple, Dict

import requests
import pandas as pd
from bs4 import BeautifulSoup, Tag, NavigableString, ResultSet
from loguru import logger
from pandas import DataFrame
from requests import sessions

from Config import Config
from utils.FileUtil import Section, replace_multiple_spaces
from utils.Decorator import timer, retry


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
    """
    根据提供的搜索术语获取PMC（PubMed Central）文章ID，并将结果保存到CSV文件中。

    :param term: 搜索术语，用于在PMC数据库中进行搜索
    :param file_name: 保存结果的CSV文件名，默认为'pmlist.csv'
    :return: 无返回值
    """

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
def download_paper_data(pmc_id: str, config: Config = None) -> Tuple[int, dict]:
    """
    下载指定PMC文章的XML数据，并将其存储到本地文件系统中。

    :param pmc_id: 文章的PMC标识符。
    :param config: 包含配置信息的对象，如API密钥和代理设置。如果未提供，则使用默认配置。
    :return: 一个元组，包含HTTP响应状态码和一个字典，字典包含文章的年份、DOI和输出路径。
    """
    # 检查是否提供了配置对象，未提供则使用默认配置
    if config is None:
        config = Config()

    # 构建下载文章数据的URL
    url = (f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmc_id}'
           f'&retmode=xml&api_key={config.pubmed_config.API_KEY}')

    # 设置HTTP请求头，伪装为Chrome浏览器发送请求
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    # 使用Session对象进行HTTP请求，支持使用代理
    with sessions.Session() as session:
        if config.pubmed_config.USE_PROXY:
            # 如果配置了使用代理，则从配置中获取代理地址
            proxies = {
                'http': config.get_proxy(),
                'https': config.get_proxy()
            }
            response = session.request("GET", url, headers=headers, proxies=proxies, timeout=10)
        else:
            # 否则直接发送请求
            response = session.request("GET", url, headers=headers, timeout=10)

    # 检查HTTP响应状态码，若非200，则抛出异常
    if response.status_code == 200:
        # 使用BeautifulSoup解析XML响应内容
        soup = BeautifulSoup(response.text, 'xml')

        # 提取文章的DOI和发表年份
        doi = soup.find('article-id', {'pub-id-type': 'doi'}).text \
            if soup.find('article-id', {'pub-id-type': 'doi'}) \
            else None

        pmid = soup.find('article-id', {'pub-id-type': 'pmid'}).text \
            if soup.find('article-id', {'pub-id-type': 'pmid'}) \
            else None

        year = soup.find('pub-date').find('year').text \
            if soup.find('pub-date') \
            else None

        # 根据DOI和年份构建本地存储路径
        xml_path = os.path.join(config.get_xml_path(), year, doi.replace('/', '@') + '.xml') if doi else None

        # 如果路径存在，创建目录并将XML数据写入文件
        if xml_path:
            os.makedirs(os.path.dirname(xml_path), exist_ok=True)

            with open(xml_path, 'w', encoding='utf-8') as f:
                f.write(response.text)

        # 返回HTTP状态码和相关元数据
        return response.status_code, {
            'year': year,
            'doi': doi,
            'pmid': pmid,
            'output_path': xml_path
        }
    else:
        # 如果请求失败，抛出异常
        raise Exception('下载请求失败')


def parse_paper_data(xml_text: str, year: str, doi: str, silent: bool = True) -> dict:
    """
    解析论文数据从给定的XML文本中。

    :param xml_text: 论文的XML格式文本。
    :param year: 论文的出版年份。
    :param doi: 论文的数字对象标识符（DOI）。
    :param silent: 是否静默运行
    :return: 包含论文标题、作者、年份、DOI、章节和规范化状态的字典。

    此函数从XML文本中提取论文的相关信息，包括标题、作者、摘要、章节以及引用信息，并将这些信息组织成一个字典返回。
    """

    # 使用BeautifulSoup解析XML文本
    soup = BeautifulSoup(xml_text, 'xml')

    try:
        # 尝试获取纸质版本的ISSN号，如果不存在则获取电子版本的ISSN号
        issn_block = soup.find('front').find('journal-meta').find('issn', {'pub-type': 'ppub'})
        if not issn_block:
            issn_block = soup.find('front').find('journal-meta').find('issn', {'pub-type': 'epub'})

        issn = issn_block.text if issn_block else None
    except Exception as e:
        if not silent:
            logger.warning(e)
        issn = None  # 如果无法获取ISSN号，则最终设置为None

    # 检查ISSN号是否需要修正，并设置相应的标识
    if issn in fix_journal:
        global id_length
        id_length = RefIdType.FIXED

    # 初始化论文章节列表
    sections: list[Section] = []

    # 提取并处理论文标题
    title = soup.find('article-title').text.replace('\n', ' ') \
        if soup.find('article-title') \
        else None
    title = replace_multiple_spaces(title)
    sections.append(Section(title, 1))

    # 提取论文作者信息
    author_block = soup.find('contrib-group').find('name')
    author = __extract_author_name(author_block) if author_block else None

    # 提取论文摘要和正文部分
    abs_block = soup.find('abstract')
    main_sections = soup.select_one('body')

    norm = True

    # 尝试提取论文引用信息
    ref_block = soup.find_all('ref-list')
    if len(ref_block) == 0:
        if not silent:
            logger.warning(f'{doi} has no reference')
        return {
            'title': title,
            'author': author,
            'year': year,
            'doi': doi,
            'sections': sections,
            'norm': False,
        }

    # 如果存在摘要，将其添加为一个章节，并处理摘要内容及引用信息
    if abs_block:
        sections.append(Section('Abstract', 2))
        sections = __solve_section(abs_block, sections, 2, ref_block[0])
    else:
        if not silent:
            logger.warning(f'{doi} has no Abstract')
        return {
            'title': title,
            'author': author,
            'year': year,
            'doi': doi,
            'sections': sections,
            'norm': False,
        }

    # 处理正文部分的章节信息
    if main_sections:
        sections = __solve_section(main_sections, sections, 1, ref_block[0])

    # 返回解析后的论文信息
    return {
        'title': title,
        'author': author,
        'year': year,
        'doi': doi,
        'sections': sections,
        'norm': norm,
    }


def __extract_author_name(xml_block: BeautifulSoup | NavigableString | None) -> str:
    """
    从XML块中提取作者的姓名。

    :param xml_block: 包含作者信息的BeautifulSoup对象或NavigableString对象，或为None。
    :return: 格式化后的作者姓名，格式为"姓, 名首字母."。
    """
    # 提取作者的姓氏
    surname = xml_block.find('surname').text
    # 尝试提取作者的名字，如果不存在则默认为空字符串
    given_names = xml_block.find('given-names').text if xml_block.find('given-names') else ''

    # 将名字拆分并提取首字母，形成缩写
    initials = ' '.join([name[0] + '.' for name in given_names.split()])

    # 返回格式化后的姓名
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


def __extract_ref(ref_soup: BeautifulSoup) -> DataFrame:
    ref_blocks = ref_soup.find_all('ref', recursive=False)
    ref_list = []

    for ref_block in ref_blocks:
        element_blocks = ref_block.find_all(['element-citation', 'mixed-citation'], recursive=False)
        if len(element_blocks) == 1:
            ref_list.append(__get_ref_info(ref_block))
        else:
            for element_block in element_blocks:
                ref_list.append(__get_ref_info(element_block))

    return pd.DataFrame(ref_list)


def __get_ref_info(ref_block: BeautifulSoup) -> Dict:
    if ref_rid := ref_block.get('id'):
        rid = ref_rid
    else:
        rid = pd.NA

    if ref_doi_block := ref_block.find('pub-id', {'pub-id-type': 'doi'}):
        ref_doi = ref_doi_block.text
    else:
        ref_doi = pd.NA

    if ref_pm_block := ref_block.find('ArticleId', {'IdType': 'pmid'}):
        ref_pm = ref_pm_block.text
    else:
        ref_pm = pd.NA

    return {'rid': rid, 'doi': ref_doi, 'pubmed': ref_pm}


def __solve_ref(ref_soup: BeautifulSoup, ref_list: list[Tag]) -> str:
    """
    处理参考文献引用，根据不同的引用类型生成对应的DOI列表。

    :param ref_soup: BeautifulSoup对象，包含参考文献的HTML文档片段
    :param ref_list: Tag列表，每一个Tag代表一个参考文献引用
    :return: 字符串，包含所有参考文献的DOI，通过逗号分隔
    """
    global id_length
    rid_list = []
    for ref in ref_list:
        content = ref.text
        if is_single_reference(content) == RefType.SINGLE:
            # 处理单个参考文献引用
            rid_list.append(ref['rid'])
        elif is_single_reference(content) == RefType.MULTI:
            # 处理范围引用的参考文献
            if id_length == RefIdType.UNFIXED:
                _r_numbers = parse_range_string(content)  # 解析范围字符串
                _rid = remove_last_digit(ref['rid'])  # 移除最后一个数字
                rid_list.extend([f'{_rid}{i}' for i in _r_numbers])  # 生成范围内的ID列表
            else:
                _r_numbers = parse_range_string(content)  # 解析范围字符串
                _rid, digit = remove_digit_and_return(ref['rid'])  # 移除数字并返回位数
                rid_list.extend([f'{_rid}{str(i).zfill(digit)}' for i in _r_numbers])  # 生成对齐位数的范围ID列表
        else:
            logger.error('unknown type')  # 记录未知的引用类型错误

    rid_list = sorted(list(set(rid_list)))  # 去重并排序引用ID列表
    logger.debug(rid_list)  # 记录处理后的引用ID列表

    doi_list = []
    for ref_id in rid_list:
        # 查找对应的参考文献块
        ref_block = ref_soup.find(attrs={'id': ref_id})

        # 尝试获取DOI
        if ref_block:
            doi_block = ref_block.find('pub-id', {'pub-id-type': 'doi'})
            doi_list.append(doi_block.text) if doi_block else None

    return ','.join([x for x in doi_list])  # 返回DOI列表，通过逗号分隔


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
