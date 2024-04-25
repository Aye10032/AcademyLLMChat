import json
import os.path
import random
import re
from enum import Enum
from typing import Tuple, Dict, List

import requests
import pandas as pd
import yaml
from bs4 import BeautifulSoup, NavigableString
from loguru import logger
from requests import sessions

from Config import Config
from utils.FileUtil import replace_multiple_spaces
from utils.Decorator import timer, retry
from utils.MarkdownPraser import Section, PaperInfo, PaperType


class RefIdType(Enum):
    NORMAL = 0
    SB = 1
    GOOD = 2


fuck_journal = {'0377-0486', '0947-6539', '2156-7085', '1422-0067', '2578-9430'}
good_journal = {'1305-7456', '0148-639X', '2155-384X', '1838-7640'}

reference_type = RefIdType.NORMAL
mix_ref = False


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


def parse_paper_data(xml_text: str, silent: bool = True) -> Tuple[bool, List[Section]]:
    """
    从给定的XML文本中解析论文数据。

    :param xml_text: 论文的XML格式文本。
    :param silent: 是否静默运行
    :return: 格式化后的section列表
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
    global reference_type
    if issn and issn in fuck_journal:
        reference_type = RefIdType.SB
    elif issn and issn in good_journal:
        reference_type = RefIdType.GOOD
    else:
        reference_type = RefIdType.NORMAL

    # 初始化论文章节列表
    sections: list[Section] = []

    # 提取论文作者信息
    author_block = soup.find('contrib-group').find('name')
    author = __extract_author_name(author_block) if author_block else ''

    # 提取文章的DOI和发表年份
    year = soup.find('pub-date').find('year').text \
        if soup.find('pub-date') \
        else ''

    doi = soup.find('article-id', {'pub-id-type': 'doi'}).text \
        if soup.find('article-id', {'pub-id-type': 'doi'}) \
        else ''

    # 关键词
    keywords = []
    if soup.find(['keywords', 'kwd-group']):
        keyword_list = soup.find(['keywords', 'kwd-group']).find_all(['term', 'kwd'])

        for kw in keyword_list:
            keywords.append(kw.text.replace('\n', '').replace('\r', ' ').strip())
    if len(keywords) == 0:
        keywords.append('')

    sections.append(PaperInfo(
        author.replace('\n', '').replace('\r', ' ').strip(),
        int(year),
        PaperType.PMC_PAPER,
        ','.join(keywords),
        True,
        doi.replace('\n', '').replace('\r', '').strip()
    ).get_section())

    # 提取并处理论文标题
    title = soup.find('article-title').text.replace('\n', ' ') \
        if soup.find('article-title') \
        else None
    title = replace_multiple_spaces(title)
    sections.append(Section(title, 1))

    # 提取论文摘要和正文部分
    abs_block = soup.find('abstract')
    main_sections = soup.select_one('body')

    # 尝试提取论文引用信息
    ref_block = soup.find('ref-list')
    if ref_block is None or len(ref_block) == 0:
        if not silent:
            logger.warning(f'{doi} has no reference')
        return False, sections

    # 如果存在摘要，将其添加为一个章节，并处理摘要内容及引用信息
    if abs_block:
        sections.append(Section('Abstract', 2))
        sections = __solve_section(abs_block, sections, 2)
    else:
        if not silent:
            logger.warning(f'{doi} has no Abstract')
        return False, sections

    # 处理正文部分的章节信息
    if main_sections:
        sections = __solve_section(main_sections, sections, 1)

    # 添加参考文献
    global mix_ref
    sections.append(Section('Reference', 2))
    sections.append(Section(__extract_ref(ref_block, mix_ref), 0))

    # 返回解析后的论文信息
    return True, sections


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
        title_level: int
) -> list[Section]:
    """
    解析给定的BeautifulSoup对象，从中提取章节信息，并将其添加到sections列表中。

    :param soup: BeautifulSoup对象，代表待解析的HTML或XML文档的一部分。
    :param sections: Section对象列表，用于收集从文档中解析出的各个章节信息。
    :param title_level: 当前解析标题的层级，用于组织章节结构。
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
            sections = __solve_section(sec, sections, title_level + 1)
    else:
        # 如果没有sec标签，尝试解析段落p标签
        p_tags = soup.find_all('p', recursive=False)
        for p_tag in p_tags:
            if p_tag and not p_tag.text == '':
                matches = re.findall(r'\[\s*\d+\s*(?:,\s*\d+\s*)*]', p_tag.text)
                if matches:
                    # 处理傻逼格式
                    section_text = p_tag.text.strip().replace('\n', ' ')
                    section_text = deal_sb_paper(section_text)
                else:
                    # 提取段落文本，处理换行符，并尝试找到引用信息
                    sup_block = p_tag.find_all('sup', recursive=False)
                    found = False
                    for sup in sup_block:
                        tag = sup.find('xref', {'ref-type': 'bibr'})
                        if tag:
                            target_info = parse_range_string(sup.text)
                            found = True
                        else:
                            continue

                        sup.insert_after(''.join([f'[^{ref_id}]' for ref_id in target_info]))
                        sup.extract()

                    if p_tag.find('xref', {'ref-type': 'bibr'}) and not found:
                        ref_block = p_tag.find_all('xref', {'ref-type': 'bibr'}, recursive=False)
                        for ref_tag in ref_block:
                            target_info = ref_tag.text
                            ref_tag.insert_after(f'[^{target_info}]')
                            ref_tag.extract()

                        section_text = p_tag.text.strip().replace('\n', ' ')
                    else:
                        section_text = p_tag.text.strip().replace('\n', ' ')

                section = replace_multiple_spaces(section_text)
                sections.append(Section(section, 0))

    return sections


def deal_sb_paper(origin_str: str) -> str:
    matches = re.findall(r'\[\s*\d+\s*(?:,\s*\d+\s*)*]', origin_str)

    for match in matches:
        match_str: str = match
        new_ref = ''.join([
            f"[^{ind.replace(' ', '')}]"
            for ind in match_str.replace('[', '').replace(']', '').split(',')
        ])
        origin_str = origin_str.replace(match, new_ref)

    matches = re.findall(r'\[\s*\d+\s*–\s*\d+\s*]', origin_str)

    for match in matches:
        match_str: str = match
        new_ref = ''.join([
            f'[^{ind}]'
            for ind in parse_range_string(match_str.replace('[', '').replace(']', ''))
        ])
        origin_str = origin_str.replace(match, new_ref)

    return origin_str


def __extract_ref(ref_soup: BeautifulSoup, check_mix: bool = False) -> str:
    ref_blocks = ref_soup.find_all('ref', recursive=False)
    ref_list = []

    for ref_block in ref_blocks:
        element_blocks = ref_block.find_all(['element-citation', 'mixed-citation'], recursive=False)
        if len(element_blocks) == 1:
            ref_list.append(__get_ref_info(ref_block))
        else:
            if check_mix:
                temp = []
                for element_block in element_blocks:
                    temp.append(__get_ref_info(element_block))
                ref_list.append(temp)
            else:
                for element_block in element_blocks:
                    ref_list.append(__get_ref_info(element_block))

    return yaml.dump(ref_list)


def __get_ref_info(ref_block: BeautifulSoup) -> Dict:
    if ref_title_block := ref_block.find('article-title'):
        ref_title = ref_title_block.text.replace('\n', '').replace('\r', ' ')
    else:
        ref_title = ''

    if ref_doi_block := ref_block.find('pub-id', {'pub-id-type': 'doi'}):
        ref_doi = ref_doi_block.text
    else:
        ref_doi = ''

    if ref_pm_block := ref_block.find('pub-id', {'pub-id-type': 'pmid'}):
        ref_pm = ref_pm_block.text
    else:
        ref_pm = ''

    return {'title': ref_title, 'pmid': ref_pm, 'pmc': '', 'doi': ref_doi}


def remove_last_digit(input_string: str) -> str:
    return input_string.rstrip('0123456789')


def remove_digit_and_return(input_string: str) -> (str, int):
    result = remove_last_digit(input_string)
    num_removed = len(input_string) - len(result)
    return result, num_removed


def parse_range_string(input_str: str) -> list[int | str]:
    result = []
    parts = input_str.strip().split(',')
    # print(parts)
    for part in parts:
        if '–' in part:
            start, end = map(int, part.split('–'))
            result.extend(range(start, end + 1))
        elif '−' in part:
            start, end = map(int, part.split('−'))
            result.extend(range(start, end + 1))
        else:
            try:
                result.append(int(part))
            except ValueError:
                global mix_ref
                mix_ref = True
                result.append(part)

    return result
