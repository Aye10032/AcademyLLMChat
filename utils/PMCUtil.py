import json
import os.path
import re
from enum import Enum

import requests
import pandas as pd
from bs4 import BeautifulSoup, Tag, NavigableString
from loguru import logger
from requests import sessions

from Config import config
from utils.FileUtil import Section, replace_multiple_spaces
from utils.TimeUtil import timer


class RefType(Enum):
    SINGLE = 0
    MULTI = 1


class RefIdType(Enum):
    FIXED = 0
    UNFIXED = 1


fix_journal = {'1176-9114'}

id_length = RefIdType.UNFIXED


@timer
def get_pmc_id(term: str, file_name: str = 'pmlist.csv') -> None:
    url = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&term={term}'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    response = requests.request("GET", url, headers=headers, timeout=10)

    data = json.loads(response.text)

    pmc_list = data['esearchresult']['idlist']
    df = pd.DataFrame({'title': pd.NA, 'pmc_id': pmc_list, 'doi': pd.NA, 'year': pd.NA})

    df.to_csv(file_name, mode='w', index=False, encoding='utf-8')


def download_paper_data(pmc_id: str) -> dict:
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

        soup = BeautifulSoup(response.text, 'xml')

        doi = soup.find('article-id', {'pub-id-type': 'doi'}).text \
            if soup.find('article-id', {'pub-id-type': 'doi'}) \
            else None

        year = soup.find('pub-date').find('year').text \
            if soup.find('pub-date') \
            else None

        xml_path = os.path.join(config.get_xml_path(), year, doi.replace('/', '@') + '.xml')
        os.makedirs(os.path.dirname(xml_path), exist_ok=True)

        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

    return {
        'year': year,
        'doi': doi,
        'output_path': xml_path
    }


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
        norm = False

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
    title = soup.find('title', recursive=False)
    if title:
        sections.append(Section(title.text, title_level))

    section_list = soup.find_all('sec', recursive=False)
    if section_list:
        for sec in section_list:
            sections = __solve_section(sec, sections, title_level + 1, ref_soup)
    else:
        p_tags = soup.select('p')
        for p_tag in p_tags:
            if p_tag and not p_tag.text == '':
                section = p_tag.text.strip().replace('\n', ' ')
                ref_block = p_tag.find_all('xref', {'ref-type': 'bibr'})
                ref = __solve_ref(ref_soup, ref_block) if ref_block else ''
                sections.append(Section(section, 0, ref))

    return sections


def __solve_ref(ref_soup: BeautifulSoup, ref_list: list[Tag]) -> str:
    global id_length
    rid_list = []
    for ref in ref_list:
        logger.debug(ref)
        content = ref.text
        if is_single_reference(content) == RefType.SINGLE:
            rid_list.append(ref['rid'])
        elif is_single_reference(content) == RefType.MULTI:
            if id_length == RefIdType.UNFIXED:
                _r_numbers = parse_range_string(content)
                _rid = remove_last_digit(ref['rid'])
                logger.debug(_rid)
                rid_list.extend([f'{_rid}{i}' for i in _r_numbers])
            else:
                _r_numbers = parse_range_string(content)
                _rid, digit = remove_digit_and_return(ref['rid'])
                logger.debug(_rid)
                rid_list.extend([f'{_rid}{str(i).zfill(digit)}' for i in _r_numbers])
        else:
            logger.error('unknown type')

    rid_list = sorted(list(set(rid_list)))
    logger.debug(rid_list)

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
