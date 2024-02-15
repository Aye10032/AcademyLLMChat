import json

import requests
import pandas as pd
from bs4 import BeautifulSoup, Tag
from loguru import logger
from requests import sessions

from Config import config
from utils.FileUtil import Section
from utils.TimeUtil import timer


@timer
def get_pmc_id(term: str):
    url = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&term={term}'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    response = requests.request("GET", url, headers=headers, timeout=10)

    data = json.loads(response.text)

    pmc_list = data['esearchresult']['idlist']
    df = pd.DataFrame({'title': pd.NA, 'pmc_id': pmc_list, 'doi': pd.NA, 'year': pd.NA})

    df.to_csv('pmlist.csv', mode='w', index=False, encoding='utf-8')


def __solve_section(soup: BeautifulSoup, sections: list[Section], title_level: int, ref_soup: BeautifulSoup):
    title = soup.find('title', recursive=False)
    if title:
        sections.append(Section(title.text, title_level, ''))

    section_list = soup.find_all('sec', recursive=False)
    if section_list:
        for sec in section_list:
            sections = __solve_section(sec, sections, title_level + 1, ref_soup)
    else:
        text_list = soup.select('p')
        for text in text_list:
            if text and not text.text == '':
                section = text.text.strip().replace('\n', ' ')
                ref_block = text.find_all('xref', {'ref-type': 'bibr'})
                ref = __solve_ref(ref_soup, ref_block) if ref_block else ''
                # TODO: ref
                sections.append(Section(section, 0, ref))

    return sections


def __solve_ref(ref_soup: BeautifulSoup, ref_list: list[Tag]) -> str:
    numbers = []
    for ref in ref_list:
        contents = ref.text.strip().split(',')
        for content in contents:
            if '–' in content:
                start, end = map(int, content.split('–'))
                numbers.extend(range(start, end + 1))
            else:
                numbers.append(int(content))

    numbers = sorted(list(set(numbers)))

    return ','.join(str(x) for x in numbers)


def download_paper_data(pmc_id: str):
    # logger.info(f'request PMC ID:{pmc_id}')

    url = (f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmc_id}'
           f'&retmode=xml&api_key={config.pubmed_config.API_KEY}')

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    sections: list[Section] = []

    with sessions.Session() as session:
        if config.pubmed_config.USE_PROXY:
            proxies = {
                'http': config.PROXY,
                'https': config.PROXY
            }
            response = session.request("GET", url, headers=headers, proxies=proxies, timeout=10)
        else:
            response = session.request("GET", url, headers=headers, timeout=10)

        soup = BeautifulSoup(response.text, 'xml')

        doi = soup.find('article-id', {'pub-id-type': 'doi'}).text \
            if soup.find('article-id', {'pub-id-type': 'doi'}) \
            else None

        title = soup.find('article-title').text.replace('\n', ' ') \
            if soup.find('article-title') \
            else None

        year = soup.find('pub-date').find('year').text \
            if soup.find('pub-date') \
            else None

        sections.append(Section(title, 1, ''))

        abs_block = soup.find('abstract')
        main_sections = soup.select_one('body')
        ref_sections = soup.select_one('back').select_one('ref-list')

        norm = True
        if abs_block:
            sections.append(Section('Abstract', 2, ''))
            sections = __solve_section(abs_block, sections, 2, ref_sections)
        else:
            logger.warning(f'PMC{pmc_id} has no Abstract')
            norm = False

        if main_sections:
            sections = __solve_section(main_sections, sections, 1, ref_sections)

    return {
        'title': title,
        'year': year,
        'doi': doi,
        'sections': sections,
        'norm': norm
    }
