import json
import random
import time
from typing import List

import requests
import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger
from requests import sessions

from Config import config
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


def solve_section(soup: BeautifulSoup, sections: List, title_level: int):
    title = soup.find('title', recursive=False)
    if title:
        sections.append({
            'text': title.text,
            'level': title_level
        })

    section_list = soup.find_all('sec', recursive=False)
    if section_list:
        for sec in section_list:
            sections = solve_section(sec, sections, title_level + 1)
    else:
        text_list = soup.select('p')
        for text in text_list:
            if text and not text.text == '':
                section = text.text.strip().replace('\n', ' ')
                sections.append({'text': section, 'level': 0})

    return sections


def download_paper_data(pmc_id: str):
    # logger.info(f'request PMC ID:{pmc_id}')

    url = (f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmc_id}'
           f'&retmode=xml&api_key={config.pubmed_config.API_KEY}')

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    sections = []
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

        sections.append({'text': title, 'level': 1})

        abs_block = soup.find('abstract')

        norm = True
        if abs_block:
            sections.append({'text': 'Abstract', 'level': 2})
            sections = solve_section(abs_block, sections, 2)
        else:
            logger.warning(f'PMC{pmc_id} has no Abstract')
            norm = False

        main_sections = soup.select_one('body')

        if main_sections:
            sections = solve_section(main_sections, sections, 1)

    return {
        'title': title,
        'year': year,
        'doi': doi,
        'sections': sections,
        'norm': norm
    }

