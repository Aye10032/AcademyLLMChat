import random
from typing import Dict, Tuple, List

import pandas as pd
import requests

from bs4 import BeautifulSoup
from loguru import logger

from Config import Config
from utils.Decorator import retry
from utils.MarkdownPraser import PaperInfo, Section, PaperType, Paper, Reference


@retry(delay=random.uniform(2.0, 5.0))
def get_paper_info(pmid: str, config: Config = None, silent: bool = True) -> Paper:
    """

    :param pmid:
    :param config:
    :param silent:
    :return:
    """

    if config is None:
        config = Config()

    if not silent:
        logger.info(f'request PMID:{pmid}')
    url = (f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}'
           f'&retmode=xml&api_key={config.pubmed_config.api_key}')

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    if config.pubmed_config.use_proxy:
        proxies = {
            'http': config.get_proxy(),
            'https': config.get_proxy()
        }
        response = requests.request("GET", url, headers=headers, proxies=proxies, timeout=10)
    else:
        response = requests.request("GET", url, headers=headers, timeout=10)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'xml')

        title = soup.find('Article').find('ArticleTitle').text if soup.find('Article') else None
        year = (soup.find('Article')
                .find('JournalIssue')
                .find('PubDate').find('Year').text)

        author = ''
        if author_block := soup.find('Author'):
            last_name = author_block.find('LastName').text if author_block.find('LastName') else ''
            initials = author_block.find('Initials').text if author_block.find('Initials') else ''
            author = f'{last_name}, {initials}'

        abstract = soup.find('AbstractText').text if soup.find('AbstractText') else None

        keyword_list = soup.find('KeywordList')
        if keyword_list:
            keywords = [keyword.text for keyword in keyword_list.find_all('Keyword')]
        else:
            keywords = []

        doi_block = soup.find('ArticleIdList').find('ArticleId', {'IdType': 'doi'})
        if doi_block:
            doi = doi_block.text
        else:
            doi = None
            logger.warning('DOI not found')

        pmc_block = soup.find('ArticleIdList').find('ArticleId', {'IdType': 'pmc'})
        if pmc_block:
            pmc = pmc_block.text.replace('PMC', '')
        else:
            pmc = None

        ref_block = soup.find('ReferenceList')

        ref_list = []
        if ref_block:
            for ref in ref_block.find_all('Reference'):
                if id_list := ref.find('ArticleIdList'):
                    if ref_pmc_block := id_list.find('ArticleId', {'IdType': 'pmc'}):
                        ref_pmc = ref_pmc_block.text
                    else:
                        ref_pmc = pd.NA

                    if ref_pm_block := id_list.find('ArticleId', {'IdType': 'pubmed'}):
                        ref_pm = ref_pm_block.text
                    else:
                        ref_pm = pd.NA

                    if ref_doi_block := id_list.find('ArticleId', {'IdType': 'doi'}):
                        ref_doi = ref_doi_block.text
                    else:
                        ref_doi = pd.NA

                    ref_list.append({'doi': ref_doi, 'pubmed': ref_pm, 'pmc': ref_pmc})

        if pmc is None:
            paper_info = PaperInfo(author, int(year), PaperType.GROBID_PAPER, ''.join(keywords), True, doi)
        else:
            paper_info = PaperInfo(author, int(year), PaperType.PMC_PAPER, ''.join(keywords), True, doi)

        section_list = [
            Section(title, 1),
            Section('Abstract', 2),
            Section(abstract, 0)
        ]
        return Paper(paper_info, section_list, Reference(doi, ref_list))
    else:
        # 请求失败时抛出异常
        raise Exception('下载请求失败')
