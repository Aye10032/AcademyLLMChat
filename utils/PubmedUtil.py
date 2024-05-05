import random
from enum import IntEnum
from typing import Dict, Any

import pandas as pd
import requests

from bs4 import BeautifulSoup
from loguru import logger

from Config import Config
from utils.Decorator import retry
from utils.MarkdownPraser import PaperInfo, Section, PaperType, Paper, Reference


class SearchType(IntEnum):
    TITLE: 0
    DOI: 1
    PM: 2


@retry(delay=random.uniform(2.0, 5.0))
def get_paper_info(pmid: str, config: Config = None, silent: bool = True) -> Paper:
    """
    根据PubMed ID (PMID) 获取论文信息。

    :param pmid: 需要查询的PubMed ID。
    :param config: 包含API密钥和代理配置的配置对象。如果未提供，则使用默认配置。
    :param silent: 是否在查询时记录日志信息。默认为True，即不记录。
    :return: 返回一个Paper对象，包含论文的各种信息，如标题、作者、年份、摘要等。
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


@retry(delay=random.uniform(2.0, 5.0))
def get_info_by_term(term: str, search_type: int, config: Config = None, silent: bool = True) -> Dict[str, Any]:
    """
    通过标题、doi号等补全参考文献信息

    :param term: 查询关键字，可以是标题、DOI号、PMID
    :param search_type: 查询类别
    :param config:
    :param silent:
    :return: 包含索引信息的字典
    """
    if config is None:
        config = Config()

    if search_type == SearchType.TITLE:
        if not silent:
            logger.info(f'search title: {term}')

        url = (f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={term}'
               f'&retmode=xml&api_key={config.pubmed_config.api_key}')

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
    elif search_type == SearchType.DOI:
        if not silent:
            logger.info(f'search title: {term}')

        url = (f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={term}[doi]'
               f'&retmode=xml&api_key={config.pubmed_config.api_key}')

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
    else:
        return __get_info(term)

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

        count: int = soup.find('Count').text if soup.find('Count') else 0

        if count > 0:
            pmid = soup.find('IdList').find_all('Id')[0].text
            return __get_info(pmid)


@retry(delay=random.uniform(2.0, 5.0))
def __get_info(pmid: str, config: Config = None) -> Dict[str, Any]:
    """
    通过 PMID获取参考文献的相关信息

    :param pmid:
    :param config:
    :return:
    """
    if config is None:
        config = Config()

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

        doi_block = soup.find('ArticleIdList').find('ArticleId', {'IdType': 'doi'})
        if doi_block:
            doi = doi_block.text
        else:
            doi = ''
            logger.warning('DOI not found')

        pmc_block = soup.find('ArticleIdList').find('ArticleId', {'IdType': 'pmc'})
        if pmc_block:
            pmc = pmc_block.text.replace('PMC', '')
        else:
            pmc = ''

        return {
            'title': title,
            'pmid': pmid,
            'pmc': pmc,
            'doi': doi
        }
