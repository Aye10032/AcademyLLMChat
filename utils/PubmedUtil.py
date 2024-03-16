import random
from typing import Dict

import pandas as pd
import requests

from bs4 import BeautifulSoup
from loguru import logger

from Config import Config
from utils.DecoratorUtil import retry


# TODO 处理传入设置
@retry(delay=random.uniform(2.0, 5.0))
def get_paper_info(pmid: str, config: Config = None) -> Dict:
    """
    获取指定PMID的论文信息。

    通过.ncbi.nlm.nih.gov的eutils服务获取论文的元数据，包括标题、年份、作者、摘要、关键词、DOI、PMC标识和参考文献列表。

    :param pmid: 论文的PubMed标识符。
    :param config: 包含API密钥和代理配置的配置对象。如果未提供，则使用默认配置。
    :return: 包含论文信息的字典。
    """

    # 如果未提供配置对象，则创建默认配置
    if config is None:
        config = Config()

    # 记录请求信息
    logger.info(f'request PMID:{pmid}')
    # 构建请求URL，包含API密钥
    url = (f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}'
           f'&retmode=xml&api_key={config.pubmed_config.API_KEY}')

    # 设置用户代理头
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    # 根据配置决定是否使用代理
    if config.pubmed_config.USE_PROXY:
        proxies = {
            'http': config.get_proxy(),
            'https': config.get_proxy()
        }
        # 发起GET请求
        response = requests.request("GET", url, headers=headers, proxies=proxies, timeout=10)
    else:
        response = requests.request("GET", url, headers=headers, timeout=10)

    # 检查响应状态
    if response.status_code == 200:
        # 使用BeautifulSoup解析XML响应
        soup = BeautifulSoup(response.text, 'xml')

        # 提取论文标题和出版年份
        title = soup.find('Article').find('ArticleTitle').text if soup.find('Article') else None
        year = (soup.find('Article')
                .find('JournalIssue')
                .find('PubDate').find('Year').text)

        # 提取第一作者信息
        author = ''
        if author_block := soup.find('Author'):
            last_name = author_block.find('LastName').text if author_block.find('LastName') else ''
            initials = author_block.find('Initials').text if author_block.find('Initials') else ''
            author = f'{last_name}, {initials}'

        # 提取摘要
        abstract = soup.find('AbstractText').text if soup.find('AbstractText') else None

        # 提取关键词
        keyword_list = soup.find('KeywordList')
        if keyword_list:
            keywords = [keyword.text for keyword in keyword_list.find_all('Keyword')]
        else:
            keywords = []

        # 提取DOI和PMC标识
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

        # 提取参考文献信息
        ref_block = soup.find('ReferenceList')

        ref_list = []
        if ref_block:
            for ref in ref_block.find_all('Reference'):
                # 对每篇参考文献提取PMID、PMC和DOI信息
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

        # 返回论文信息
        return {
            'title': title,
            'author': author,
            'year': year,
            'abstract': abstract,
            'keywords': keywords,
            'doi': doi,
            'pmc': pmc,
            'ref_list': pd.DataFrame(ref_list)
        }
    else:
        # 请求失败时抛出异常
        raise Exception('下载请求失败')
