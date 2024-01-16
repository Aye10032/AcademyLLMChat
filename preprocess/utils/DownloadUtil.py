import requests

from Config import API_KEY
from bs4 import BeautifulSoup
from loguru import logger


def parse_paper_info(pmid: str, use_proxy: bool = False):
    logger.info(f'request PMID:{pmid}')
    url = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&rettype=abstract&retmode=xml&api_key={API_KEY}'
    proxies = {
        'http': 'http://127.0.0.1:11451',
        'https': 'http://127.0.0.1:11451'
    }
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    if use_proxy:
        response = requests.request("GET", url, headers=headers, proxies=proxies, timeout=10)
    else:
        response = requests.request("GET", url, headers=headers, timeout=10)

    # print(response.text)
    soup = BeautifulSoup(response.text, 'xml')

    # year = (soup.find('Article')
    #         .find('JournalIssue')
    #         .find('PubDate').find('Year').text)
    abstract = soup.find('AbstractText').text if soup.find('AbstractText') else None
    doi_block = soup.find('ArticleIdList').find('ArticleId', {'IdType': 'doi'})
    if doi_block:
        doi = doi_block.text
    else:
        doi = None
        logger.warning('DOI not found')

    return {
        # 'year': year,
        'abstract': abstract,
        'doi': doi
    }


if __name__ == '__main__':
    parse_paper_info('26846317')
