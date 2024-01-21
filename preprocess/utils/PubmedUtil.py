import requests

from Config import config
from bs4 import BeautifulSoup
from loguru import logger


def get_paper_info(pmid: str):
    logger.info(f'request PMID:{pmid}')
    url = (f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}'
           f'&rettype=abstract&retmode=xml&api_key={config.pubmed_config.API_KEY}')

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    if config.pubmed_config.USE_PROXY:
        proxies = {
            'http': config.PROXY,
            'https': config.PROXY
        }
        response = requests.request("GET", url, headers=headers, proxies=proxies, timeout=10)
    else:
        response = requests.request("GET", url, headers=headers, timeout=10)

    soup = BeautifulSoup(response.text, 'xml')

    title = soup.find('Article').find('Journal').find('Title').text
    year = (soup.find('Article')
            .find('JournalIssue')
            .find('PubDate').find('Year').text)
    abstract = soup.find('AbstractText').text if soup.find('AbstractText') else None

    keyword_list = soup.find('KeywordList').find_all('Keyword')
    if keyword_list:
        keywords = [keyword.text for keyword in keyword_list]
    else:
        keywords = []

    doi_block = soup.find('ArticleIdList').find('ArticleId', {'IdType': 'doi'})
    if doi_block:
        doi = doi_block.text
    else:
        doi = None
        logger.warning('DOI not found')

    return {
        'title': title,
        'year': year,
        'abstract': abstract,
        'keywords': keywords,
        'doi': doi
    }


if __name__ == '__main__':
    data = get_paper_info('26846317')
    print(data)

