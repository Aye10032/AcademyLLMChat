import requests

from Config import config
from bs4 import BeautifulSoup
from loguru import logger


def get_paper_info(pmid: str):
    """
    :param pmid: pubmed id
    :return:
        'title': str,
        'author': str,
        'year': str,
        'abstract': str,
        'keywords': list[str,
        'doi': str
    """
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

    return {
        'title': title,
        'author': author,
        'year': year,
        'abstract': abstract,
        'keywords': keywords,
        'doi': doi
    }


if __name__ == '__main__':
    data = get_paper_info('26846317')
    print(data)
