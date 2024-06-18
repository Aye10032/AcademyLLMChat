import json
import random
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests import sessions
from tqdm import tqdm

from utils.Decorator import retry


@retry(delay=random.uniform(2.0, 5.0))
def get_ids(term: str, max_size: int) -> None:
    url = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gene&term={term}&retmode=json&retmax={max_size}'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    response = requests.request("GET", url, headers=headers, timeout=10)

    if response.status_code == 200:
        data = json.loads(response.text)

        id_list = data['esearchresult']['idlist']
        df = pd.DataFrame({'gene': pd.NA, 'geneid': id_list, 'summary': pd.NA, 'location': pd.NA})

        df.to_csv('gene.csv', mode='w', index=False, encoding='utf-8')
    else:
        raise Exception('下载请求失败')


@retry(delay=random.uniform(2.0, 5.0))
def get_info(gene_id: str):
    url = (f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=gene&id={gene_id}&retmode=xml')

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    with sessions.Session() as session:
        response = session.request("GET", url, headers=headers, timeout=10)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'xml')

        gene = soup.find('Gene-ref_locus').text \
            if soup.find('Gene-ref_locus') \
            else pd.NA

        summary = soup.find('Entrezgene_summary').text \
            if soup.find('Entrezgene_summary') \
            else pd.NA

        start = int(soup.find('Seq-interval_from').text) + 1 \
            if soup.find('Seq-interval_from') \
            else pd.NA

        end = int(soup.find('Seq-interval_to').text) + 1 \
            if soup.find('Seq-interval_to') \
            else pd.NA

        location = f'from {start} to {end} bases on chromosome 7 of homo sapience'

        return gene, summary, location
    else:
        raise Exception('下载请求失败')


def update_gene_info():
    df = pd.read_csv('gene.csv', encoding='utf-8', dtype={'gene': 'str', 'geneid': 'str', 'summary': 'str', 'location': 'str'})

    df_output = df.copy()
    length = df_output.shape[0]
    for index, row in tqdm(df_output.iterrows(), total=length, desc='adding gene'):
        if not pd.isna(row.gene):
            continue

        gene_id = row.geneid
        gene, summary, location = get_info(gene_id)
        df_output.at[index, 'gene'] = gene
        df_output.at[index, 'summary'] = summary
        df_output.at[index, 'location'] = location
        df_output.to_csv('gene.csv', index=False, encoding='utf-8')
        time.sleep(random.uniform(2.0, 5.0))


def main() -> None:
    # term = '"homo sapience"[Organism] AND 7[Chromosome] AND alive[prop]'
    # get_ids(term, 2000)

    update_gene_info()


if __name__ == '__main__':
    main()
