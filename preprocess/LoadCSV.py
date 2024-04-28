import os
import random
import sys
import time
import pandas as pd

from tqdm import tqdm
from loguru import logger

from Config import Config
from utils.MarkdownPraser import save_to_md
from utils.PMCUtil import download_paper_data, parse_paper_data
from utils.PubmedUtil import get_paper_info
from utils.Decorator import timer


@timer
def download_from_csv(year: int):
    """
    :param year:
    :return:
    """
    df = pd.read_csv('nandesyn_pub.csv', encoding='utf-8',
                     dtype={'Title': 'str', 'PMID': 'str', 'DOI': 'str', 'PMC': 'str'})
    df.sort_values(by=['Year', 'PMID'], inplace=True)
    out_put_df = df.copy()
    df_10 = df[df['Year'] == year]

    length = df_10.shape[0]
    for index, row in tqdm(df_10.iterrows(), total=length, desc=f'search documents in {year}'):
        if pd.isna(row.PMID):
            continue

        if not pd.isna(row.Title):
            continue

        time.sleep(random.uniform(2.0, 5.0))
        pm_data = get_paper_info(row.PMID, config)
        if pm_data['pmc']:
            time.sleep(random.uniform(1.0, 4.0))
            _, download_info = download_paper_data(pm_data['pmc'], config)
            doi = download_info['doi']
            year = download_info['year']
            xml_path = download_info['output_path']

            with open(xml_path, 'r', encoding='utf-8') as f:
                xml_text = f.read()
            flag, xml_data = parse_paper_data(xml_text)

            if not flag:
                out_put_df.at[index, 'Title'] = 'skip'
                out_put_df.at[index, 'DOI'] = doi
                out_put_df.at[index, 'PMC'] = pm_data['pmc']
                out_put_df.to_csv('nandesyn_pub.csv', index=False, encoding='utf-8')
                continue

            output_path = os.path.join(config.get_md_path(), year, doi.replace('/', '@') + '.md')
            os.makedirs(os.path.join(config.get_md_path(), year), exist_ok=True)
            save_to_md(xml_data, output_path)

            out_put_df.at[index, 'Title'] = 'done'
            out_put_df.at[index, 'DOI'] = doi
            out_put_df.at[index, 'PMC'] = pm_data['pmc']
        else:
            title = pm_data['title']
            year = pm_data['year']
            doi = pm_data['doi']

            if doi is None:
                out_put_df.at[index, 'Title'] = title
                out_put_df.to_csv('nandesyn_pub.csv', index=False, encoding='utf-8')
                continue

            out_put_df.at[index, 'Title'] = title
            out_put_df.at[index, 'DOI'] = doi

        out_put_df.to_csv('nandesyn_pub.csv', index=False, encoding='utf-8')


def load_csv(year: int):
    df = pd.read_csv('nandesyn_pmc.csv', encoding='utf-8',
                     dtype={'Title': 'str', 'PMID': 'str', 'DOI': 'str', 'PMC': 'str'})
    out_put_df = df.copy()
    df_10 = df[df['Year'] == year]

    length = df_10.shape[0]
    for index, row in tqdm(df_10.iterrows(), total=length, desc=f'search documents in {year}'):
        doi: str = row.DOI
        xml_path = os.path.join(config.get_xml_path(), str(year), doi.replace('/', '@') + '.xml')

        if not os.path.exists(xml_path):
            out_put_df['Title'] = 'not exist'
            out_put_df.to_csv('nandesyn_pmc.csv', index=False, encoding='utf-8')
            continue

        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_text = f.read()
        flag, xml_data = parse_paper_data(xml_text)

        if not flag:
            out_put_df['Title'] = 'skip'
            out_put_df.to_csv('nandesyn_pmc.csv', index=False, encoding='utf-8')
            continue

        output_path = os.path.join(config.get_md_path(), str(year), doi.replace('/', '@') + '.md')
        os.makedirs(os.path.join(config.get_md_path(), str(year)), exist_ok=True)
        save_to_md(xml_data, output_path)

        out_put_df['Title'] = 'done'
        out_put_df.to_csv('nandesyn_pmc.csv', index=False, encoding='utf-8')


def init_csv():
    df = pd.read_csv('nandesyn_pub_bk.csv', encoding='utf-8',
                     dtype={'Title': 'str', 'PMID': 'str', 'DOI': 'str', 'PMC': 'str'})
    df.sort_values(by=['Year', 'PMID'], inplace=True)
    out_put_df = df.copy().drop('Journal', axis=1)
    out_put_df['Title'] = pd.NA
    out_put_df.to_csv('nandesyn_pub.csv', index=False, encoding='utf-8')


def get_pmc_list():
    df = pd.read_csv('nandesyn_pub.csv', encoding='utf-8',
                     dtype={'Title': 'str', 'PMID': 'str', 'DOI': 'str', 'PMC': 'str'})
    pmc_df = df[df['PMC'].notnull()].copy()
    pmc_df.sort_values(by=['Year', 'DOI'], inplace=True)
    pmc_df['Title'] = pd.NA
    pmc_df.to_csv('nandesyn_pmc.csv', index=False, encoding='utf-8')


if __name__ == '__main__':
    logger.remove()
    handler_id = logger.add(sys.stderr, level="INFO")
    logger.add('log/load_csv.log')

    config = Config()
    config.set_collection(0)

    # init_csv()

    get_pmc_list()

    for i in range(2010, 2024):
        # download_from_csv(i)
        load_csv(i)
