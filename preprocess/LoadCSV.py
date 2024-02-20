import os
import random
import sys
import time
import pandas as pd

from tqdm import tqdm
from loguru import logger

from Config import config
from utils.FileUtil import Section, save_to_md
from utils.PMCUtil import download_paper_data, parse_paper_data
from utils.PubmedUtil import get_paper_info
from utils.TimeUtil import timer


@timer
def load_csv(year: int):
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
        time.sleep(random.uniform(2.0, 5.0))
        if pd.isna(row.PMID):
            logger.warning(f'PMID {row.Title} is None')
            continue

        if not pd.isna(row.Title):
            continue

        pm_data = get_paper_info(row.PMID)
        if pm_data['pmc']:
            time.sleep(random.uniform(1.0, 4.0))
            download_info = download_paper_data(pm_data['pmc'])
            doi = download_info['doi']
            year = download_info['year']
            xml_path = download_info['output_path']

            with open(xml_path, 'r', encoding='utf-8') as f:
                xml_text = f.read()
            xml_data = parse_paper_data(xml_text, year, doi)
            title = xml_data['title']
            author = xml_data['author']

            if not xml_data['norm']:
                out_put_df.at[index, 'Title'] = title
                out_put_df.at[index, 'DOI'] = doi
                out_put_df.at[index, 'PMC'] = pm_data['pmc']
                out_put_df.to_csv('nandesyn_pub.csv', index=False, encoding='utf-8')
                continue

            output_path = os.path.join(config.get_md_path(), year, doi.replace('/', '@') + '.md')
            os.makedirs(os.path.join(config.get_md_path(), year), exist_ok=True)
            save_to_md(xml_data['sections'], output_path, ref=True, year=year, doi=doi, author=author)

            out_put_df.at[index, 'Title'] = title
            out_put_df.at[index, 'DOI'] = doi
            out_put_df.at[index, 'PMC'] = pm_data['pmc']
        else:
            title = pm_data['title']
            author = pm_data['author']
            year = pm_data['year']
            abstract = pm_data['abstract']
            doi = pm_data['doi']

            if doi is None:
                out_put_df.at[index, 'Title'] = title
                out_put_df.to_csv('nandesyn_pub.csv', index=False, encoding='utf-8')
                continue

            if abstract is None:
                out_put_df.at[index, 'Title'] = title
                out_put_df.at[index, 'DOI'] = doi
                out_put_df.to_csv('nandesyn_pub.csv', index=False, encoding='utf-8')
                continue

            sections = [Section(title, 1), Section('Abstract', 2), Section(abstract, 0)]

            output_path = os.path.join(config.get_md_path(), year, doi.replace('/', '@') + '.md')
            os.makedirs(os.path.join(config.get_md_path(), year), exist_ok=True)
            save_to_md(sections, output_path, ref=False, year=year, doi=doi, author=author)

            out_put_df.at[index, 'Title'] = title
            out_put_df.at[index, 'DOI'] = doi

        out_put_df.to_csv('nandesyn_pub.csv', index=False, encoding='utf-8')


if __name__ == '__main__':
    logger.remove()
    handler_id = logger.add(sys.stderr, level="INFO")
    logger.add('log/load_csv.log')
    config.set_collection(0)

    # df = pd.read_csv('nandesyn_pub_bk.csv', encoding='utf-8',
    #                  dtype={'Title': 'str', 'PMID': 'str', 'DOI': 'str', 'PMC': 'str'})
    # df.sort_values(by=['Year', 'PMID'], inplace=True)
    # out_put_df = df.copy().drop('Journal', axis=1)
    # out_put_df['Title'] = pd.NA
    # out_put_df.to_csv('nandesyn_pub.csv', index=False, encoding='utf-8')
    for i in range(2010, 2024):
        load_csv(i)
