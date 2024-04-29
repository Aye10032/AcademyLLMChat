import os
import random
import sys
import time

import pandas as pd
from loguru import logger
from tqdm import tqdm

from Config import Config
from utils.MarkdownPraser import save_to_md
from utils.PMCUtil import download_paper_data, parse_paper_data, get_pmc_id

logger.remove()
handler_id = logger.add(sys.stderr, level="INFO")
logger.add('log/pmc.log')


def download_from_pmc(csv_file: str):
    df = pd.read_csv(csv_file, encoding='utf-8', dtype={'title': 'str', 'pmc_id': 'str', 'doi': 'str', 'year': 'str'})

    df_output = df.copy()
    length = df_output.shape[0]
    for index, row in tqdm(df_output.iterrows(), total=length, desc='download xml'):
        if not pd.isna(row.year):
            continue

        pmcid = row.pmc_id
        _, data = download_paper_data(pmcid, config)

        df_output.at[index, 'doi'] = data['doi']
        df_output.at[index, 'year'] = data['year']
        df_output.to_csv(csv_file, index=False, encoding='utf-8')
        time.sleep(random.uniform(2.0, 5.0))


def solve_xml(csv_file: str):
    df = pd.read_csv(csv_file, encoding='utf-8', dtype={'title': 'str', 'pmc_id': 'str', 'doi': 'str', 'year': 'str'})

    df_output = df.copy()
    length = df_output.shape[0]
    for index, row in tqdm(df_output.iterrows(), total=length, desc='adding documents'):
        if not pd.isna(row.title):
            continue

        year: str = row.year
        doi: str = row.doi
        with open(os.path.join(config.get_xml_path(), year, doi.replace('/', '@') + '.xml'), 'r',
                  encoding='utf-8') as f:
            xml_text = f.read()

        try:
            flag, data = parse_paper_data(xml_text)
        except Exception as e:
            logger.error(f'{year} {doi} {e}')
            break

        if not flag:
            df_output.at[index, 'title'] = 'skip'
            df_output.to_csv(csv_file, index=False, encoding='utf-8')
            continue

        output_path = os.path.join(config.get_md_path(), year, doi.replace('/', '@') + '.md')
        os.makedirs(os.path.join(config.get_md_path(), year), exist_ok=True)
        save_to_md(data, output_path)

        df_output.at[index, 'title'] = 'done'
        df_output.to_csv(csv_file, index=False, encoding='utf-8')


def reset_csv(path: str) -> None:
    df = pd.read_csv(path, encoding='utf-8', dtype={'title': 'str', 'pmc_id': 'str', 'doi': 'str', 'year': 'str'})
    df['title'] = pd.NA

    df.to_csv(path, index=False, encoding='utf-8')


if __name__ == '__main__':
    config = Config()
    # term = 'Raman[Title] AND ("2019/04/26"[PDat] : "2024/04/23"[PDat])&retmode=json&retmax=2000'
    # get_pmc_id(term)

    config.set_collection(1)
    # reset_csv('pmlist.csv')
    # download_from_pmc('pmlist.csv')
    solve_xml('pmlist.csv')
