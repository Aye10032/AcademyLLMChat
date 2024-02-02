import os
import random
import time

import pandas as pd
from loguru import logger
from tqdm import tqdm

from Config import config
from utils.PMCUtil import get_pmc_id, download_paper_data, save_to_md

logger.add('log/pmc.log')


def download_from_pmc(csv_file: str):
    df = pd.read_csv(csv_file, encoding='utf-8', dtype={'title': 'str', 'pmc_id': 'str', 'doi': 'str', 'year': 'str'})

    df_output = df.copy()
    length = df_output.shape[0]
    for index, row in tqdm(df_output.iterrows(), total=length):
        if not pd.isna(row.year):
            continue

        pmcid = row.pmc_id
        data = download_paper_data(pmcid)
        year = data['year']
        filename = data['doi'].replace('/', '@') if data['doi'] else f'PMC{pmcid}'
        if not data['norm']:
            filename += '_(no abstract)'
        output_path = os.path.join(config.MD_PATH, year, f'{filename}.md')

        if not os.path.exists(os.path.join(config.MD_PATH, year)):
            os.makedirs(os.path.join(config.MD_PATH, year))

        save_to_md(data['sections'], output_path)

        df_output.at[index, 'title'] = data['title']
        df_output.at[index, 'doi'] = data['doi']
        df_output.at[index, 'year'] = year
        df_output.to_csv(csv_file, index=False, encoding='utf-8')
        time.sleep(random.uniform(2.0, 5.0))


if __name__ == '__main__':
    # term = 'Raman[Title] AND ("2019/02/01"[PDat] : "2024/01/30"[PDat])&retmode=json&retmax=2000'
    # get_pmc_id(term)

    config.set_collection(1)
    download_from_pmc('pmlist.csv')
