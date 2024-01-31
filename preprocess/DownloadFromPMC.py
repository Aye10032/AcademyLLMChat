import os
import random
import time

import pandas as pd
from loguru import logger

from Config import config
from utils.PMCUtil import get_pmc_id, download_paper_data, save_to_md


def download_from_pmc(csv_file: str):
    df = pd.read_csv(csv_file, encoding='utf-8', dtype={'title': 'str', 'pmc_id': 'str', 'doi': 'str', 'year': 'str'})

    df_output = df.copy()
    for row in df_output.itertuples():
        if not pd.isna(row.doi):
            logger.info(f'skip {row.doi}')
            continue

        pmcid = row.pmc_id
        data = download_paper_data(pmcid)
        year = data['year']
        filename = data['doi'].replace('/', '@')
        output_path = os.path.join(config.MD_PATH, year, f'{filename}.md')

        if not os.path.exists(os.path.join(config.MD_PATH, year)):
            os.makedirs(os.path.join(config.MD_PATH, year))

        save_to_md(data['sections'], output_path)

        df_output.loc[row.Index, 'title'] = data['title']
        df_output.loc[row.Index, 'doi'] = data['doi']
        df_output.loc[row.Index, 'year'] = year
        df_output.to_csv(csv_file, index=False, encoding='utf-8')
        logger.info(f'finish download {pmcid} ({year})')

        time.sleep(random.uniform(2.0, 5.0))


if __name__ == '__main__':
    # term = 'Raman[Title] AND ("2019/02/01"[PDat] : "2024/01/30"[PDat])&retmode=json&retmax=2000'
    # get_pmc_id(term)

    config.set_collection(1)
    # download_from_pmc('pmlist.csv')
    pmcid = '10807091'
    data = download_paper_data(pmcid)
    year = data['year']
    filename = data['doi'].replace('/', '@')
    output_path = os.path.join(config.MD_PATH, year, f'{filename}.md')

    if not os.path.exists(os.path.join(config.MD_PATH, year)):
        os.makedirs(os.path.join(config.MD_PATH, year))

    print(output_path)
    save_to_md(data['sections'], output_path)
