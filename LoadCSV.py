import os
import time
import pandas as pd

from utils.DownloadUtil import parse_paper_info
from loguru import logger

from utils.TimeUtil import timer


@timer
def load_csv(year: int, output_path: str):
    df = pd.read_csv('nandesyn_pub.csv', encoding='utf-8', dtype={'PMID': 'str'})
    df.sort_values(by='Year', inplace=True)

    df_10 = df[df['Year'] == year]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for row in df_10.itertuples():
        article = parse_paper_info(row.PMID)
        abstract = article['abstract']

        if abstract is None:
            logger.warning(f'PMID {row.PMID} has no abstract')
            continue

        if article['doi'] is None:
            file_name = row.PMID + '.md'
        else:
            file_name = article['doi'].replace('/', '@') + '.md'
        with open(f'{output_path}{file_name}', 'w', encoding='utf-8') as f:
            f.write(f'# {row.Title}\n\n')
            f.write(f'## Abstract\n\n{abstract}\n\n')


if __name__ == '__main__':
    logger.add('log/load_csv.log')
    for i in range(2013, 2023):
        logger.info(f'Loading paper in {i}...')
        load_csv(i, f'./output/{i}/')
