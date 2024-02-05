import os
import random
import time
import pandas as pd

from loguru import logger

from Config import config
from utils.PubmedUtil import get_paper_info
from utils.TimeUtil import timer


@timer
def load_csv(year: int, output_path: str):
    """
    :param year:
    :param output_path:
    :return:
    """
    df = pd.read_csv('nandesyn_pub.csv', encoding='utf-8', dtype={'PMID': 'str', 'DOI': 'str'})
    df.sort_values(by=['Year', 'PMID'], inplace=True)

    out_put_df = df.copy()

    df_10 = df[df['Year'] == year]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for row in df_10.itertuples():
        if pd.isna(row.PMID):
            logger.warning(f'PMID {row.Title} is None')
            continue

        if not pd.isna(row.DOI):
            logger.info(f'skip {row.PMID}')
            continue

        article = get_paper_info(row.PMID)
        title = article['title']
        abstract = article['abstract']

        if abstract is None:
            logger.warning(f'PMID {row.PMID} has no abstract')
            continue

        if article['doi'] is None:
            file_name = row.PMID + '.md'
        else:
            file_name = article['doi'].replace('/', '@') + '.md'
            out_put_df.loc[row.Index, 'DOI'] = article['doi']
        with open(f'{output_path}{file_name}', 'w', encoding='utf-8') as f:
            f.write(f'# {title}\n\n')
            f.write(f'## Abstract\n\n{abstract}\n\n')

        out_put_df.to_csv('nandesyn_pub.csv', index=False, encoding='utf-8')
        time.sleep(random.uniform(2.0, 5.0))


if __name__ == '__main__':
    logger.add('log/load_csv.log')

    for i in range(2010, 2024):

        logger.info(f'Loading paper in {i}...')
        load_csv(i, f'{config.MD_PATH}/{i}/')
        # get_doi('output/md/')

        # d_f = pd.read_csv('nandesyn_pub.csv', encoding='utf-8', dtype={'PMID': 'str', 'DOI': 'str'})
        # d_f.sort_values(by=['Year', 'Title'], inplace=True)
        #
        # df_merge = d_f[d_f['Year'] == i].copy().drop('Journal', axis=1)
        # df_merge['DOI'] = df_merge['DOI'].str.replace('/', '@')
