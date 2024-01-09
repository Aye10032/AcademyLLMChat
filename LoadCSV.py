import os
import random
import time
import pandas as pd

from utils.DownloadUtil import parse_paper_info
from loguru import logger

from utils.TimeUtil import timer


@timer
def load_csv(year: int, output_path: str):
    """
    加载CSV文件并按照给定年份筛选数据，将结果保存到指定路径的文件中

    参数:
    year (int): 指定的年份
    output_path (str): 指定的保存路径

    返回:
    无
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

        article = parse_paper_info(row.PMID, True)
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
            f.write(f'# {row.Title}\n\n')
            f.write(f'## Abstract\n\n{abstract}\n\n')

        out_put_df.to_csv('nandesyn_pub.csv', index=False, encoding='utf-8')
        time.sleep(random.uniform(2.0, 3.0))


def get_doi(md_path: str):
    md_list = []
    for root, dirs, files in os.walk(md_path):
        for file in files:
            if file.endswith('.md'):
                md_list.append(file)
    print(md_list)


if __name__ == '__main__':
    # logger.add('log/load_csv.log')

    i = 2023

    logger.info(f'Loading paper in {i}...')
    load_csv(i, f'./output/md/{i}/')
    get_doi('output/md/')

    d_f = pd.read_csv('nandesyn_pub.csv', encoding='utf-8', dtype={'PMID': 'str', 'DOI': 'str'})
    d_f.sort_values(by=['Year', 'PMID'], inplace=True)

    df_merge = d_f[d_f['Year'] == i]
