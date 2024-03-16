import time
import random

import pandas as pd
from tqdm import tqdm

from Config import Config
from utils.PMCUtil import download_paper_data


def __download(csv_path: str):
    df = pd.read_csv(csv_path, encoding='utf-8',
                     dtype={'Title': 'str', 'Year': 'str'})
    df_output = df.copy()

    length = df_output.shape[0]
    for index, row in tqdm(df_output.iterrows(), total=length, desc='download xml'):
        if not pd.isna(row.Year):
            continue

        pmcid = row.PMCID
        _, data = download_paper_data(pmcid)

        df_output.at[index, 'Year'] = data['year']
        df_output.to_csv(csv_path, index=False, encoding='utf-8')
        time.sleep(random.uniform(2.0, 5.0))


def __init_csv(file_path: str, new_name: str):
    df = pd.read_csv(file_path, encoding='utf-8')
    out_put_df = (df.copy()
                  .drop(['Authors', 'Citation', 'Create Date', 'NIHMS ID'], axis=1)
                  .dropna(subset=['PMCID'], axis=0))

    out_put_df['Publication Year'] = pd.NA
    out_put_df.to_csv(new_name, index=False, encoding='utf-8')


if __name__ == '__main__':
    config = Config()
    config.set_collection(0)
    # __init_csv('csv-NatureJour-set.csv', 'csv-NatureJour.csv')
    __download('csv-NatureJour.csv')
