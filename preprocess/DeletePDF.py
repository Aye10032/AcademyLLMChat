import os

import pandas as pd

from Config import config

df = pd.read_csv('nandesyn_pub.csv', encoding='utf-8',
                 dtype={'Title': 'str', 'PMID': 'str', 'DOI': 'str', 'PMC': 'str'})
df.sort_values(by=['Year', 'PMID'], inplace=True)

pmc_file = df.dropna(subset=['PMC']).loc[(df['Year'] != 2022) & (df['Year'] != 2023)]

count = 0
for row in pmc_file.itertuples():
    doi: str = row.DOI.replace('/', '@')
    filename = f'{doi}.pdf'
    year = row.Year

    file_path = os.path.join(config.get_pdf_path(), str(year), filename)
    if not os.path.exists(file_path):
        print(file_path)
    else:
        os.remove(file_path)
        count += 1

print(count)
