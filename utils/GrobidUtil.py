import re

from grobid_client.grobid_client import GrobidClient
from bs4 import BeautifulSoup, NavigableString

from utils.FileUtil import format_filename


def parse_pdf(pdf_path: str, xml_path: str):
    client = GrobidClient(config_path="../config/grobid.json")
    client.process("processFulltextDocument", pdf_path, output=xml_path, n=10)


def parse_xml(xml_path: str):
    with open(xml_path, 'r', encoding='utf-8') as f:
        xml_data = f.read()
        soup = BeautifulSoup(xml_data, 'xml')

        # Title
        title = soup.find('titleStmt').find('title', {'type': 'main'})
        title = format_filename(title.text.strip()) if title is not None else ''

        # Authors
        authors = []
        for author in soup.find('sourceDesc').find_all('persName'):
            first_name = author.find('forename', {'type': 'first'})
            first_name = first_name.text.strip() if first_name is not None else ''
            middle_name = author.find('forename', {'type': 'middle'})
            middle_name = middle_name.text.strip() if middle_name is not None else ''
            last_name = author.find('surname')
            last_name = last_name.text.strip() if last_name is not None else ''
            if middle_name != '':
                authors.append(f'{first_name} {middle_name} {last_name}')
            else:
                authors.append(f'{first_name} {last_name}')

        # date
        pub_date = soup.find('publicationStmt')
        year = pub_date.find('date')
        year = year.attrs.get('when') if year is not None else ''

        # Abstract
        abstract = ''
        abstract_list = soup.find('profileDesc').select('abstract p')
        for p in abstract_list:
            abstract += p.text.strip() + ' '

        # keywords
        keywords = split_words(soup.find('profileDesc').select('keywords')[0].get_text())

        # sections
        sections = []
        for section in soup.find('text').find_all('div', {'xmlns': 'http://www.tei-c.org/ns/1.0'}):
            title = section.find('head').text.strip()
            title_level = section.find('head').attrs.get('n')

            text = []
            for p in section.find_all('p'):
                text.append(replace_multiple_spaces(p.text.strip()))

            sections.append({'title': title, 'title_level': title_level, 'text': text})

    return {'title': title, 'authors': authors, 'year': year, 'abstract': abstract, 'keywords': keywords,
            'sections': sections}


def split_words(string):
    pattern = re.compile(r'\([^()]*\)|\S+')
    words = pattern.findall(string)
    clean_words = [word.replace('(', '').replace(')', '') for word in words]

    return clean_words


def replace_multiple_spaces(text):
    pattern = re.compile(r'\s+')
    clean_text = pattern.sub(' ', text)

    return clean_text


# parse_pdf('../../../DATA/documents/11/', '../output')
data = parse_xml('../output/10.1016@j.febslet.2011.05.015.grobid.tei.xml')
