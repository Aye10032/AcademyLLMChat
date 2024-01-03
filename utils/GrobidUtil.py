from grobid_client.grobid_client import GrobidClient
from bs4 import BeautifulSoup


def parse_pdf(pdf_path: str, xml_path: str):
    client = GrobidClient(config_path="../config/grobid.json")
    client.process("processFulltextDocument", pdf_path, output=xml_path, n=10)


def parse_xml(xml_path: str):
    soup = BeautifulSoup(xml_path, 'xml')
    for p in soup.find_all('p'):
        print(p.text)


parse_pdf('../../../DATA/documents/11/', '../output')
# soup = BeautifulSoup(xml_data, 'xml')
