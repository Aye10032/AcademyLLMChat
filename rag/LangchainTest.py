from langchain.text_splitter import MarkdownHeaderTextSplitter

md_path = '../preprocess/output/markdown/11/10.1016@j.febslet.2011.05.015.grobid.tei.md'


md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[('#', 'Title'), ('##', 'SubTitle'), ('###', 'Title3')])
with open(md_path, 'r') as f:
    md_text = f.read()
md_docs = md_splitter.split_text(md_text)
