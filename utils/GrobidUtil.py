import os

import scipdf
from loguru import logger


class GrobidOfflineException(Exception):
    pass


def parse_pdf_to_md(pdf_path: str, output_path: str, grobid_url: str):
    """
    批量解析根目录下的PDF文件，并按照原目录结构保存为MD文件
    :param pdf_path: pdf文件的根目录
    :param output_path: 输出MD文件的根目录
    :param grobid_url: grobid api链接
    :return:
    """
    # 读取根目录下的所有PDF文件
    pdf_paths = []
    for root, dirs, files in os.walk(pdf_path):
        for file in files:
            if file.endswith('.pdf'):
                pdf_paths.append(os.path.join(root, file))

    # 解析PDF文件
    for _path in pdf_paths:
        # 获取PDF文件相对于根目录的路径
        rel_pdf_path = os.path.relpath(_path, pdf_path)
        # 获取输出MD文件的路径
        rel_md_path = rel_pdf_path.replace('.pdf', '.md')
        md_path = os.path.join(output_path, rel_md_path)
        # 创建输出MD文件的目录
        os.makedirs(os.path.dirname(md_path), exist_ok=True)

        # 解析PDF文件
        pdf_dict = parse_pdf(_path, grobid_url)
        save_to_md(pdf_dict, md_path)


def parse_pdf(pdf_path: str, grobid_url: str):
    """
    解析PDF文件，返回解析结果
    :param pdf_path: pdf文件的根目录
    :param grobid_url: grobid api链接
    :return:
    """
    if grobid_url.endswith('/'):
        grobid_url = grobid_url.rstrip('/')

    try:
        article_dict = scipdf.parse_pdf_to_dict(pdf_path, grobid_url=grobid_url)
    except GrobidOfflineException:
        raise GrobidOfflineException('GROBID服务不可用，请修改config中的GROBID_URL，可修改成本地GROBID服务。')
    except Exception as e:
        raise RuntimeError('解析PDF失败，请检查PDF是否损坏。')

    return article_dict


def save_to_md(_dict: dict, output_path: str):
    """
    将解析结果保存为markdown文件
    :param _dict: 解析结果
    :param output_path: 输出MD文件的根目录
    :return:
    """
    title = _dict['title']
    authors = _dict['authors']
    publication_date = _dict['pub_date']
    doi = _dict['doi']
    abstract = _dict['abstract']
    sections = _dict['sections']
    logger.info(f'title: {title}, doi: {doi}')

    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(f'# {title}\n\n')

        f.write(f'## Abstract\n\n{abstract}\n\n')

        for section in sections:
            section_title = section['heading']
            section_text = section['text'].replace('\n', '\n\n')
            f.write(f'## {section_title}\n\n')
            f.write(f'{section_text}\n\n')


parse_pdf_to_md('../../../documents/', 'output', 'https://aye10032-grobid.hf.space')
