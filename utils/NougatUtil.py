import os

from loguru import logger


def parse_pdf(pdf_path: str, output_path: str):
    """
    批量解析根目录下的PDF文件，并按照原目录结构保存为MD文件
    :param pdf_path: pdf文件的根目录
    :param output_path: 输出MD文件的根目录
    :return:
    """
    # 读取根目录下的所有PDF文件
    pdf_paths = []
    for root, dirs, files in os.walk(pdf_path):
        for file in files:
            if file.endswith('.pdf'):
                pdf_paths.append(os.path.join(root, file))

    # 解析PDF文件
    count = len(pdf_paths)
    for index, _path in enumerate(pdf_paths):
        # 获取PDF文件相对于根目录的路径
        rel_pdf_path = os.path.relpath(_path, pdf_path)
        # 获取输出MD文件的路径
        md_base, _ = os.path.split(os.path.join(output_path, rel_pdf_path))

        # 创建输出MD文件的目录
        os.makedirs(os.path.dirname(md_base), exist_ok=True)

        # 解析PDF文件
        logger.info(f'Processing file {_path} [{index + 1}/{count}]')
        __parse_pdf_to_md(_path, md_base)

    logger.info(f'save {count} files to {output_path}')


def __parse_pdf_to_md(pdf_path: str, md_path: str):
    """
    解析PDF文件，返回解析结果
    :param pdf_path: pdf文件路径
    :param md_path: md文件输出路径
    :return:
    """
    os.system(f'nougat {pdf_path} --no-skip -o {md_path}')
