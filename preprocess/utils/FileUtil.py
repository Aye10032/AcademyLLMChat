import os
import re
import string

from loguru import logger


def format_filename(filename):
    """
    Format filename to remove invalid characters
    :param filename:
    :return:
    """
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c for c in filename if c in valid_chars)

    filename = filename.replace('\\', ' or ').replace('|', '')

    filename = re.sub(r'_+', '_', filename)

    filename = filename.strip('._')

    return filename


def save_to_md(_dict: dict, output_path: str):
    """
    将解析结果保存为markdown文件
    :param _dict: 解析结果
    :param output_path: 输出MD文件的根目录
    :return:
    """
    title = _dict['title']
    abstract = _dict['abstract']
    sections = _dict['sections']

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(f'# {title}\n\n')

        f.write(f'## Abstract\n\n{abstract}\n\n')

        father_title = ''
        father_level = ''
        check_sub = False
        has_sub = False
        for section in sections:
            section_title = section['title']
            title_level = section['title_level']
            text = section['text']

            if section_title.lower() in ['introduction', 'results', 'discussion']:
                f.write(f'## {section_title}\n\n')

                if text:
                    for paragraph in text:
                        f.write(paragraph + '\n\n')
                    check_sub = False
                else:
                    if title_level and title_level.endswith('.'):
                        father_title = section_title
                        father_level = title_level
                        check_sub = True
                        has_sub = False
                    else:
                        check_sub = False
                        logger.warning(f'Section {section_title} has no title level')
                continue

            if check_sub:
                if title_level and title_level.startswith(father_level):
                    f.write(f'### {section_title}\n\n')
                    for paragraph in text:
                        f.write(paragraph + '\n\n')
                    has_sub = True

                if not has_sub:
                    logger.warning(f'Section {father_title} has no sub section')
