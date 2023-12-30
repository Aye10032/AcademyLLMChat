import re
import string


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
    authors = _dict['authors']
    publication_date = _dict['pub_date']
    doi = _dict['doi']
    abstract = _dict['abstract']
    sections = _dict['sections']

    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(f'# {title}\n\n')

        f.write(f'## Abstract\n\n{abstract}\n\n')

        for section in sections:
            section_title = section['heading']
            section_text = section['text'].replace('\n', '\n\n')
            f.write(f'## {section_title}\n\n')
            f.write(f'{section_text}\n\n')
