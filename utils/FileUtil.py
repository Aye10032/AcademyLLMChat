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


def split_words(text):
    pattern = re.compile(r'\([^()]*\)|\S+')
    words = pattern.findall(text)
    clean_words = [word.replace('(', '').replace(')', '') for word in words]

    return clean_words


def replace_multiple_spaces(text):
    pattern = re.compile(r'\s+')
    clean_text = pattern.sub(' ', text)

    return clean_text


def save_to_md(sections: list, output_path, append: bool = False):
    """
    Save sections to markdown file
    :param sections: markdown结构化段落
                    - text: 文本
                    - level: 标题等级(1-4)，0代表文本
    :param output_path: markdown文件输出路径
    :param append: 是否追加写入
    :return:
    """
    if append:
        open_type = 'a'
    else:
        open_type = 'w'

    with open(output_path, open_type, encoding='utf-8') as f:
        for sec in sections:
            text = sec['text']
            level = sec['level']
            if level == 0:
                f.write(f'{text}\n\n')
            elif level == 1:
                f.write(f'# {text}\n\n')
            elif level == 2:
                f.write(f'## {text}\n\n')
            elif level == 3:
                f.write(f'### {text}\n\n')
            else:
                f.write(f'#### {text}\n\n')
