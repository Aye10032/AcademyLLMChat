import re
import string
from typing import List


def format_filename(filename: str) -> str:
    """
    格式化文件名，移除无效字符
    :param filename: 需要格式化的文件名
    :return: 格式化后的文件名
    """
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c for c in filename if c in valid_chars)
    filename = filename.replace('\\', ' or ').replace('|', '')
    filename = re.sub(r'_+', '_', filename)
    filename = filename.strip('._')

    return filename


def split_words(text: str) -> List[str]:
    """
    分割文本中的单词。

    此函数使用正则表达式来识别并分割文本中的单词，包括那些被括号包围的单词。分割后的单词会移除括号。

    :param text: 要分割的文本字符串。
    :return: 包含分割出的单词的列表，其中括号已被移除。
    """

    pattern = re.compile(r'\([^()]*\)|\S+')
    words = pattern.findall(text)
    clean_words = [word.replace('(', '').replace(')', '') for word in words]

    return clean_words


def replace_multiple_spaces(text: str) -> str:
    """
    替换文本中的多个连续空格为单个空格。

    :param text: 待处理的字符串，其中可能包含多个连续空格。
    :return: 清理后的字符串，其中多个连续空格被替换为单个空格。
    """

    pattern = re.compile(r'\s+')
    clean_text = pattern.sub(' ', text)

    return clean_text


def is_en(text: str) -> bool:
    """
    检查文本是否只包含英文和数字。

    :param text: 需要检查的文本字符串。
    :return: 如果文本只包含英文和数字，返回True；否则返回False。
    """

    if re.match(r'[a-zA-Z0-9]+', text):
        return True
    else:
        return False
