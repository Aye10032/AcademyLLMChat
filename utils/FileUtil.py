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


def split_words(string):
    pattern = re.compile(r'\([^()]*\)|\S+')
    words = pattern.findall(string)
    clean_words = [word.replace('(', '').replace(')', '') for word in words]

    return clean_words


def replace_multiple_spaces(text):
    pattern = re.compile(r'\s+')
    clean_text = pattern.sub(' ', text)

    return clean_text
