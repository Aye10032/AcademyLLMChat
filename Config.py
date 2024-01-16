# PDF解析部分
import os
import shutil
import sys
from typing import Dict

import yaml
from loguru import logger


def get_work_path():
    return os.path.dirname(os.path.abspath(__file__))


class MilvusConfig:
    def __init__(self, milvus_host: str, milvus_port: int, collection_name: str):
        self.MILVUS_HOST = milvus_host
        self.MILVUS_PORT = milvus_port
        self.COLLECTION_NAME = collection_name

    @classmethod
    def from_dict(cls, data: Dict[str, any]):
        return cls(**data)


class Config:
    def __init__(self):
        if not os.path.exists('config.yml'):
            logger.info('config dose not exits')
            shutil.copy('config.example.yml', 'config.yml')

        with open(file='config.yml', mode='r', encoding='utf-8') as file:
            self.yml = yaml.load(file, Loader=yaml.FullLoader)

            self.PDF_PARSER = self.yml['pdf_parser']

            self.PDF_ROOT = os.path.join(get_work_path(), self.yml['pdf_root'])
            self.MD_OUTPUT = os.path.join(get_work_path(), self.yml['md_output'])
            self.XML_OUTPUT = os.path.join(get_work_path(), self.yml['xml_output'])

            self.milvus_config: MilvusConfig = MilvusConfig.from_dict(self.yml['milvus'])


config = Config()
