# PDF解析部分
import os
import shutil
import sys
from typing import Dict

import yaml
from loguru import logger


def get_work_path():
    return os.path.dirname(os.path.abspath(__file__))


class PubmedConfig:
    def __init__(self, use_proxy: bool, api_key: str):
        self.USE_PROXY = use_proxy
        self.API_KEY = api_key

    @classmethod
    def from_dict(cls, data: Dict[str, any]):
        return cls(**data)


class GrobidConfig:
    def __init__(self, config_path: str, service: str, multi_process: int):
        self.CONFIG_PATH = os.path.join(get_work_path(), config_path)
        self.SERVICE = service
        self.MULTI_PROCESS = multi_process

    @classmethod
    def from_dict(cls, data: Dict[str, any]):
        return cls(**data)


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
        yml_path = os.path.join(get_work_path(), 'config.yml')
        if not os.path.exists(yml_path):
            logger.info('config dose not exits')
            shutil.copy(os.path.join(get_work_path(), 'config.example.yml'), yml_path)

        with open(file=yml_path, mode='r', encoding='utf-8') as file:
            self.yml = yaml.load(file, Loader=yaml.FullLoader)

            self.PDF_PARSER = self.yml['pdf_parser']

            self.PDF_ROOT = os.path.join(get_work_path(), self.yml['pdf_root'])
            self.MD_OUTPUT = os.path.join(get_work_path(), self.yml['md_output'])
            self.XML_OUTPUT = os.path.join(get_work_path(), self.yml['xml_output'])

            _proxy_type = self.yml['proxy']['type']
            _proxy_host = self.yml['proxy']['host']
            _proxy_port = self.yml['proxy']['port']
            self.PROXY = f'{_proxy_type}://{_proxy_host}:{_proxy_port}'

            self.pubmed_config: PubmedConfig = PubmedConfig.from_dict(self.yml['pubmed'])

            self.grobid_config: GrobidConfig = GrobidConfig.from_dict(self.yml['grobid'])

            self.milvus_config: MilvusConfig = MilvusConfig.from_dict(self.yml['milvus'])


config = Config()
