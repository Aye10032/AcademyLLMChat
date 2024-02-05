# PDF解析部分
import json
import os
import shutil
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


class Collection:
    def __init__(self, collection_name: str, language: str, description: str):
        self.NAME = collection_name
        self.LANGUAGE = language
        self.DESCRIPTION = description

    @classmethod
    def from_dict(cls, data: Dict[str, any]):
        return cls(**data)


class MilvusConfig:
    def __init__(self, milvus_host: str, milvus_port: int, config_path: str, en_model: str, zh_model: str,
                 using_remote: bool, remote_database: Dict):
        self.MILVUS_HOST = milvus_host
        self.MILVUS_PORT = milvus_port
        self.CONFIG_PATH = os.path.join(get_work_path(), config_path)
        self.COLLECTIONS: list[Collection] = []
        self.EN_MODEL = en_model
        self.ZH_MODEL = zh_model
        self.USING_REMOTE = using_remote
        self.REMOTE_DATABASE = remote_database

        if not os.path.exists(self.CONFIG_PATH):
            logger.info('config dose not exits')
            default = {
                "collections": [
                    {
                        "collection_name": "default",
                        "language": "en",
                        "description": "示例知识库"
                    }
                ]
            }
            with open(file=self.CONFIG_PATH, mode='w', encoding='utf-8') as file:
                json.dump(default, file)

        with open(file=self.CONFIG_PATH, mode='r', encoding='utf-8') as file:
            json_data = json.load(file)['collections']
            for col in json_data:
                self.COLLECTIONS.append(Collection.from_dict(col))
            self.DEFAULT_COLLECTION = 0

    @classmethod
    def from_dict(cls, data: Dict[str, any]):
        return cls(**data)

    def get_collection(self):
        collection: Collection = self.COLLECTIONS[self.DEFAULT_COLLECTION]
        return collection


class OpenaiConfig:
    def __init__(self, use_proxy: bool, api_key: str):
        self.USE_PROXY = use_proxy
        self.API_KEY = api_key

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

            self.DATA_ROOT = self.yml['data_root']
            self.PDF_PATH = self.yml['pdf_path']
            self.MD_PATH = self.yml['md_path']
            self.XML_PATH = self.yml['xml_path']

            _proxy_type = self.yml['proxy']['type']
            _proxy_host = self.yml['proxy']['host']
            _proxy_port = self.yml['proxy']['port']
            self.PROXY = f'{_proxy_type}://{_proxy_host}:{_proxy_port}'

            self.pubmed_config: PubmedConfig = PubmedConfig.from_dict(self.yml['pubmed'])
            self.grobid_config: GrobidConfig = GrobidConfig.from_dict(self.yml['grobid'])
            self.milvus_config: MilvusConfig = MilvusConfig.from_dict(self.yml['milvus'])
            self.openai_config: OpenaiConfig = OpenaiConfig.from_dict(self.yml['openai'])

    def set_collection(self, collection: int):
        if collection >= len(self.milvus_config.COLLECTIONS):
            logger.error('collection index out of range')
            return
        self.milvus_config.DEFAULT_COLLECTION = collection
        collection_name: str = self.milvus_config.get_collection().NAME

        logger.info(f'set default collection to {collection_name}')

    def get_pdf_path(self):
        collection_name: str = self.milvus_config.get_collection().NAME
        return os.path.join(get_work_path(), self.DATA_ROOT, collection_name, self.PDF_PATH)

    def get_md_path(self):
        collection_name: str = self.milvus_config.get_collection().NAME
        return os.path.join(get_work_path(), self.DATA_ROOT, collection_name, self.MD_PATH)

    def get_xml_path(self):
        collection_name: str = self.milvus_config.get_collection().NAME
        return os.path.join(get_work_path(), self.DATA_ROOT, collection_name, self.XML_PATH)


config = Config()
