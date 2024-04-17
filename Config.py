# PDF解析部分
import json
import os
import shutil
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Tuple

import yaml
from loguru import logger


class UserRole(IntEnum):
    VISITOR = 0
    ADMIN = 1
    OWNER = 2


def get_work_path():
    return os.path.dirname(os.path.abspath(__file__))


@dataclass
class PubmedConfig:
    USE_PROXY: bool = field(metadata={'key': 'use_proxy'})
    API_KEY: str = field(metadata={'key': 'api_key'})

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


@dataclass
class Collection:
    NAME: str = field(metadata={'key': 'collection_name'})
    LANGUAGE: str = field(metadata={'key': 'language'})
    TITLE: str = field(metadata={'key': 'title'})
    DESCRIPTION: str = field(metadata={'key': 'description'})
    INDEX_PARAM: str = field(metadata={'key': 'index_param'})

    @classmethod
    def from_dict(cls, data: Dict[str, any]):
        return cls(**data)

    def to_dict(self):
        return {
            "collection_name": self.NAME,
            "language": self.LANGUAGE,
            "title": self.TITLE,
            "description": self.DESCRIPTION,
            "index_param": self.INDEX_PARAM
        }


class MilvusConfig:
    def __init__(self, data_root: str, milvus_host: str, milvus_port: int, en_model: str, zh_model: str,
                 using_remote: bool, remote_database: Dict):
        self.MILVUS_HOST = milvus_host
        self.MILVUS_PORT = milvus_port
        self.CONFIG_PATH = os.path.join(get_work_path(), data_root, 'collections.json')
        self.COLLECTIONS: list[Collection] = []
        self.EN_MODEL = en_model
        self.ZH_MODEL = zh_model
        self.USING_REMOTE = using_remote
        self.REMOTE_DATABASE = remote_database

        if not os.path.exists(self.CONFIG_PATH):
            logger.error('no collection config file find')
            exit()

        with open(file=self.CONFIG_PATH, mode='r', encoding='utf-8') as file:
            json_data = json.load(file)['collections']
            for col in json_data:
                self.COLLECTIONS.append(Collection.from_dict(col))
            self.DEFAULT_COLLECTION = 0

    @classmethod
    def from_dict(cls, data_root: str, data: Dict[str, any]):
        return cls(data_root, **data)

    def get_collection(self):
        collection: Collection = self.COLLECTIONS[self.DEFAULT_COLLECTION]
        return collection

    def get_conn_args(self):
        if self.USING_REMOTE:
            return {
                'uri': self.REMOTE_DATABASE['url'],
                'user': self.REMOTE_DATABASE['username'],
                'password': self.REMOTE_DATABASE['password'],
                'secure': True,
            }
        else:
            return {
                'uri': f'http://{self.MILVUS_HOST}:{self.MILVUS_PORT}'
            }

    def add_collection(self, collection: Collection):
        self.COLLECTIONS.append(collection)
        json.dump({"collections": [c.to_dict() for c in self.COLLECTIONS]},
                  open(self.CONFIG_PATH, 'w', encoding='utf-8'))
        logger.info('update collection index file')

    def remove_collection(self, index: int):
        del self.COLLECTIONS[index]
        json.dump({"collections": [c.to_dict() for c in self.COLLECTIONS]},
                  open(self.CONFIG_PATH, 'w', encoding='utf-8'))
        logger.info('update collection index file')

    def rename_collection(self, index: int, new_name: str):
        self.COLLECTIONS[index].TITLE = new_name
        json.dump({"collections": [c.to_dict() for c in self.COLLECTIONS]},
                  open(self.CONFIG_PATH, 'w', encoding='utf-8'))
        logger.info('update collection index file')


@dataclass
class OpenaiConfig:
    USE_PROXY: bool = field(metadata={'key': 'use_proxy'})
    API_KEY: str = field(metadata={'key': 'api_key'})

    @classmethod
    def from_dict(cls, data: Dict[str, any]):
        return cls(**data)


@dataclass
class ClaudeConfig:
    USE_PROXY: bool = field(metadata={'key': 'use_proxy'})
    MODEL: str = field(metadata={'key': 'model'})
    API_KEY: str = field(metadata={'key': 'api_key'})

    @classmethod
    def from_dict(cls, data: Dict[str, any]):
        return cls(**data)


@dataclass
class QianfanConfig:
    API_KEY: str = field(metadata={'key': 'api_key'})
    SECRET_KEY: str = field(metadata={'key': 'secret_key'})
    MODEL: str = field(metadata={'key': 'model'})

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
            self.SQLITE_PATH = self.yml['sqlite_path']

            self.PROXY_TYPE = self.yml['proxy']['type']
            self.PROXY_HOST = self.yml['proxy']['host']
            self.PROXY_PORT = self.yml['proxy']['port']

            self.ADMIN_TOKEN = self.yml['auth']['admin_token']
            self.OWNER_TOKEN = self.yml['auth']['owner_token']

            self.pubmed_config: PubmedConfig = PubmedConfig.from_dict(self.yml['pubmed'])
            self.grobid_config: GrobidConfig = GrobidConfig.from_dict(self.yml['grobid'])
            self.milvus_config: MilvusConfig = MilvusConfig.from_dict(self.DATA_ROOT, self.yml['milvus'])
            self.openai_config: OpenaiConfig = OpenaiConfig.from_dict(self.yml['llm']['openai'])
            self.claude_config: ClaudeConfig = ClaudeConfig.from_dict(self.yml['llm']['claude3'])
            self.claude_config: ClaudeConfig = ClaudeConfig.from_dict(self.yml['llm']['qianfan'])

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

    def get_sqlite_path(self):
        collection_name: str = self.milvus_config.get_collection().NAME
        sqlite_path = os.path.join(get_work_path(), self.DATA_ROOT, collection_name, self.SQLITE_PATH)
        os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
        return sqlite_path

    def get_proxy(self):
        return f'{self.PROXY_TYPE}://{self.PROXY_HOST}:{self.PROXY_PORT}'
