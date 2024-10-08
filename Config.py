import json
import os
import shutil
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Any

import yaml
from loguru import logger


def get_work_path():
    return os.path.dirname(os.path.abspath(__file__))


@dataclass
class Collection:
    collection_name: str
    language: str
    title: str
    description: str
    index_param: dict[str, Any]
    visitor_visible: bool

    @classmethod
    def from_dict(cls, data: dict[str, any]):
        return cls(**data)


@dataclass
class MilvusConfig:
    data_root: str
    milvus_host: str
    milvus_port: int
    using_remote: bool
    remote_database: dict[str, Any]

    collections: list[Collection] = field(default_factory=list, init=False)
    config_path: str = field(init=False)
    default_collection: int = field(init=False)

    def __post_init__(self):
        self.config_path = os.path.join(get_work_path(), self.data_root, 'collections.json')

        if not os.path.exists(self.config_path):
            logger.error(f'no collection config file find at {self.config_path}')
            exit()

        with open(self.config_path, mode='r', encoding='utf-8') as file:
            json_data = json.load(file)['collections']
            for col in json_data:
                collection = Collection.from_dict(col)
                if collection.visitor_visible:
                    self.collections.append(collection)
            self.default_collection = 0

    @classmethod
    def from_dict(cls, data_root: str, data: dict[str, any]):
        return cls(data_root, **data)

    def set_group_visibility(self, visible: bool) -> None:
        self.collections.clear()

        with open(self.config_path, mode='r', encoding='utf-8') as file:
            json_data = json.load(file)['collections']

        for col in json_data:
            collection = Collection.from_dict(col)
            if visible or collection.visitor_visible:
                self.collections.append(collection)

        self.default_collection = 0

    def get_collection(self) -> Collection:
        collection: Collection = self.collections[self.default_collection]
        return collection

    def get_collection_by_id(self, index: int) -> Collection:
        collection: Collection = self.collections[index]
        return collection

    def get_conn_args(self) -> dict[str, Any]:
        if self.using_remote:
            return {
                'uri': self.remote_database['url'],
                'user': self.remote_database['username'],
                'password': self.remote_database['password'],
                'secure': True,
            }
        else:
            return {
                'uri': f'http://{self.milvus_host}:{self.milvus_port}'
            }

    def add_collection(self, collection: Collection) -> None:
        self.collections.append(collection)
        json.dump(
            {"collections": [asdict(c) for c in self.collections]},
            open(self.config_path, 'w', encoding='utf-8')
        )
        logger.info('update collection index file')

    def remove_collection(self, index: int) -> None:
        del self.collections[index]
        json.dump(
            {"collections": [asdict(c) for c in self.collections]},
            open(self.config_path, 'w', encoding='utf-8')
        )
        logger.info('update collection index file')

    def rename_collection(self, index: int, new_name: str) -> None:
        self.collections[index].title = new_name
        json.dump(
            {"collections": [asdict(c) for c in self.collections]},
            open(self.config_path, 'w', encoding='utf-8')
        )
        logger.info('update collection index file')

    def set_collection_visibility(self, index: int, visible: bool) -> None:
        self.collections[index].visitor_visible = visible
        json.dump(
            {"collections": [asdict(c) for c in self.collections]},
            open(self.config_path, 'w', encoding='utf-8')
        )
        logger.info('update collection index file')


@dataclass
class EmbeddingConfig:
    model: str
    save_local: bool
    fp16: bool
    normalize: bool
    device: str
    local_path: str = field(init=False)

    def __post_init__(self):
        self.local_path = os.path.join(get_work_path(), 'data/model', self.model)
        os.makedirs(self.local_path, exist_ok=True)

    @classmethod
    def from_dict(cls, data: dict[str, any]):
        return cls(**data)


@dataclass
class OpenaiConfig:
    use_proxy: bool
    api_key: str

    @classmethod
    def from_dict(cls, data: dict[str, any]):
        return cls(**data)


@dataclass
class ZhipuConfig:
    api_key: str
    model: str

    @classmethod
    def from_dict(cls, data: dict[str, any]):
        return cls(**data)


@dataclass
class PubmedConfig:
    use_proxy: bool
    api_key: str

    @classmethod
    def from_dict(cls, data: dict[str, any]):
        return cls(**data)


@dataclass
class SerperConfig:
    use_proxy: bool
    api_key: str

    @classmethod
    def from_dict(cls, data: dict[str, any]):
        return cls(**data)


@dataclass
class GrobidConfig:
    grobid_server: str
    service: str
    batch_size: int
    sleep_time: int
    timeout: int
    coordinates: list[str]
    multi_process: int

    @classmethod
    def from_dict(cls, data: dict[str, any]):
        return cls(**data)


class Config:
    def __init__(self):
        yml_path = os.path.join(get_work_path(), 'config.yml')
        if not os.path.exists(yml_path):
            logger.info('config dose not exits')
            shutil.copy(os.path.join(get_work_path(), 'config.example.yml'), yml_path)

        with open(file=yml_path, mode='r', encoding='utf-8') as file:
            self.yml = yaml.load(file, Loader=yaml.FullLoader)

            self.milvus_config: MilvusConfig = MilvusConfig.from_dict(
                self.yml['paper_directory']['data_root'],
                self.yml['retrieve']['milvus']
            )
            self.embedding_config: EmbeddingConfig = EmbeddingConfig.from_dict(self.yml['retrieve']['embedding'])
            self.reranker_config: EmbeddingConfig = EmbeddingConfig.from_dict(self.yml['retrieve']['reranker'])
            self.openai_config: OpenaiConfig = OpenaiConfig.from_dict(self.yml['llm']['openai'])
            self.zhipu_config: ZhipuConfig = ZhipuConfig.from_dict(self.yml['llm']['zhipu'])
            self.pubmed_config: PubmedConfig = PubmedConfig.from_dict(self.yml['tools']['pubmed'])
            self.serper_config: SerperConfig = SerperConfig.from_dict(self.yml['tools']['serper'])
            self.grobid_config: GrobidConfig = GrobidConfig.from_dict(self.yml['tools']['grobid'])

    def set_collection(self, collection: int) -> None:
        if collection >= len(self.milvus_config.collections):
            logger.error('collection index out of range')
            return
        self.milvus_config.default_collection = collection
        collection_name: str = self.milvus_config.get_collection().collection_name

        logger.info(f'set default collection to {collection_name}')

    def get_pdf_path(self, collection_name: str) -> str | bytes:
        data_root = self.yml['paper_directory']['data_root']
        pdf_path = self.yml['paper_directory']['pdf_path']
        pdf_path = os.path.join(get_work_path(), data_root, collection_name, pdf_path)

        os.makedirs(pdf_path, exist_ok=True)
        return pdf_path

    def get_md_path(self, collection_name: str) -> str | bytes:
        data_root = self.yml['paper_directory']['data_root']
        md_path = self.yml['paper_directory']['md_path']
        md_path = os.path.join(get_work_path(), data_root, collection_name, md_path)

        os.makedirs(md_path, exist_ok=True)
        return md_path

    def get_xml_path(self, collection_name: str) -> str | bytes:
        data_root = self.yml['paper_directory']['data_root']
        xml_path = self.yml['paper_directory']['xml_path']
        xml_path = os.path.join(get_work_path(), data_root, collection_name, xml_path)

        os.makedirs(xml_path, exist_ok=True)
        return xml_path

    def get_sqlite_path(self, collection_name: str) -> str | bytes:
        data_root = self.yml['paper_directory']['data_root']
        sqlite_path = self.yml['paper_directory']['sqlite_path']
        sqlite_path = os.path.join(get_work_path(), data_root, collection_name, sqlite_path)

        os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
        return sqlite_path

    def get_reference_path(self):
        data_root = self.yml['paper_directory']['data_root']
        reference_path = os.path.join(get_work_path(), data_root, 'reference.db')

        os.makedirs(os.path.dirname(reference_path), exist_ok=True)
        return reference_path

    def get_user_path(self):
        user_root = self.yml['user_login_config']['user_root']

        return os.path.join(get_work_path(), user_root)

    def get_user_db(self):
        user_root = self.yml['user_login_config']['user_root']
        user_profile = self.yml['user_login_config']['sqlite_filename']

        return os.path.join(get_work_path(), user_root, user_profile)

    def get_proxy(self) -> str:
        proxy_type = self.yml['proxy']['type']
        proxy_host = self.yml['proxy']['host']
        proxy_port = self.yml['proxy']['port']

        return f'{proxy_type}://{proxy_host}:{proxy_port}'
