import json
import os
import shutil
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Dict, Tuple, Any, LiteralString

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
    use_proxy: bool
    api_key: str

    @classmethod
    def from_dict(cls, data: Dict[str, any]):
        return cls(**data)


@dataclass
class GrobidConfig:
    config_path: str
    service: str
    multi_process: int

    @classmethod
    def from_dict(cls, data: Dict[str, any]):
        return cls(**data)

    def get_config_path(self):
        return os.path.join(get_work_path(), self.config_path)


@dataclass
class Collection:
    collection_name: str
    language: str
    title: str
    description: str
    index_param: str

    @classmethod
    def from_dict(cls, data: Dict[str, any]):
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

        if os.path.exists(self.config_path):
            logger.error(f'no collection config file find at {self.config_path}')
            exit()

        with open(file=self.config_path, mode='r', encoding='utf-8') as file:
            json_data = json.load(file)['collections']
            for col in json_data:
                self.collections.append(Collection.from_dict(col))
            self.default_collection = 0

    @classmethod
    def from_dict(cls, data_root: str, data: Dict[str, any]):
        return cls(data_root, **data)

    def get_collection(self):
        collection: Collection = self.collections[self.default_collection]
        return collection

    def get_conn_args(self):
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

    def add_collection(self, collection: Collection):
        self.collections.append(collection)
        json.dump(
            {"collections": [asdict(c) for c in self.collections]},
            open(self.config_path, 'w', encoding='utf-8')
        )
        logger.info('update collection index file')

    def remove_collection(self, index: int):
        del self.collections[index]
        json.dump({"collections": [asdict(c) for c in self.collections]},
                  open(self.config_path, 'w', encoding='utf-8'))
        logger.info('update collection index file')

    def rename_collection(self, index: int, new_name: str):
        self.collections[index].title = new_name
        json.dump({"collections": [asdict(c) for c in self.collections]},
                  open(self.config_path, 'w', encoding='utf-8'))
        logger.info('update collection index file')


@dataclass
class EmbeddingConfig:
    model: str
    save_local: bool
    fp16: bool
    normalize_embeddings: bool
    local_path: str = field(init=False)

    def __post_init__(self):
        self.local_path = os.path.join(get_work_path(), 'model')
        os.makedirs(self.local_path, exist_ok=True)

    @classmethod
    def from_dict(cls, data: Dict[str, any]):
        return cls(**data)


@dataclass
class OpenaiConfig:
    use_proxy: bool
    api_key: str

    @classmethod
    def from_dict(cls, data: Dict[str, any]):
        return cls(**data)


@dataclass
class ClaudeConfig:
    use_proxy: bool
    model: str
    api_key: str

    @classmethod
    def from_dict(cls, data: Dict[str, any]):
        return cls(**data)


@dataclass
class QianfanConfig:
    api_key: str
    secret_key: str
    model: str

    @classmethod
    def from_dict(cls, data: Dict[str, any]):
        return cls(**data)


@dataclass
class MoonshotConfig:
    api_key: str
    model: str

    @classmethod
    def from_dict(cls, data: Dict[str, any]):
        return cls(**data)


@dataclass
class ZhipuConfig:
    api_key: str
    model: str

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

            self.data_root = self.yml['paper_directory']['data_root']
            self.pdf_path = self.yml['paper_directory']['pdf_path']
            self.md_path = self.yml['paper_directory']['md_path']
            self.xml_path = self.yml['paper_directory']['xml_path']
            self.sqlite_path = self.yml['paper_directory']['sqlite_path']

            self.proxy_type = self.yml['proxy']['type']
            self.proxy_host = self.yml['proxy']['host']
            self.proxy_port = self.yml['proxy']['port']

            self.admin_token = self.yml['auth']['admin_token']
            self.owner_token = self.yml['auth']['owner_token']

            self.pubmed_config: PubmedConfig = PubmedConfig.from_dict(self.yml['pubmed'])
            self.grobid_config: GrobidConfig = GrobidConfig.from_dict(self.yml['grobid'])
            self.milvus_config: MilvusConfig = MilvusConfig.from_dict(self.data_root, self.yml['milvus'])
            self.embedding_config: EmbeddingConfig = EmbeddingConfig.from_dict(self.yml['embedding'])
            self.openai_config: OpenaiConfig = OpenaiConfig.from_dict(self.yml['llm']['openai'])
            self.claude_config: ClaudeConfig = ClaudeConfig.from_dict(self.yml['llm']['claude3'])
            self.qianfan_config: QianfanConfig = QianfanConfig.from_dict(self.yml['llm']['qianfan'])
            self.moonshot_config: MoonshotConfig = MoonshotConfig.from_dict(self.yml['llm']['moonshot'])
            self.zhipu_config: ZhipuConfig = ZhipuConfig.from_dict(self.yml['llm']['zhipu'])

    def set_collection(self, collection: int) -> None:
        if collection >= len(self.milvus_config.collections):
            logger.error('collection index out of range')
            return
        self.milvus_config.DEFAULT_COLLECTION = collection
        collection_name: str = self.milvus_config.get_collection().collection_name

        logger.info(f'set default collection to {collection_name}')

    def get_pdf_path(self):
        collection_name: str = self.milvus_config.get_collection().collection_name
        pdf_path = os.path.join(get_work_path(), self.data_root, collection_name, self.pdf_path)

        os.makedirs(pdf_path, exist_ok=True)
        return pdf_path

    def get_md_path(self):
        collection_name: str = self.milvus_config.get_collection().collection_name
        md_path = os.path.join(get_work_path(), self.data_root, collection_name, self.md_path)
        return md_path

    def get_xml_path(self):
        collection_name: str = self.milvus_config.get_collection().collection_name
        xml_path = os.path.join(get_work_path(), self.data_root, collection_name, self.xml_path)

        os.makedirs(xml_path, exist_ok=True)
        return xml_path

    def get_sqlite_path(self):
        collection_name: str = self.milvus_config.get_collection().collection_name
        sqlite_path = os.path.join(get_work_path(), self.data_root, collection_name, self.sqlite_path)

        os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
        return sqlite_path

    def get_reference_path(self):
        reference_path = os.path.join(get_work_path(), self.data_root, 'reference.db')

        os.makedirs(os.path.dirname(reference_path), exist_ok=True)
        return reference_path

    def get_proxy(self) -> str:
        return f'{self.proxy_type}://{self.proxy_host}:{self.proxy_port}'
