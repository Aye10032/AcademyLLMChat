from streamlit.connections import BaseConnection

from pymilvus import connections, utility, Collection, MilvusClient


class MilvusConnection:
    def __init__(self, uri: str, **kwargs):
        """
        可选参数:
            - user
            - password
            - secure
        """
        self.uri = uri
        self.kwargs = kwargs

        connections.connect('default', uri=self.uri, **self.kwargs)
        self.client = None

    def __enter__(self):
        self.client = MilvusClient(self.uri, **self.kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        connections.disconnect('default')
        connections.disconnect()

    def __del__(self):
        connections.disconnect()

    def list_collections(self) -> list:
        return utility.list_collections()

    def has_collection(self, collection_name) -> bool:
        return utility.has_collection(collection_name)

    def get_collection(self, collection_name) -> Collection:
        return Collection(collection_name)

    def get_describe(self, collection_name) -> dict:
        return self.client.describe_collection(collection_name)

    def get_entity_num(self, collection_name) -> int:
        return self.client.num_entities(collection_name=collection_name)

    def get_query_segment_info(self, collection_name):
        return utility.get_query_segment_info(collection_name)

    def drop_collection(self, collection_name) -> None:
        utility.drop_collection(collection_name)
