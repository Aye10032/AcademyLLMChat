from streamlit.connections import BaseConnection

from pymilvus import connections, utility, Collection, MilvusClient


class MilvusConnection(BaseConnection[connections]):
    def _connect(self, **kwargs) -> connections:
        if 'uri' in kwargs:
            uri = kwargs.pop('uri')
        elif 'uri' in self._secrets:
            uri = self._secrets['uri']
        else:
            raise Exception('no milvus uri found')

        """
        可选参数:
            - user
            - password
            - secure
        """

        self.client = MilvusClient(uri, **kwargs)

        return connections.connect('default', uri=uri, **kwargs)

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

    def disconnect(self):
        connections.disconnect('default')
