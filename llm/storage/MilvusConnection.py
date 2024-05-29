from pymilvus import connections, utility, Collection, MilvusClient


class MilvusConnection:
    def __init__(self, uri: str, **kwargs):
        """
        可选参数:
            - user
            - password
            - secure
        """

        connections.connect('default', uri=uri, **kwargs)
        self.client = MilvusClient(uri=uri, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        connections.disconnect('default')

    def __del__(self):
        connections.disconnect('default')

    @staticmethod
    def list_collections() -> list:
        return utility.list_collections()

    @staticmethod
    def has_collection(collection_name) -> bool:
        return utility.has_collection(collection_name)

    @staticmethod
    def get_collection(collection_name) -> Collection:
        return Collection(collection_name)

    def get_describe(self, collection_name) -> dict:
        return self.client.describe_collection(collection_name)

    def get_entity_num(self, collection_name) -> int:
        result = self.client.query(collection_name, '', ['count(*)'])
        return result[0].get('count(*)')

    @staticmethod
    def get_query_segment_info(collection_name):
        return utility.get_query_segment_info(collection_name)

    @staticmethod
    def drop_collection(collection_name) -> None:
        utility.drop_collection(collection_name)
