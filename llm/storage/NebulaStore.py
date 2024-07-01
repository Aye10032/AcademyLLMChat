import time
from enum import StrEnum

from loguru import logger
from nebula3.Config import Config as NConfig
from nebula3.data.ResultSet import ResultSet
from nebula3.gclient.net import ConnectionPool


class VidType(StrEnum):
    STRING255 = 'FIXED_STRING(255)'
    INT64 = 'INT[64]'


class NebulaGraphStore:
    def __init__(self, address: str = '127.0.0.1', port: int = 9669, username: str = 'root', password: str = 'nebula'):
        self.address = address
        self.port = port
        self.username = username
        self.password = password

        self.__init_connection()

    def __init_connection(self):
        n_config = NConfig()
        n_config.max_connection_pool_size = 2

        self.connection_pool = ConnectionPool()
        assert self.connection_pool.init([(self.address, self.port)], n_config)

        self.client = self.connection_pool.get_session(self.username, self.password)
        assert self.client is not None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.release()
        self.connection_pool.close()

    def create_space(
            self,
            space_name: str,
            partition_num: int = 20,
            replica_factor: int = 1,
            comment: str = None,
            *,
            vid_type: str,
    ) -> ResultSet:
        """
        :param space_name:
        :param partition_num:
        :param replica_factor:
        :param vid_type:
        :param comment:
        :return:

        CREATE SPACE [IF NOT EXISTS] <graph_space_name> (
        [partition_num = <partition_number>,]
        [replica_factor = <replica_number>,]
        vid_type = {FIXED_STRING(<N>) | INT[64]}
        )
        [COMMENT = '<comment>'];
        """

        assert vid_type == 'INT[64]' or vid_type.startswith('FIXED_STRING')

        if comment is not None:
            result = self.client.execute(
                f'CREATE SPACE IF NOT EXISTS {space_name} ('
                f'partition_num={partition_num}, replica_factor={replica_factor}, vid_type={vid_type}'
                f') comment="{comment}"'
            )
        else:
            result = self.client.execute(
                f'CREATE SPACE IF NOT EXISTS {space_name} ('
                f'partition_num={partition_num}, replica_factor={replica_factor}, vid_type={vid_type}'
                f')'
            )

        logger.info('Creating graph database, please wait...')
        time.sleep(6)
        logger.info('done')
        return result

    def drop_space(self, space_name: str):
        result = self.client.execute(f'DROP SPACE IF EXISTS {space_name};')

        logger.info('Deleting graph database, please wait...')
        time.sleep(6)
        logger.info('done')
        return result


def main() -> None:
    with NebulaGraphStore() as store:
        # store.create_space('reference', vid_type=VidType.STRING255)
        result = store.drop_space('reference')

    print(result)


if __name__ == '__main__':
    main()
