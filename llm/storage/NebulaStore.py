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
        创建图空间

        :param space_name: 在NebulaGraph实例中唯一标识一个图空间，仅支持 1~4 字节的 UTF-8 编码字符，包括英文字母（区分大小写）、数字、中文等
        :param partition_num: 指定图空间的分片数量。建议设置为集群中硬盘数量的 20 倍（HDD 硬盘建议为 2 倍）
        :param replica_factor: 指定每个分片的副本数量。建议在生产环境中设置为 3，在测试环境中设置为 1。副本数量必须是奇数
        :param vid_type: 指定点 ID 的数据类型
        :param comment: 图空间的描述
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

    def use_space(self, space_name: str) -> ResultSet:
        """
        切换到指定图空间

        :param space_name: 图空间唯一标识
        :return:

        USE <graph_space_name>;
        """
        result = self.client.execute(f'USE {space_name};')
        return result

    def clear_space(self, space_name: str) -> ResultSet:
        """
        清空图空间中的点和边，但不会删除图空间本身以及其中的 Schema 信息

        :param space_name: 图空间唯一标识
        :return:

        CLEAR SPACE [IF EXISTS] <graph_space_name>
        """
        result = self.client.execute(f'CLEAR SPACE IF EXISTS {space_name};')
        return result

    def drop_space(self, space_name: str) -> ResultSet:
        """
        删除指定图空间

        :param space_name: 图空间唯一标识
        :return:

        DROP SPACE [IF EXISTS] <graph_space_name>
        """
        result = self.client.execute(f'DROP SPACE IF EXISTS {space_name};')

        logger.info('Deleting graph database, please wait...')
        time.sleep(6)
        logger.info('done')
        return result


def main() -> None:
    with NebulaGraphStore() as store:
        # result = store.create_space('reference', vid_type=VidType.STRING255)
        # result = store.drop_space('reference')
        store.use_space('reference')


if __name__ == '__main__':
    main()
