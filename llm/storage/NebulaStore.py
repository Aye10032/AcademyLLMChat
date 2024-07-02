import time
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar, Generic, Optional

from loguru import logger
from nebula3.Config import Config as NConfig
from nebula3.data.ResultSet import ResultSet
from nebula3.gclient.net import ConnectionPool

T = TypeVar('T')


class VidType(StrEnum):
    STRING255 = 'FIXED_STRING(255)'
    INT64 = 'INT[64]'


class PropType(StrEnum):
    INT64 = 'int64'
    INT32 = 'int32'
    INT16 = 'int16'
    INT8 = 'int8'
    FLOAT = 'float'
    DOUBLE = 'double'
    BOOL = 'bool'
    STRING = 'string'
    DATE = 'date'
    TIME = 'time'
    DATETIME = 'datetime'
    TIMESTAMP = 'timestamp'
    DURATION = 'duration'
    GEO = 'geography'
    POINT = 'geography(point)'
    LINESTRING = 'geography(linestring)'
    POLYGON = 'geography(polygon)'


@dataclass
class Prop(Generic[T]):
    prop_name: str
    data_type: str
    not_null: bool = False
    default: Optional[T] = None
    comment: str = ''

    def __str__(self):
        prop_str = f'{self.prop_name} {self.data_type}'
        if self.not_null:
            prop_str += ' NOT NULL'

        if self.default is not None:
            if isinstance(self.default, str):
                prop_str += f' DEFAULT {self.default}' if self.default.endswith("()") else f' DEFAULT "{self.default}"'
            else:
                prop_str += f" DEFAULT {self.default}"

        if self.comment:
            prop_str += f' COMMENT "{self.comment}"'

        return prop_str


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
            check_exist: bool = True,
    ) -> ResultSet:
        """
        创建图空间

        :param space_name: 在NebulaGraph实例中唯一标识一个图空间，仅支持 1~4 字节的 UTF-8 编码字符，包括英文字母（区分大小写）、数字、中文等
        :param partition_num: 指定图空间的分片数量。建议设置为集群中硬盘数量的 20 倍（HDD 硬盘建议为 2 倍）
        :param replica_factor: 指定每个分片的副本数量。建议在生产环境中设置为 3，在测试环境中设置为 1。副本数量必须是奇数
        :param vid_type: 指定点 ID 的数据类型
        :param comment: 图空间的描述
        :param check_exist:
        :return:

        CREATE SPACE [IF NOT EXISTS] <graph_space_name> (
        [partition_num = <partition_number>,]
        [replica_factor = <replica_number>,]
        vid_type = {FIXED_STRING(<N>) | INT[64]}
        )
        [COMMENT = '<comment>'];
        """

        assert vid_type == 'INT[64]' or vid_type.startswith('FIXED_STRING')

        if check_exist:
            stmt = (f'CREATE SPACE IF NOT EXISTS {space_name} ('
                    f'partition_num={partition_num}, replica_factor={replica_factor}, vid_type={vid_type})')
        else:
            stmt = (f'CREATE SPACE {space_name} ('
                    f'partition_num={partition_num}, replica_factor={replica_factor}, vid_type={vid_type})')

        if comment is not None:
            stmt += f' comment="{comment}"'

        result = self.client.execute(stmt)

        logger.info('Creating graph database, please wait...')
        time.sleep(3)
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

    def clear_space(self, space_name: str, *, check_exist: bool = True) -> ResultSet:
        """
        清空图空间中的点和边，但不会删除图空间本身以及其中的 Schema 信息

        :param space_name: 图空间唯一标识
        :param check_exist:
        :return:

        CLEAR SPACE [IF EXISTS] <graph_space_name>
        """
        stmt = f'CLEAR SPACE IF EXISTS {space_name};' if check_exist else f'CLEAR SPACE {space_name};'

        result = self.client.execute(stmt)
        return result

    def drop_space(self, space_name: str, *, check_exist: bool = True) -> ResultSet:
        """
        删除指定图空间

        :param space_name: 图空间唯一标识
        :param check_exist:
        :return:

        DROP SPACE [IF EXISTS] <graph_space_name>
        """
        stmt = f'DROP SPACE IF EXISTS {space_name};' if check_exist else f'DROP SPACE {space_name};'
        result = self.client.execute(stmt)

        logger.info('Deleting graph database, please wait...')
        time.sleep(3)
        logger.info('done')
        return result

    def create_tag(
            self,
            tag_name: str, props: list[str], *,
            ttl_duration: int = None, ttl_col: Prop = None, comment: str = '', check_exist: bool = True
    ) -> ResultSet:
        """
        创建TAG

        :param tag_name: Tag名称,每个图空间内的Tag必须是唯一的。包括英文字母（区分大小写）、数字、中文等。不能包含下划线以外的特殊字符，且不能以数字开头
        :param props: 属性列表
        :param ttl_duration: 指定时间戳差值，单位：秒
        :param ttl_col: 指定要设置存活时间的属性, 属性的数据类型必须是int或者timestamp
        :param comment:
        :param check_exist:
        :return:

        CREATE TAG [IF NOT EXISTS] <tag_name> (
        <prop_name> <data_type>
        [NULL | NOT NULL]
        [DEFAULT <default_value>]
        [COMMENT '<comment>']
        [{, <prop_name> <data_type> [NULL | NOT NULL] [DEFAULT <default_value>] [COMMENT '<comment>']} ...]
        )
        [TTL_DURATION = <ttl_duration>]
        [TTL_COL = <prop_name>]
        [COMMENT = '<comment>']
        """

        prop_str = ' ,'.join(props)
        stmt_1 = f'CREATE TAG IF NOT EXISTS {tag_name}({prop_str})' if check_exist else f'CREATE TAG {tag_name}({prop_str})'
        stmt_2 = []
        if ttl_duration is not None:
            assert ttl_col and ttl_col.data_type in [PropType.INT64, PropType.INT32, PropType.INT16, PropType.INT8, PropType.TIMESTAMP]
            stmt_2.append(f'TTL_DURATION = {ttl_duration}')
            stmt_2.append(f'TTL_COL = "{ttl_col.prop_name}"')

        if comment:
            stmt_2.append(f'COMMENT = "{comment}"')

        stmt = f'{stmt_1} {", ".join(stmt_2)};'
        result = self.client.execute(stmt)

        return result

    def drop_tag(self, tag_name: str, *, check_exist: bool = True) -> ResultSet:
        """
        删除当前工作空间内所有点上的指定 Tag

        :param tag_name: 要删除的tag名称
        :param check_exist:
        :return:

        DROP TAG [IF EXISTS] <tag_name>;
        """
        stmt = f'DROP TAG IF EXISTS {tag_name};' if check_exist else f'DROP TAG {tag_name};'
        result = self.client.execute(stmt)
        return result

    def delete_tag(self, tag_names: list[str], vid_list: list[str]):
        """

        :param tag_names:
        :param vid_list:
        :return:

        DELETE TAG <tag_name_list> FROM <VID_list>;
        """

        if len(tag_names) == 0:
            tag_str = '*'
        else:
            tag_str = ','.join(tag_names)

        vid_list = [f'"{vid}"' for vid in vid_list]

        stmt = f'DELETE TAG {tag_str} FROM {",".join(vid_list)};'
        result = self.client.execute(stmt)
        return result

    def create_edge(
            self,
            edge_name: str, props: list[str], *,
            ttl_duration: int = None, ttl_col: Prop = None, comment: str = '', check_exist: bool = True
    ) -> ResultSet:
        """
        创建边类型

        :param edge_name: 边类型名称
        :param props: 属性列表
        :param ttl_duration: 指定时间戳差值，单位：秒
        :param ttl_col: 指定要设置存活时间的属性, 属性的数据类型必须是int或者timestamp
        :param comment:
        :param check_exist:
        :return:

        CREATE EDGE [IF NOT EXISTS] <edge_type_name>(
          <prop_name> <data_type> [NULL | NOT NULL] [DEFAULT <default_value>] [COMMENT '<comment>']
          [{, <prop_name> <data_type> [NULL | NOT NULL] [DEFAULT <default_value>] [COMMENT '<comment>']} ...]
        )
        [TTL_DURATION = <ttl_duration>]
        [TTL_COL = <prop_name>]
        [COMMENT = '<comment>'];
        """

        prop_str = ' ,'.join(props)
        stmt_1 = f'CREATE EDGE IF NOT EXISTS {edge_name}({prop_str})' if check_exist else f'CREATE EDGE {edge_name}({prop_str})'
        stmt_2 = []
        if ttl_duration is not None:
            assert ttl_col and ttl_col.data_type in [PropType.INT64, PropType.INT32, PropType.INT16, PropType.INT8, PropType.TIMESTAMP]
            stmt_2.append(f'TTL_DURATION = {ttl_duration}')
            stmt_2.append(f'TTL_COL = "{ttl_col.prop_name}"')

        if comment:
            stmt_2.append(f'COMMENT = "{comment}"')

        stmt = f'{stmt_1} {", ".join(stmt_2)};'
        result = self.client.execute(stmt)
        return result

    def drop_edge(self, edge_name: str, *, check_exist: bool = True) -> ResultSet:
        """
        删除当前工作空间内的指定 Edge type

        :param edge_name: 要删除的edge名称
        :param check_exist:
        :return:

        DROP EDGE [IF EXISTS] <edge_type_name>;
        """
        stmt = f'DROP EDGE IF EXISTS {edge_name};' if check_exist else f'DROP EDGE {edge_name};'
        result = self.client.execute(stmt)
        return result


def main() -> None:
    with NebulaGraphStore() as store:
        # print(store.create_space('reference', vid_type=VidType.STRING255))
        # result = store.drop_space('reference')
        store.use_space('reference')

        # prop1 = Prop('DOI', PropType.STRING, True)
        # prop2 = Prop('Title', PropType.STRING, True)
        # prop3 = Prop('Author', PropType.STRING)
        # props = [str(prop1), str(prop2), str(prop3)]
        # print(store.create_tag('paper', props))

        # prop1 = Prop('ref_id', PropType.INT64, True)
        # props = [str(prop1)]
        # print(store.create_edge('cite', props))


if __name__ == '__main__':
    main()
