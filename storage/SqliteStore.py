import sqlite3
from typing import Any, Generic, Iterator, List, Optional, Sequence, Tuple, TypeVar

import pandas as pd
from langchain_core.documents import Document
from langchain_core.load import Serializable, dumps, loads
from langchain_core.stores import BaseStore
from loguru import logger

from utils.MarkdownPraser import Reference
from utils.entities.UserProfile import User, UserGroup
from werkzeug.security import generate_password_hash, check_password_hash

V = TypeVar("V")

ITERATOR_WINDOW_SIZE = 500

LANGCHAIN_DEFAULT_TABLE_NAME = "langchain"
REFERENCE_DEFAULT_TABLE_NAME = "reference"


class SqliteBaseStore(BaseStore[str, V], Generic[V]):
    def __init__(
            self,
            connection_string: str,
            table_name: str = LANGCHAIN_DEFAULT_TABLE_NAME,
            drop_old: bool = False,
            connection: Optional[sqlite3.connect] = None,
            engine_args: Optional[dict[str, Any]] = None,
    ) -> None:
        self.connection_string = connection_string
        self.table_name = table_name
        self.drop_old = drop_old
        self.engine_args = engine_args or {}

        self._conn = connection if connection else self.__connect()
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.drop_old:
            self.__delete_table()
        self.__create_tables_if_not_exists()

    def __connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.connection_string, **self.engine_args, check_same_thread=False)
        return conn

    def __create_tables_if_not_exists(self) -> None:
        cur = self._conn.cursor()
        res = cur.execute(f"SELECT name FROM sqlite_master WHERE name='{self.table_name}'")
        if res.fetchone() is None:
            stmt = f"""CREATE TABLE {self.table_name}
                    (
                        content TEXT,
                        doc_id TEXT
                    );
                    """
            cur.execute(stmt)
            self._conn.commit()
            logger.info(f'Create table {self.table_name}')

        cur.close()

    def __delete_table(self):
        cur = self._conn.cursor()
        res = cur.execute(f"SELECT name FROM sqlite_master WHERE name='{self.table_name}'")
        if res.fetchone() is not None:
            stmt = f"DROP table {self.table_name}"
            cur.execute(stmt)
            self._conn.commit()

        cur.close()

    def __del__(self) -> None:
        if self._conn:
            self._conn.close()

    @staticmethod
    def __serialize_value(obj: V) -> str:
        try:
            return dumps(obj)
        except Exception as e:
            logger.error(e)

    @staticmethod
    def __deserialize_value(obj: V) -> Any:
        try:
            return loads(obj)
        except Exception as e:
            logger.error(e)

    def mget(self, keys: Sequence[str]) -> List[Optional[V]]:
        cur = self._conn.cursor()
        query = f"""
        SELECT content, doc_id 
        FROM {self.table_name} 
        WHERE doc_id  IN ({','.join(['?'] * len(keys))})
        """

        cur.execute(query, keys)
        items = cur.fetchall()
        cur.close()

        ordered_values = {key: type[Document] for key in keys}
        for item in items:
            v = item[0]
            val: Document = self.__deserialize_value(v)
            k = item[1]
            val.metadata['doc_id'] = k
            ordered_values[k] = val

        return [ordered_values[key] for key in keys]

    def mset(self, key_value_pairs: Sequence[Tuple[str, V]]) -> None:
        cur = self._conn.cursor()
        data = []
        for _id, item in key_value_pairs:
            content = self.__serialize_value(item)
            data.append((content, _id))

        cur.executemany(f"INSERT INTO {self.table_name} VALUES(?, ?)", data)
        self._conn.commit()
        cur.close()

    def mdelete(self, keys: Sequence[str]) -> None:
        cur = self._conn.cursor()
        res = cur.execute(f"SELECT name FROM sqlite_master WHERE name='{self.table_name}'")
        if res.fetchone() is None:
            raise ValueError("Collection not found")
        if keys is not None:
            stmt = f"DELETE FROM {self.table_name} WHERE doc_id IN ({','.join(['?'] * len(keys))})"
            cur.execute(stmt)
        self._conn.commit()
        cur.close()

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        cur = self._conn.cursor()
        start = 0
        while True:
            query = f"SELECT doc_id FROM {self.table_name}"
            if prefix is not None:
                query += f" AND doc_id LIKE '{prefix}%'"
            query += f" LIMIT {start}, {ITERATOR_WINDOW_SIZE}"
            cur.execute(query)
            items = cur.fetchall()

            if len(items) == 0:
                break
            for item in items:
                yield item[0]
            start += ITERATOR_WINDOW_SIZE

        cur.close()


class ReferenceStore:
    def __init__(
            self,
            connection_string: str,
            table_name: str = REFERENCE_DEFAULT_TABLE_NAME,
    ) -> None:
        self.connection_string = connection_string
        self.table_name = table_name

        self._conn = self.__connect()
        self.__post_init__()

    def __connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.connection_string, check_same_thread=False)
        return conn

    def __post_init__(self):
        cur = self._conn.cursor()

        res = cur.execute(f"SELECT name FROM sqlite_master WHERE name='{self.table_name}'")
        if res.fetchone() is None:
            stmt = f"""CREATE TABLE {self.table_name}
                            (
                                source_doi TEXT,
                                ref_id TEXT,
                                ref_doi TEXT,
                                ref_title TEXT,
                                ref_pmid TEXT,
                                ref_pmc TEXT
                            );
                            """
            cur.execute(stmt)
            self._conn.commit()
            logger.info(f'Create table {self.table_name}')

        cur.close()

    def drop_old(self):
        cur = self._conn.cursor()
        res = cur.execute(f"SELECT name FROM sqlite_master WHERE name='{self.table_name}'")
        if res.fetchone() is not None:
            stmt = f"DROP table {self.table_name}"
            cur.execute(stmt)
            self._conn.commit()

        cur.close()

    def __del__(self):
        if self._conn:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._conn:
            self._conn.close()

    def add_reference(self, ref_data: Reference) -> None:
        cur = self._conn.cursor()
        data = []
        source_doi = ref_data.source_doi
        ref_list = ref_data.ref_list
        for index, reference in enumerate(ref_list):
            if isinstance(reference, list):
                for sub_index, sub_ref in enumerate(reference):
                    data.append((
                        source_doi,
                        f'{index + 1}{chr(sub_index + 96)}',
                        sub_ref.get('doi'),
                        sub_ref.get('title'),
                        sub_ref.get('pmid'),
                        sub_ref.get('pmc'),
                    ))
            elif isinstance(reference, dict):
                data.append((
                    source_doi,
                    str(index + 1),
                    reference.get('doi'),
                    reference.get('title'),
                    reference.get('pmid'),
                    reference.get('pmc'),
                ))

        if len(data) > 0:
            cur.executemany(f"INSERT INTO {self.table_name} VALUES(?, ?, ?, ?, ?, ?)", data)
            self._conn.commit()

        cur.close()


class ProfileStore:
    def __init__(
            self,
            connection_string: str,
    ) -> None:
        self.connection_string = connection_string

        self._conn = self.__connect()

    def __connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.connection_string, check_same_thread=False)
        return conn

    def create_user(self, user: User) -> bool:
        """
        Create a new user in the database.

        This function checks if the 'user' table exists in the database and creates it if it does not.
        It then checks if a user with the given name already exists. If the user does not exist, it inserts
        the new user into the 'user' table with a hashed password.

        :param user: An instance of the User class containing the user's details.
        :return: True if the user was successfully created, False if the user already exists.
        """
        cur = self._conn.cursor()

        def check_table():
            """
            Check if the 'user' table exists in the database. If it does not exist, create it.
            """
            res = cur.execute("SELECT name FROM sqlite_master WHERE name='user'")
            if res.fetchone() is None:
                create_stmt = f"""
                        create table user(
                            name       TEXT    not null,
                            passwd     TEXT    not null,
                            user_group INTEGER not null
                        );"""
                cur.execute(create_stmt)
                self._conn.commit()
                logger.info(f'Create table user')

        def user_exists(name: str) -> bool:
            """
            Check if a user with the given name already exists in the 'user' table.

            :param name: The name of the user to check.
            :return: True if the user does not exist, False if the user exists.
            """
            cur.execute("SELECT name FROM user WHERE name = ?", (name,))
            result = cur.fetchone()

            return result is None

        check_table()
        if user_exists(user.name):
            cur = self._conn.cursor()
            hashed_password = generate_password_hash(user.password)

            stmt = """
            INSERT INTO user (name, passwd, user_group)
            VALUES (?, ?, ?)
            """
            cur.execute(stmt, (user.name, hashed_password, user.user_group))
            self._conn.commit()
            cur.close()

            logger.info(f'Create user {user.name}')
            return True
        else:
            logger.warning(f'User {user.name} already exist!')
            return False

    def valid_user(self, user_name: str, passwd: str) -> tuple[bool, User | None]:
        cur = self._conn.cursor()
        try:
            # 查询用户信息
            cur.execute("SELECT name, passwd, user_group FROM user WHERE name = ?", (user_name,))
            result = cur.fetchone()

            if result is None:
                logger.warning(f"用户 '{user_name}' 不存在")
                return False, None

            name, hashed_password, user_group = result

            if check_password_hash(hashed_password, passwd):
                user = User(
                    name=name,
                    password='',
                    user_group=UserGroup(user_group)
                )
                logger.info(f"User '{user_name}' valid success.")
                return True, user
            else:
                logger.warning(f"User '{user_name}' password error!")
                return False, None

        except Exception as e:
            logger.error(f"Error while valid user '{user_name}': {str(e)}")
            return False, None
        finally:
            cur.close()

    def get_users(self) -> pd.DataFrame:
        cur = self._conn.cursor()
        try:
            cur.execute("SELECT name, user_group FROM user")
            results = cur.fetchall()
            
            if not results:
                return pd.DataFrame(columns=['name', 'user_group'])
            
            df = pd.DataFrame(results, columns=['name', 'user_group'])
            df['user_group'] = df['user_group'].apply(lambda x: UserGroup(x).name)
            
            return df
        except Exception as e:
            logger.error(f"获取用户列表时出错: {str(e)}")
            return pd.DataFrame(columns=['name', 'user_group'])
        finally:
            cur.close()

    def __del__(self):
        if self._conn:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._conn:
            self._conn.close()


SqliteDocStore = SqliteBaseStore[Document]


def main() -> None:
    user = User(
        name='test',
        password='12345678',
        user_group=UserGroup.ADMIN.value
    )

    with ProfileStore(
            connection_string='D:/program/github/AcademyLLMChat/data/user/user_info.db'
    ) as profile_store:
        # profile_store.create_user(user)

        # user = profile_store.valid_user('test', '12345678')
        user_list = profile_store.get_users()
        print(user_list)


if __name__ == '__main__':
    main()
