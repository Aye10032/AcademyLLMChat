import sqlite3
from datetime import datetime
from typing import Any, Generic, Iterator, List, Optional, Sequence, Tuple, TypeVar

import pandas as pd
from langchain_core.documents import Document
from langchain_core.load import dumps, loads
from langchain_core.stores import BaseStore
from loguru import logger
from werkzeug.security import generate_password_hash, check_password_hash

from utils.MarkdownPraser import Reference
from utils.entities.UserProfile import User, UserGroup, Project, ChatHistory

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

    def init_tables(self) -> None:
        cur = self._conn.cursor()

        res = cur.execute(f"SELECT name FROM sqlite_master WHERE name='user'")
        if res.fetchone() is None:
            create_stmt = f"""
                create table user(
                    id          INTEGER  not null
                        primary key autoincrement,
                    name         TEXT    not null,
                    passwd       TEXT    not null,
                    user_group   INTEGER not null,
                    last_project TEXT
                );"""
            cur.execute(create_stmt)
            self._conn.commit()
            logger.info(f'Create table user')

        res = cur.execute(f"SELECT name FROM sqlite_master WHERE name='project'")
        if res.fetchone() is None:
            create_stmt = f"""
                create table project(
                    id          INTEGER            not null
                        primary key autoincrement,
                    name        TEXT               not null,
                    owner       TEXT               not null,
                    last_chat   TEXT               not null,
                    update_time TIMESTAMP          not null,
                    create_time TIMESTAMP          not null,
                    time_zone   TEXT               not null 
                );"""
            cur.execute(create_stmt)
            self._conn.commit()
            logger.info(f'Create table project')

        res = cur.execute(f"SELECT name FROM sqlite_master WHERE name='chat_history'")
        if res.fetchone() is None:
            create_stmt = """
                create table chat_history
                (
                    id          INTEGER   not null
                        primary key autoincrement,
                    session_id  TEXT      not null,
                    description TEXT      not null,
                    owner       TEXT      not null,
                    project     TEXT      not null,
                    update_time TIMESTAMP not null,
                    create_time TIMESTAMP not null
                );"""
            cur.execute(create_stmt)
            self._conn.commit()
            logger.info(f'Create table chat_history')

        cur.close()

    def user_exists(self, name: str) -> bool:
        """
        Check if a user with the given name already exists in the 'user' table.

        :param name: The name of the user to check.
        :return: True if the user does not exist, False if the user exists.
        """
        cur = self._conn.cursor()
        cur.execute("SELECT id FROM user WHERE name = ?", (name,))
        result = cur.fetchone()
        cur.close()

        return result is not None

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

        if not self.user_exists(user.name):
            hashed_password = generate_password_hash(user.password)

            stmt = """
            INSERT INTO user (name, passwd, user_group, last_project)
            VALUES (?, ?, ?, ?)
            """
            cur.execute(stmt, (user.name, hashed_password, user.user_group, user.last_project))
            self._conn.commit()
            cur.close()

            logger.info(f'Create user {user.name}')
            return True
        else:
            cur.close()

            logger.warning(f'User {user.name} already exist!')
            return False

    def valid_user(self, user_name: str, passwd: str) -> tuple[bool, User | None]:
        cur = self._conn.cursor()
        try:
            # 查询用户信息
            cur.execute("SELECT * FROM user WHERE name = ?", (user_name,))
            result = cur.fetchone()

            if result is None:
                logger.warning(f"User '{user_name}' does not exist!")
                return False, None

            _, name, hashed_password, user_group, last_project = result

            if check_password_hash(hashed_password, passwd):
                user = User(
                    name=name,
                    password='',
                    user_group=UserGroup(user_group),
                    last_project=last_project
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
            cur.execute("SELECT name, user_group, last_project FROM user")
            results = cur.fetchall()

            if not results:
                return pd.DataFrame(columns=['name', 'user group', 'last_project'])

            df = pd.DataFrame(results, columns=['name', 'user group', 'last project'])
            df['user group'] = df['user group'].apply(lambda x: UserGroup(x).name)

            return df
        except Exception as e:
            logger.error(f"Error occurred while retrieving the user list: {str(e)}")
            return pd.DataFrame(columns=['name', 'user group', 'last project'])
        finally:
            cur.close()

    def update_user(self, user: User) -> bool:
        cur = self._conn.cursor()

        if not self.user_exists(user.name):
            logger.warning(f'User {user.name} does not exist!')
            return False

        stmt = """
            UPDATE user 
            SET user_group = ?, last_project = ? 
            WHERE name = ?
            """
        cur.execute(stmt, (user.user_group, user.last_project, user.name))
        self._conn.commit()
        cur.close()

        logger.info(f'Updated user {user.name}')
        return True

    def project_exists(self, owner: str, project_name: str) -> bool:
        """
        Check if a project with the given name and owner exists in the 'project' table.

        :param project_name: The name of the project to check.
        :param owner: The owner of the project to check.
        :return: True if the project exists, False otherwise.
        """
        cur = self._conn.cursor()
        cur.execute("SELECT id FROM project WHERE name = ? AND owner = ?", (project_name, owner,))
        result = cur.fetchone()
        cur.close()

        return result is not None

    def create_project(self, project: Project) -> bool:
        cur = self._conn.cursor()

        if not self.user_exists(project.owner):
            logger.warning(f'User {project.owner} dose not exits!')
            return False

        if not self.project_exists(project.owner, project.name):
            stmt = """
                INSERT INTO project (name, owner, last_chat, update_time, create_time, time_zone)
                VALUES (?, ?, ?, ?, ?, ?)
                """
            cur.execute(stmt, (
                project.name,
                project.owner,
                project.last_chat,
                project.update_time,
                project.create_time,
                project.time_zone
            ))
            self._conn.commit()
            cur.close()

            logger.info(f'Create project {project.owner}/{project.name}')
            return True
        else:
            cur.close()

            logger.warning(f'Project {project.owner}/{project.name} already exist!')
            return False

    def get_project_list(self, user: str) -> list[Project]:
        cur = self._conn.cursor()

        try:
            cur.execute("SELECT * FROM project where owner=?", (user,))
            results = cur.fetchall()

            return [
                Project.from_list(result)
                for result in results
            ]
        except Exception as e:
            logger.error(f"Error occurred while retrieving the user project list: {str(e)}")
            return []
        finally:
            cur.close()

    def get_project(self, owner: str, project_name: str) -> Optional[Project]:
        cur = self._conn.cursor()

        try:
            cur.execute("SELECT * FROM project where owner=? AND name=?", (owner, project_name,))
            result = cur.fetchone()

            if result:
                return Project.from_list(result)
            else:
                logger.warning(f'Project {owner}/{project_name} does not exist!')
        except Exception as e:
            logger.error(f"Error occurred while retrieving project {owner}/{project_name}: {str(e)}")
            return None
        finally:
            cur.close()

    def update_project(self, project: Project) -> bool:
        cur = self._conn.cursor()

        try:
            stmt = """
                UPDATE project 
                SET last_chat = ?, update_time = ? 
                WHERE name = ? AND owner = ?
                """

            cur.execute(stmt, (
                project.last_chat,
                project.update_time,
                project.name,
                project.owner,
            ))
            self._conn.commit()

            logger.info(f'Update project {project.owner}/{project.name}')
            return True
        except Exception as e:
            logger.error(f"Error while update project {project.owner}/{project.name}: {e}")
            return False
        finally:
            cur.close()

    def chat_exists(
            self,
            owner: str,
            project: str,
            session_id: str,
    ) -> bool:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT id FROM chat_history WHERE session_id = ? AND project = ? AND owner = ?",
            (session_id, project, owner,)
        )
        result = cur.fetchone()
        cur.close()

        return result is not None

    def get_chat_list(self, owner: str, project_name: str) -> list[ChatHistory]:
        cur = self._conn.cursor()

        try:
            cur.execute(
                "SELECT * FROM chat_history where owner=? AND project=?",
                (owner, project_name,)
            )
            results = cur.fetchall()

            return [
                ChatHistory.from_list(result)
                for result in results
            ]
        except Exception as e:
            logger.error(f"Error occurred while retrieving chat list for {owner}/{project_name}: {str(e)}")
            return []
        finally:
            cur.close()

    def create_chat_history(self, chat_history: ChatHistory):
        if not self.user_exists(chat_history.owner):
            logger.warning(f'User {chat_history.owner} does not exits!')
            return False

        if not self.project_exists(chat_history.owner, chat_history.project):
            logger.warning(f'Project {chat_history.owner}/{chat_history.project} does not exits!')
            return False

        if self.chat_exists(chat_history.owner, chat_history.project, chat_history.session_id):
            logger.warning(f'Chat {chat_history.session_id} already exist!')
            return False

        cur = self._conn.cursor()
        try:
            stmt = """
                INSERT INTO chat_history (session_id, description, owner, project, update_time, create_time)
                VALUES (?, ?, ?, ?, ?, ?)
                """
            cur.execute(stmt, (
                chat_history.session_id,
                chat_history.description,
                chat_history.owner,
                chat_history.project,
                chat_history.update_time,
                chat_history.create_time,
            ))
            self._conn.commit()

            logger.info(f'Create new chat for {chat_history.owner}/{chat_history.project}')
            return True
        except Exception as e:
            logger.error(f'Error while creating new chat for {chat_history.owner}/{chat_history.project}: {str(e)}')
            return False
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
        user_group=UserGroup.ADMIN.value,
        last_project='test_project'
    )

    now_time = datetime.now().timestamp()
    project1 = Project(
        name='test',
        owner='user114',
        last_chat='14521',
        create_time=now_time,
        update_time=now_time,
    )

    with ProfileStore(
            connection_string='D:/program/github/AcademyLLMChat/data/user/user_info.db'
    ) as profile_store:
        # profile_store.init_tables()
        # profile_store.create_user(user)

        # user = profile_store.valid_user('test', '12345678')
        user_list = profile_store.get_users()
        print(user_list)
        #
        # print(profile_store.create_project(project1))
        # print(profile_store.create_project(project2))
        # print(profile_store.create_project(project2))


if __name__ == '__main__':
    main()
