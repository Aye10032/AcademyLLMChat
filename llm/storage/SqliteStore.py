import sqlite3
from typing import Any, Generic, Iterator, List, Optional, Sequence, Tuple, TypeVar

from langchain_core.documents import Document
from langchain_core.load import Serializable, dumps, loads
from langchain_core.stores import BaseStore
from loguru import logger

from utils.MarkdownPraser import Reference

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


SqliteDocStore = SqliteBaseStore[Document]


def main() -> None:
    doc_store: SqliteBaseStore = SqliteDocStore(
        connection_string='../../data/test/sqlite/document.db'
    )

    docs = doc_store.mget(['215e7644-558b-4c0e-82ec-c6c7b80f5f82'])
    print(docs)

    # for ids in [
    #     '6e8664e6-186d-4451-b75d-0e341368f047', '215e7644-558b-4c0e-82ec-c6c7b80f5f82', '18112a0c-b4ea-421f-acab-86517bb7ce6f',
    #     '382f6616-d9fb-43e1-877f-2764275ee79e', '70eb810e-f05e-4a09-a446-4786699deecf', '60978dc6-1b91-4f54-b431-14219e5bedb3',
    #     '9b3ae9ce-59e6-4344-abc5-90ade3ce1fa3', 'ce339e57-754e-4715-8b78-a7192a8e5bd3', '59058637-7a87-42f7-855d-bf7739e74e67',
    #     '286d774d-9022-49dd-a768-c77d7f3ce48f', '910d43e9-4b35-484c-99f7-fcbb9fe00e85', '3a6f6d63-bc40-4428-ae2f-bb1ac917cc65',
    #     '909c81bd-2cd4-4a28-9c62-d55b3f6c6ea2', 'f12d70a8-63e2-4d93-809c-fca736a64447', '18106ce7-d8dd-4c3a-a9ea-1231706a7d46',
    #     'ac8c6b7b-c000-4c66-80cb-1c8aaaf41624', '4f19a0c8-96d9-457b-b916-18d834ad286d', 'c3ddd415-05b5-4264-9c62-03593a383045'
    # ]:
    #     docs = doc_store.mget([ids])
    #     query = 'test'
    #     sentence_pairs = [(query, doc.page_content) for doc in docs]
    #     print(sentence_pairs)


if __name__ == '__main__':
    main()
