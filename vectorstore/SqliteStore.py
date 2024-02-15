import sqlite3
from typing import Any, Generic, Iterator, List, Optional, Sequence, Tuple, TypeVar

from langchain_core.documents import Document
from langchain_core.load import Serializable, dumps, loads
from langchain_core.stores import BaseStore
from loguru import logger

V = TypeVar("V")

ITERATOR_WINDOW_SIZE = 1000

_LANGCHAIN_DEFAULT_TABLE_NAME = "langchain"


class SqliteBaseStore(BaseStore[str, V], Generic[V]):
    def __init__(
            self,
            connection_string: str,
            table_name: str = _LANGCHAIN_DEFAULT_TABLE_NAME,
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
        conn = sqlite3.connect(self.connection_string, **self.engine_args)
        return conn

    def __create_tables_if_not_exists(self) -> None:
        cur = self._conn.cursor()
        res = cur.execute(f"SELECT name FROM sqlite_master WHERE name='{self.table_name}'")
        if res.fetchone() is None:
            stmt = f"""CREATE TABLE {self.table_name}
                    (
                        content TEXT,
                        custom_id TEXT
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

    def __serialize_value(self, obj: V) -> str:
        if isinstance(obj, Serializable):
            return dumps(obj)
        return obj

    def __deserialize_value(self, obj: V) -> str:
        try:
            return loads(obj)
        except Exception:
            return obj

    def mget(self, keys: Sequence[str]) -> List[Optional[V]]:
        cur = self._conn.cursor()
        query = f"""
        SELECT content, custom_id 
        FROM {self.table_name} 
        WHERE custom_id  IN ({','.join(['?'] * len(keys))})
        """

        cur.execute(query, keys)
        items = cur.fetchall()
        cur.close()

        ordered_values = {key: None for key in keys}
        for item in items:
            v = item[0]
            val = self.__deserialize_value(v) if v is not None else v
            k = item[1]
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
            stmt = f"DELETE FROM {self.table_name} WHERE custom_id IN ({','.join(['?'] * len(keys))})"
            cur.execute(stmt)
        self._conn.commit()
        cur.close()

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        cur = self._conn.cursor()
        start = 0
        while True:
            stop = start + ITERATOR_WINDOW_SIZE
            query = f"SELECT custom_id FROM {self.table_name}"
            if prefix is not None:
                query += f" AND custom_id LIKE '{prefix}%'"
            query += f" LIMIT {start}, {ITERATOR_WINDOW_SIZE}"
            cur.execute(query)
            items = cur.fetchall()

            if len(items) == 0:
                break
            for item in items:
                yield item[0]
            start += ITERATOR_WINDOW_SIZE

        cur.close()


SqliteDocStore = SqliteBaseStore[Document]
SqliteStrStore = SqliteBaseStore[str]
