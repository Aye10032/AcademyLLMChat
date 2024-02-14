"""SQL storage that persists data in a SQL database
and supports data isolation using collections."""
# from __future__ import annotations

import sqlite3
import uuid
from typing import Any, Generic, Iterator, List, Optional, Sequence, Tuple, TypeVar

# import sqlalchemy
# from sqlalchemy import JSON, UUID
# from sqlalchemy.orm import Session, relationship
#
# try:
#     from sqlalchemy.orm import declarative_base
# except ImportError:
#     from sqlalchemy.ext.declarative import declarative_base

from langchain_core.documents import Document
from langchain_core.load import Serializable, dumps, loads
from langchain_core.stores import BaseStore

V = TypeVar("V")

ITERATOR_WINDOW_SIZE = 1000

# Base = declarative_base()  # type: Any


_LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"


# class BaseModel(Base):
#     """Base model for the SQL stores."""
#
#     __abstract__ = True
#     uuid = sqlalchemy.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
#
#
# _classes: Any = None
#
#
# def _get_storage_stores() -> Any:
#     global _classes
#     if _classes is not None:
#         return _classes
#
#     class CollectionStore(BaseModel):
#         """Collection store."""
#
#         __tablename__ = "langchain_storage_collection"
#
#         name = sqlalchemy.Column(sqlalchemy.String)
#         cmetadata = sqlalchemy.Column(JSON)
#
#         items = relationship(
#             "ItemStore",
#             back_populates="collection",
#             passive_deletes=True,
#         )
#
#         @classmethod
#         def get_by_name(
#             cls, session: Session, name: str
#         ) -> Optional["CollectionStore"]:
#             # type: ignore
#             return session.query(cls).filter(cls.name == name).first()
#
#         @classmethod
#         def get_or_create(
#             cls,
#             session: Session,
#             name: str,
#             cmetadata: Optional[dict] = None,
#         ) -> Tuple["CollectionStore", bool]:
#             """
#             Get or create a collection.
#             Returns [Collection, bool] where the bool is True if the collection was created.
#             """  # noqa: E501
#             created = False
#             collection = cls.get_by_name(session, name)
#             if collection:
#                 return collection, created
#
#             collection = cls(name=name, cmetadata=cmetadata)
#             session.add(collection)
#             session.commit()
#             created = True
#             return collection, created
#
#     class ItemStore(BaseModel):
#         """Item store."""
#
#         __tablename__ = "langchain_storage_items"
#
#         collection_id = sqlalchemy.Column(
#             UUID(as_uuid=True),
#             sqlalchemy.ForeignKey(
#                 f"{CollectionStore.__tablename__}.uuid",
#                 ondelete="CASCADE",
#             ),
#         )
#         collection = relationship(CollectionStore, back_populates="items")
#
#         content = sqlalchemy.Column(sqlalchemy.String, nullable=True)
#
#         # custom_id : any user defined id
#         custom_id = sqlalchemy.Column(sqlalchemy.String, nullable=True)
#
#     _classes = (ItemStore, CollectionStore)
#
#     return _classes


class SQLBaseStore(BaseStore[str, V], Generic[V]):
    def __init__(
            self,
            connection_string: str,
            table_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
            collection_metadata: Optional[dict] = None,
            pre_delete_collection: bool = False,
            connection: Optional[sqlite3.connect] = None,
            engine_args: Optional[dict[str, Any]] = None,
    ) -> None:
        self.connection_string = connection_string
        self.table_name = table_name
        self.collection_metadata = collection_metadata
        self.pre_delete_collection = pre_delete_collection
        self.engine_args = engine_args or {}
        # Create a connection if not provided, otherwise use the provided connection
        self._conn = connection if connection else self.__connect()
        self.__post_init__()

    def __post_init__(
            self,
    ) -> None:
        """Initialize the store."""
        # ItemStore, CollectionStore = _get_storage_stores()
        # self.CollectionStore = CollectionStore
        # self.ItemStore = ItemStore
        self.__create_tables_if_not_exists()
        # self.__create_collection()

    def __connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.connection_string)
        return conn

    def __create_tables_if_not_exists(self) -> None:
        cur = self._conn.cursor()
        res = cur.execute(f"SELECT name FROM sqlite_master WHERE name='{self.table_name}'")
        if res.fetchone() is None:
            stmt = f"""create table {self.table_name}
                    (
                        content TEXT,
                        custom_id TEXT
                    );
                    """
            cur.execute(stmt)
            self._conn.commit()

        cur.close()

    # def __create_collection(self) -> None:
    #     if self.pre_delete_collection:
    #         self.delete_collection()
    #     with Session(self._conn) as session:
    #         self.CollectionStore.get_or_create(
    #             session, self.collection_name, cmetadata=self.collection_metadata
    #         )

    # def delete_collection(self) -> None:
    #     with self._conn.cursor() as cur:
    #         collection = self.__get_collection(session)
    #         if not collection:
    #             return
    #         session.delete(collection)
    #         session.commit()

    # def __get_collection(self, session: Session) -> Any:
    #     return self.CollectionStore.get_by_name(session, self.collection_name)

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


SQLDocStore = SQLBaseStore[Document]
SQLStrStore = SQLBaseStore[str]
