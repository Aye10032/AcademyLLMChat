import os.path
from typing import List, Any, Dict, Sequence, Optional

from langchain_community.embeddings.huggingface import (
    DEFAULT_QUERY_BGE_INSTRUCTION_EN,
    DEFAULT_QUERY_BGE_INSTRUCTION_ZH,
    DEFAULT_BGE_MODEL
)
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import run_in_executor
from loguru import logger


class Bgem3Embeddings(BaseModel, Embeddings):
    model_name: str = DEFAULT_BGE_MODEL
    client: Any

    """
    Keyword arguments to pass to the model.
    
    pooling_method: str = 'cls',
    normalize_embeddings: bool = True,
    use_fp16: bool = True,
    device: str = None
    """
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    """
    Keyword arguments to pass when calling the `encode` method of the model.
    
    batch_size: int = 12,
    max_length: int = 8192,
    return_dense: bool = True,
    return_sparse: bool = False,
    return_colbert_vecs: bool = False
    """
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)

    """Instruction to use for embedding query."""
    query_instruction: str = DEFAULT_QUERY_BGE_INSTRUCTION_EN

    """Instruction to use for embedding document."""
    embed_instruction: str = ""

    local_load: bool = False
    local_path: str = ''

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        try:
            from FlagEmbedding import BGEM3FlagModel

        except ImportError as exc:
            raise ImportError(
                "Could not import FlagEmbedding python package. "
                "Please install it with `pip install FlagEmbedding`."
            ) from exc

        if self.local_load:
            try:
                self.client: BGEM3FlagModel = BGEM3FlagModel(self.local_path, **self.model_kwargs)
            except EnvironmentError:
                logger.error('Load model from local fail. Download from huggingface...')

                self.client: BGEM3FlagModel = BGEM3FlagModel(self.model_name, **self.model_kwargs)
                self.client.model.save(self.local_path)
                self.client.tokenizer.save_pretrained(self.local_path)
        else:
            self.client: BGEM3FlagModel = BGEM3FlagModel(self.model_name, **self.model_kwargs)

        if "-zh" in self.model_name:
            self.query_instruction = DEFAULT_QUERY_BGE_INSTRUCTION_ZH

    def load_model(self, model_name, **kwargs):
        return

    def embed_query(self, text: str) -> List[float]:
        text = text.replace("\n", " ")
        embedding = self.client.encode(
            self.query_instruction + text, **self.encode_kwargs
        )['dense_vecs']
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [self.embed_instruction + t.replace("\n", " ") for t in texts]
        embeddings = self.client.encode(texts, **self.encode_kwargs)['dense_vecs']

        return embeddings.tolist()

    def compress_documents(
            self,
            documents: list[Document],
            query: str,
            callbacks: Optional[Callbacks] = None,
    ) -> List[Document]:

        sentence_pairs = [
            (query, doc.page_content)
            for doc in documents
            if isinstance(doc, Document)
        ]
        rerank_scores = []

        for i in range(0, len(sentence_pairs), 10):
            if i + 10 >= len(sentence_pairs):
                batch_pairs = sentence_pairs[i:]
            else:
                batch_pairs = sentence_pairs[i:i + 10]

            batch_scores = self.client.compute_score(
                batch_pairs,
                weights_for_different_modes=[0.5, 0.2, 0.3]
            ).get('colbert+sparse+dense')
            rerank_scores.extend(batch_scores)

        rerank_results = list(zip(rerank_scores, documents))
        rerank_results = sorted(rerank_results, reverse=True)

        final_results = []
        for r in rerank_results:
            doc: Document = r[1]
            doc.metadata['score'] = r[0]
            final_results.append(doc)
        return final_results

    async def acompress_documents(
            self,
            documents: list[Document],
            query: str,
            callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Compress retrieved documents given the query context."""
        return await run_in_executor(
            None, self.compress_documents, documents, query, callbacks
        )
