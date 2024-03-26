from typing import List, Any, Dict, Sequence, Optional

from langchain_community.embeddings.huggingface import (
    DEFAULT_QUERY_BGE_INSTRUCTION_EN,
    DEFAULT_QUERY_BGE_INSTRUCTION_ZH,
    DEFAULT_BGE_MODEL
)
from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Field


class Bgem3Embeddings(BaseModel, Embeddings, BaseDocumentCompressor):
    model_name: str = DEFAULT_BGE_MODEL

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

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        try:
            from FlagEmbedding import BGEM3FlagModel

        except ImportError as exc:
            raise ImportError(
                "Could not import FlagEmbedding python package. "
                "Please install it with `pip install FlagEmbedding`."
            ) from exc

        self.client: BGEM3FlagModel = BGEM3FlagModel(self.model_name, **self.model_kwargs)

        if "-zh" in self.model_name:
            self.query_instruction = DEFAULT_QUERY_BGE_INSTRUCTION_ZH

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
            documents: Sequence[Document],
            query: str,
            callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        sentence_pairs = [(query, doc.page_content) for doc in documents]
        rerank_scores = self.client.compute_score(sentence_pairs, weights_for_different_modes=[0, 0.8, 0.2])
        rerank_results = list(zip(rerank_scores, documents))
        rerank_results = sorted(rerank_results, reverse=True)

        final_results = []
        for r in rerank_results:
            doc: Document = r[1]
            doc.metadata['score'] = r[0]
            final_results.append(doc)
        return final_results
