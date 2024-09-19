from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
import torch.cuda
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.runnables import run_in_executor
from loguru import logger
from torch import Tensor
from tqdm import tqdm

class BgeM3Embeddings(BaseModel, Embeddings):
    model_name: str
    tokenizer: Any = None
    model: Any = None

    """
    Keyword arguments to pass to the model.
    
    pooling_method: str = 'cls',
    use_fp16: bool = True,
    device: str = None
    """
    pooling_method: str = 'cls'
    use_fp16: bool = True
    device: Optional[str] = None

    """
    Keyword arguments to pass when calling the `encode` method of the model.
    
    normalize_embeddings: bool = True,
    batch_size: int = 12,
    max_length: int = 8192,
    """
    encode_kwargs: dict[str, Any] = Field(default_factory=dict)

    local_load: bool = False
    local_path: str = ''

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        try:
            from transformers import (
                AutoTokenizer,
                AutoModel,
                PreTrainedTokenizerFast,
                PreTrainedModel,
            )

        except ImportError as exc:
            raise ImportError(
                "Could not import transformers python package. "
                "Please install it with `pip install transformers`."
            ) from exc

        if self.local_load:
            try:
                self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(self.local_path)
                self.model: PreTrainedModel = AutoModel.from_pretrained(self.local_path)
            except EnvironmentError:
                logger.error('Load model from local fail. Download from huggingface...')

                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)

                # save to local
                self.tokenizer.save_pretrained(self.local_path)
                self.model.save_pretrained(self.local_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if not torch.cuda.is_available():
            self.use_fp16 = False

        if self.use_fp16:
            self.model.half()

        self.model = self.model.to(torch.device(self.device))
        self.model.eval()

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    def dense_embedding(self, hidden_state: Tensor, mask: Tensor) -> Tensor:
        if self.pooling_method == 'cls':
            return hidden_state[:, 0]
        elif self.pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(dim=1, keepdim=True).float()
            return s / d

    @torch.no_grad()
    def encode(
            self,
            sentences: Union[list[str], str],
            normalize_embeddings: bool = True,
            batch_size: int = 12,
            max_length: int = 8192,
    ) -> np.ndarray:
        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        all_dense_embeddings = []
        for start_index in tqdm(
                range(0, len(sentences), batch_size),
                desc="Inference Embeddings",
                disable=len(sentences) < 256
        ):
            sentences_batch = sentences[start_index:start_index + batch_size]
            batch_data = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length,
            ).to(self.device)

            # TODO
            last_hidden_state = self.model(**batch_data, return_dict=True).last_hidden_state
            dense_vecs = self.dense_embedding(last_hidden_state, batch_data['attention_mask'])

            if normalize_embeddings:
                dense_vecs = torch.nn.functional.normalize(dense_vecs, dim=-1)
            all_dense_embeddings.append(dense_vecs.cpu().numpy())

        all_dense_embeddings = np.concatenate(all_dense_embeddings, axis=0)
        if input_was_string:
            all_dense_embeddings = all_dense_embeddings[0]

        return all_dense_embeddings

    def embed_query(self, text: str) -> list[float]:
        text = text.replace("\n", " ")
        embedding = self.encode(text, **self.encode_kwargs)
        return embedding.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        texts = [t.replace("\n", " ") for t in texts]
        embeddings = self.encode(texts, **self.encode_kwargs)

        return embeddings.tolist()


class BgeReranker(BaseModel):
    model_name: str = 'BAAI/bge-reranker-v2-m3'
    tokenizer: Any = None
    model: Any = None

    """
    Keyword arguments to pass to the model.

    use_fp16: bool = False,
    device: Union[str, int] = None
    """
    use_fp16: bool = False,
    device: Optional[str] = None

    """
    Keyword arguments to pass when calling the `compress_documents` method of the model.

    batch_size: int = 256,
    max_length: int = 512, 
    normalize: bool = False
    """
    encode_kwargs: dict[str, Any] = Field(default_factory=dict)

    local_load: bool = False
    local_path: str = ''

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
                PreTrainedTokenizerFast,
                PreTrainedModel,
            )

        except ImportError as exc:
            raise ImportError(
                "Could not import transformers python package. "
                "Please install it with `pip install transformers`."
            ) from exc

        if self.local_load:
            try:
                self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(self.local_path)
                self.model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(self.local_path)
            except EnvironmentError:
                logger.error('Load model from local fail. Download from huggingface...')

                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

                # save to local
                self.tokenizer.save_pretrained(self.local_path)
                self.model.save_pretrained(self.local_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if not torch.cuda.is_available():
            self.use_fp16 = False

        if self.use_fp16:
            self.model.half()

        self.model = self.model.to(torch.device(self.device))
        self.model.eval()

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    @torch.no_grad()
    def compute_score(
            self,
            sentence_pairs: Union[list[tuple[str, str]], tuple[str, str]],
            batch_size: int = 256,
            max_length: int = 512,
            normalize: bool = False
    ) -> list[float]:

        assert isinstance(sentence_pairs, list)
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]

        all_scores = []
        for start_index in tqdm(
                range(0, len(sentence_pairs), batch_size),
                desc="Compute Scores",
                disable=len(sentence_pairs) < 128
        ):
            sentences_batch = sentence_pairs[start_index:start_index + batch_size]
            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length,
            ).to(self.device)

            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            all_scores.extend(scores.cpu().numpy().tolist())

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        if normalize:
            all_scores = [sigmoid(score) for score in all_scores]

        return all_scores

    def compress_documents(
            self,
            documents: list[Document],
            query: str,
            callbacks: Optional[Callbacks] = None,
    ) -> list[Document]:

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

            batch_scores = self.compute_score(batch_pairs, **self.encode_kwargs)
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


def main() -> None:
    embedding = BgeM3Embeddings(
        model_name='BAAI/bge-m3',
        use_fp16=True,
        encode_kwargs={
            'normalize_embeddings': True
        },
        local_load=True,
        local_path='../data/model/BAAI/bge-m3'
    )

    sentences = ["What is BGE M3?", "Defination of BM25"]
    print(embedding.embed_documents(sentences))
    # reranker = BgeReranker(
    #     use_fp16=True,
    #     encode_kwargs={
    #         'normalize': True
    #     },
    #     local_load=True,
    #     local_path='../model/BAAI/bge-reranker-v2-m3'
    # )
    # docs = [
    #     Document(page_content='hi'),
    #     Document(
    #         page_content='The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.')
    # ]
    # query = 'what is panda?'
    # print(reranker.compress_documents(docs, query))


if __name__ == '__main__':
    main()
