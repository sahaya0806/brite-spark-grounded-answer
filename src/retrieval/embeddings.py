"""
Embedding provider interface and implementations.

Architecture
------------
``EmbeddingProvider`` is a simple Protocol that any embedding backend must
implement.  This allows:

- Production code to use the real OpenAI text-embedding-3-small model.
- Tests to use a deterministic ``FakeEmbeddingProvider`` that requires no
  API key and no network access.

The Protocol avoids coupling the vector index to a specific provider, keeps
tests fast and offline, and makes it easy to swap the embedding model without
touching the retrieval or indexing code.

Configuration
-------------
The OpenAI provider reads from environment variables:

    OPENAI_API_KEY   — required by the openai library
    OPENAI_EMBEDDING_MODEL — optional, defaults to "text-embedding-3-small"

These must be set in the .env file (which is git-ignored).
Never hard-code credentials.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class EmbeddingProvider(Protocol):
    """
    Minimal interface for an embedding backend.

    Both methods return 2-D float32 arrays of shape (n_texts, dim).
    """

    def embed_documents(self, texts: list[str]) -> NDArray[np.float32]:
        """
        Embed a batch of document texts.

        Parameters
        ----------
        texts:
            Non-empty list of strings.

        Returns
        -------
        NDArray[np.float32]
            Shape ``(len(texts), embedding_dim)``.
        """
        ...

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """
        Embed a single query string.

        Parameters
        ----------
        text:
            The user's query.

        Returns
        -------
        NDArray[np.float32]
            Shape ``(embedding_dim,)``.
        """
        ...


# ---------------------------------------------------------------------------
# OpenAI implementation
# ---------------------------------------------------------------------------

class OpenAIEmbeddingProvider:
    """
    Embedding provider backed by an OpenAI embeddings model.

    Configuration is read from environment variables so that no credential
    ever appears in source code.

    Parameters
    ----------
    model:
        OpenAI embedding model name.  Defaults to the value of the
        ``OPENAI_EMBEDDING_MODEL`` environment variable, falling back to
        ``"text-embedding-3-small"`` if not set.
    api_key:
        OpenAI API key.  Defaults to the ``OPENAI_API_KEY`` environment
        variable (which the openai library also reads automatically).
    """

    DEFAULT_MODEL = "text-embedding-3-small"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        # Lazy import so the rest of the module is importable without
        # openai installed (e.g. in environments that only run lexical tests).
        import openai  # noqa: F401 — validate it is installed

        self._model = (
            model
            or os.environ.get("OPENAI_EMBEDDING_MODEL")
            or self.DEFAULT_MODEL
        )
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key not found.  "
                "Set the OPENAI_API_KEY environment variable or pass api_key=."
            )

    def _client(self):  # type: ignore[return]
        import openai
        return openai.OpenAI(api_key=self._api_key)

    def embed_documents(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed a batch of texts using the OpenAI embeddings API."""
        if not texts:
            raise ValueError("texts must not be empty")
        response = self._client().embeddings.create(
            model=self._model,
            input=texts,
        )
        vectors = np.array(
            [item.embedding for item in response.data],
            dtype=np.float32,
        )
        return vectors  # shape (n, dim)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Embed a single query string."""
        result = self.embed_documents([text])
        return result[0]  # shape (dim,)


# ---------------------------------------------------------------------------
# Deterministic fake for tests
# ---------------------------------------------------------------------------

class FakeEmbeddingProvider:
    """
    Deterministic embedding provider for testing.

    Produces reproducible embeddings using a simple hash-based approach
    so that:
    - No API key is required.
    - No network access is needed.
    - Results are identical across runs.

    The embeddings are low-dimensional (``dim=32`` by default) and have no
    semantic meaning.  They are sufficient for testing index mechanics,
    deduplication, retrieval API contracts, and score ordering consistency.

    Two texts that share more characters will have closer embeddings — this
    provides just enough structure for testing without actual semantics.

    Parameters
    ----------
    dim:
        Embedding dimensionality.  Default 32.
    seed:
        Random seed for reproducibility.  Default 42.
    """

    def __init__(self, dim: int = 32, seed: int = 42) -> None:
        self._dim = dim
        self._rng = np.random.default_rng(seed)

    def _text_to_vector(self, text: str) -> NDArray[np.float32]:
        """
        Map a text to a reproducible unit vector.

        Uses a seeded PRNG keyed on the text content so identical texts
        always produce identical vectors.  Uses hashlib (not Python's
        built-in hash()) to avoid Python's PYTHONHASHSEED randomization.
        """
        import hashlib
        # MD5 is used here purely for deterministic seeding — not security.
        digest = hashlib.md5(text.encode("utf-8"), usedforsecurity=False).digest()
        # Use first 4 bytes as a 32-bit seed
        text_seed = int.from_bytes(digest[:4], "big")
        rng = np.random.default_rng(text_seed)
        vec = rng.standard_normal(self._dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            vec = np.ones(self._dim, dtype=np.float32)
            norm = np.linalg.norm(vec)
        return vec / norm

    def embed_documents(self, texts: list[str]) -> NDArray[np.float32]:
        if not texts:
            raise ValueError("texts must not be empty")
        return np.stack([self._text_to_vector(t) for t in texts])

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return self._text_to_vector(text)
