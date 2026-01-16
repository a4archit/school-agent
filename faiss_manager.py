
# faiss_manager.py
"""
FaissManager
============

A focused utility class for managing multiple FAISS vector stores using
LangChain and OpenAI embeddings. This module intentionally contains **no CLI**.

Key capabilities:
- Create/load/save/delete named FAISS stores under a root directory.
- Add texts/documents (with optional metadata and IDs).
- Delete items by IDs.
- Similarity search (with or without scores) and MMR search.
- List, rename, and merge stores.
- Inspect store statistics.

Requirements:
- langchain, langchain-community, langchain-openai
- faiss-cpu
- openai-compatible environment variables (e.g., OPENAI_API_KEY) or explicit key.

Notes:
- By default, embeddings are L2-normalized to approximate cosine similarity.
- Persisted stores are saved under `<root_dir>/<store_name>/`.
- Loading uses `allow_dangerous_deserialization=True` as required by FAISS serialization.

Example:
```python
from faiss_manager import FaissManager
fm = FaissManager(root_dir="./vectorstores", openai_api_key="sk-...")
fm.create_store("my_kb", texts=["hello world"], metadatas=[{"source":"init"}])
results = fm.similarity_search("my_kb", "hello")
```
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings

# `Document` import varies across LangChain versions; try the modern core first.
try:
    from langchain_core.documents import Document
except ImportError:  # Fallback for older versions
    from langchain.schema import Document  # type: ignore


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())








class FaissManager:
    """
    Manage multiple FAISS vector stores with OpenAI embeddings via LangChain.

    Parameters
    ----------
    root_dir : str | os.PathLike
        Directory under which all stores are persisted (one subdir per store).
    embedding_model : str, optional
        OpenAI embedding model name, by default "text-embedding-3-small".
        (Examples: "text-embedding-3-small", "text-embedding-3-large")
    openai_api_key : Optional[str], optional
        Explicit OpenAI API key; if not provided, the SDK will use env vars.
    embedding_kwargs : Optional[Dict[str, Any]], optional
        Extra kwargs passed to `OpenAIEmbeddings(...)` constructor.
        Useful for Azure/OpenAI params like `openai_api_base`, `openai_api_type`, etc.
    normalize_L2 : bool, optional
        If True (default), embeddings are L2-normalized so inner product ~ cosine.

    Persistence Layout
    ------------------
    <root_dir>/
        manifest.json             # optional registry metadata
        <store_name>/
            index.faiss
            index.pkl

    Warning
    -------
    `load_local(..., allow_dangerous_deserialization=True)` is required by FAISS
    persistence. Only load from trusted locations.
    """

    MANIFEST_FILENAME = "manifest.json"

    def __init__(
        self,
        root_dir: Union[str, os.PathLike],
        embedding_model: str = "text-embedding-ada-002",
        openai_api_key: Optional[str] = None,
        embedding_kwargs: Optional[Dict[str, Any]] = None,
        normalize_L2: bool = True,
    ) -> None:
        
        self.root_dir = Path(root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_model = embedding_model
        self.normalize_L2 = normalize_L2

        # Initialize the OpenAI embeddings (kept as a singleton for this manager)
        embedding_kwargs = embedding_kwargs or {}
        if openai_api_key:
            embedding_kwargs.setdefault("api_key", openai_api_key)

        # Create the embedder up front (explicit, single construction)
        self._embeddings = AzureOpenAIEmbeddings(
            model=self.embedding_model,
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("OPENAI_API_VERSION"),
            **embedding_kwargs,
        )

        # Initialize manifest for simple bookkeeping
        self._manifest_path = self.root_dir / self.MANIFEST_FILENAME
        self._manifest: Dict[str, Any] = self._load_or_init_manifest()


    # ---------------------------------------------------------------------
    # Internal utilities
    # ---------------------------------------------------------------------
    def _load_or_init_manifest(self) -> Dict[str, Any]:
        """Load the manifest file or initialize a new one if absent."""
        if self._manifest_path.exists():
            try:
                with open(self._manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except Exception as e:
                logger.warning("Failed to load manifest.json: %s", e)

        manifest = {
            "created_at": datetime.utcnow().isoformat(),
            "stores": {},  # name -> metadata
            "embedding_model": self.embedding_model,
            "normalize_L2": self.normalize_L2,
        }
        self._write_manifest(manifest)
        return manifest


    def _write_manifest(self, manifest: Dict[str, Any]) -> None:
        """Persist the manifest to disk."""
        tmp_path = self._manifest_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        os.replace(tmp_path, self._manifest_path)


    def _store_dir(self, name: str) -> Path:
        """Physical directory path for a given store name."""
        if not name or any(ch in name for ch in r"\/:*?\"<>|"):
            raise ValueError(f"Invalid store name: {name!r}")
        return self.root_dir / name


    def _ensure_store_exists(self, name: str) -> Path:
        """Raise if the store directory does not exist; return its path if it does."""
        path = self._store_dir(name)
        if not path.exists():
            raise FileNotFoundError(f"Vector store {name!r} not found at {path}")
        return path


    def _save_store(self, name: str, vs: FAISS) -> None:
        """Persist a FAISS vector store to its directory and update manifest."""
        path = self._store_dir(name)
        path.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(path))

        # Update manifest store metadata
        self._manifest["stores"].setdefault(name, {})
        self._manifest["stores"][name].update(
            {
                "updated_at": datetime.utcnow().isoformat(),
                "embedding_model": self.embedding_model,
                "normalize_L2": self.normalize_L2,
                "doc_count": self._safe_doc_count(vs),
                "dimension": self._safe_dimension(vs),
            }
        )
        self._write_manifest(self._manifest)

    @staticmethod
    def _safe_doc_count(vs: FAISS) -> int:
        """Best-effort extraction of document count from a FAISS vector store."""
        try:
            return len(getattr(vs, "index_to_docstore_id", {}) or {})
        except Exception:
            return -1

    @staticmethod
    def _safe_dimension(vs: FAISS) -> int:
        """Best-effort extraction of embedding dimension from FAISS index."""
        try:
            # FAISS index exposes `d` for dimension.
            return int(getattr(vs.index, "d", -1))
        except Exception:
            return -1



    # ---------------------------------------------------------------------
    # Public API: Store lifecycle
    # ---------------------------------------------------------------------

    def list_stores(self) -> List[str]:
        """Return the list of known vector store names (from manifest)."""
        return sorted(self._manifest.get("stores", {}).keys())


    def store_exists(self, name: str) -> bool:
        """Check whether a store exists on disk (by directory presence)."""
        return self._store_dir(name).exists()


    def create_store(
        self,
        name: str,
        *,
        documents: Optional[Sequence[Document]] = None,
        texts: Optional[Sequence[str]] = None,
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        ids: Optional[Sequence[str]] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Create a new FAISS store (optionally populated) and persist it.

        Provide one of:
        - `documents`: a list of LangChain `Document` objects, or
        - `texts`: a list of strings (optionally with `metadatas` and `ids`).

        Parameters
        ----------
        name : str
            Unique store name (subdirectory).
        documents : Optional[Sequence[Document]], optional
            Documents to index initially.
        texts : Optional[Sequence[str]], optional
            Texts to index initially (alternative to `documents`).
        metadatas : Optional[Sequence[Dict[str, Any]]], optional
            Metadata aligned with `texts`.
        ids : Optional[Sequence[str]], optional
            Custom IDs aligned with `texts` or `documents`.
        overwrite : bool, optional
            If True, replace an existing store directory; otherwise error.

        Raises
        ------
        ValueError
            If both `documents` and `texts` are provided or neither is provided (when initial data is required).
        FileExistsError
            If store exists and `overwrite=False`.
        """
        store_dir = self._store_dir(name)
        if store_dir.exists():
            if not overwrite:
                raise FileExistsError(f"Store {name!r} already exists at {store_dir}.")
            shutil.rmtree(store_dir)

        # Create an empty store first if no initial data provided
        vs: Optional[FAISS] = None

        if documents is not None and texts is not None:
            raise ValueError("Provide either `documents` or `texts`, not both.")

        if documents:
            vs = FAISS.from_documents(
                documents=list(documents),
                embedding=self._embeddings,
                normalize_L2=self.normalize_L2,
            )
        elif texts:
            vs = FAISS.from_texts(
                texts=list(texts),
                embedding=self._embeddings,
                metadatas=list(metadatas) if metadatas is not None else None,
                ids=list(ids) if ids is not None else None,
                normalize_L2=self.normalize_L2,
            )
        else:
            # Start an empty index by adding nothing; LangChain doesn't expose an "empty" ctor.
            # Workaround: create from a single dummy, then delete it right away.
            vs = FAISS.from_texts(
                texts=["__faiss_manager_empty_placeholder__"],
                embedding=self._embeddings,
                # metadatas=[{"_placeholder": True}],
                ids=["__placeholder__"],
                normalize_L2=self.normalize_L2,
            )
            vs.delete(ids=["__placeholder__"])

        self._save_store(name, vs)

        # Update manifest creation info
        self._manifest["stores"].setdefault(name, {})
        self._manifest["stores"][name].setdefault(
            "created_at", datetime.utcnow().isoformat()
        )
        self._write_manifest(self._manifest)



    def load_store(self, name: str) -> FAISS:
        """
        Load a FAISS store from disk.

        Parameters
        ----------
        name : str
            Store name.

        Returns
        -------
        FAISS
            The LangChain FAISS vector store instance.
        """
        path = self._ensure_store_exists(name)
        return FAISS.load_local(
            folder_path=str(path),
            embeddings=self._embeddings,
            allow_dangerous_deserialization=True,
        )

    def delete_store(self, name: str) -> None:
        """
        Permanently delete a store directory and remove it from the manifest.

        Parameters
        ----------
        name : str
            Store name to delete.
        """
        path = self._ensure_store_exists(name)
        shutil.rmtree(path, ignore_errors=True)
        if self._manifest["stores"].pop(name, None) is not None:
            self._write_manifest(self._manifest)


    def rename_store(self, old_name: str, new_name: str, *, overwrite: bool = False) -> None:
        """
        Rename a store directory and update the manifest.

        Parameters
        ----------
        old_name : str
            Existing store name.
        new_name : str
            New store name.
        overwrite : bool, optional
            If True, overwrite any existing `new_name` store; otherwise error.
        """
        src = self._ensure_store_exists(old_name)
        dst = self._store_dir(new_name)
        if dst.exists():
            if not overwrite:
                raise FileExistsError(f"Target store {new_name!r} already exists.")
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))

        # Move manifest entry
        meta = self._manifest["stores"].pop(old_name, {})
        self._manifest["stores"][new_name] = meta
        self._manifest["stores"][new_name]["updated_at"] = datetime.utcnow().isoformat()
        self._write_manifest(self._manifest)


    # ---------------------------------------------------------------------
    # Public API: Add / Delete content
    # ---------------------------------------------------------------------
    def add_documents(
        self,
        name: str,
        documents: Sequence[Document],
        ids: Optional[Sequence[str]] = None,
        save: bool = True,
    ) -> int:
        """
        Add LangChain `Document` objects to a store.

        Parameters
        ----------
        name : str
            Store name.
        documents : Sequence[Document]
            Documents to add.
        ids : Optional[Sequence[str]], optional
            Optional IDs aligned with documents.
        save : bool, optional
            Persist changes to disk immediately, by default True.

        Returns
        -------
        int
            Number of added items.
        """
        if not documents:
            return 0
        vs = self.load_store(name)
        vs.add_documents(list(documents), ids=list(ids) if ids is not None else None)
        if save:
            self._save_store(name, vs)
        return len(documents)


    def add_texts(
        self,
        name: str,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        ids: Optional[Sequence[str]] = None,
        save: bool = True,
    ) -> int:
        """
        Add plain texts (with optional metadata/IDs) to a store.

        Parameters
        ----------
        name : str
            Store name.
        texts : Sequence[str]
            Texts to add.
        metadatas : Optional[Sequence[Dict[str, Any]]], optional
            Metadata aligned with texts.
        ids : Optional[Sequence[str]], optional
            Optional IDs aligned with texts.
        save : bool, optional
            Persist changes to disk immediately, by default True.

        Returns
        -------
        int
            Number of added items.
        """
        if not texts:
            return 0
        vs = self.load_store(name)
        vs.add_texts(
            list(texts),
            metadatas=list(metadatas) if metadatas is not None else None,
            ids=list(ids) if ids is not None else None,
        )
        if save:
            self._save_store(name, vs)
        return len(texts)


    def delete_documents(self, name: str, ids: Sequence[str], save: bool = True) -> int:
        """
        Delete items by IDs from a store.

        Parameters
        ----------
        name : str
            Store name.
        ids : Sequence[str]
            IDs to delete.
        save : bool, optional
            Persist changes to disk immediately, by default True.

        Returns
        -------
        int
            Number of IDs requested for deletion (best-effort).
        """
        if not ids:
            return 0
        vs = self.load_store(name)
        vs.delete(list(ids))
        if save:
            self._save_store(name, vs)
        return len(ids)


    # ---------------------------------------------------------------------
    # Public API: Search
    # ---------------------------------------------------------------------
    def similarity_search(
        self,
        name: str,
        query: str,
        k: int = 4,
        *,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Perform a text similarity search and return documents only.

        Parameters
        ----------
        name : str
            Store name.
        query : str
            Natural language query.
        k : int, optional
            Number of results to return, by default 4.
        filter : Optional[Dict[str, Any]], optional
            LangChain metadata filter, if your store uses metadata-aware retrievers.

        Returns
        -------
        List[Document]
        """
        vs = self.load_store(name)
        return vs.similarity_search(query=query, k=k, filter=filter)



    def similarity_search_with_scores(
        self,
        name: str,
        query: str,
        k: int = 4,
        *,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Perform a text similarity search and return (Document, score) tuples.

        Parameters
        ----------
        name : str
            Store name.
        query : str
            Natural language query.
        k : int, optional
            Number of results to return, by default 4.
        filter : Optional[Dict[str, Any]], optional
            LangChain metadata filter, if used.

        Returns
        -------
        List[Tuple[Document, float]]
            Lower scores indicate closer matches if using L2; with normalized
            vectors, the internal metric corresponds to cosine similarity.
        """
        vs = self.load_store(name)
        return vs.similarity_search_with_score(query=query, k=k, filter=filter)



    def similarity_search_by_vector(
        self,
        name: str,
        embedding: Sequence[float],
        k: int = 4,
        *,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Perform similarity search using a precomputed embedding vector.

        Parameters
        ----------
        name : str
            Store name.
        embedding : Sequence[float]
            Precomputed embedding.
        k : int, optional
            Number of results to return, by default 4.
        filter : Optional[Dict[str, Any]], optional
            LangChain metadata filter, if used.

        Returns
        -------
        List[Document]
        """
        vs = self.load_store(name)
        return vs.similarity_search_by_vector(embedding=embedding, k=k, filter=filter)

    def mmr_search(
        self,
        name: str,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        *,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Maximal Marginal Relevance (MMR) search for diverse, relevant results.

        Parameters
        ----------
        name : str
            Store name.
        query : str
            Query string.
        k : int, optional
            Number of results to return, by default 4.
        fetch_k : int, optional
            Candidate pool size before diversification, by default 20.
        lambda_mult : float, optional
            Diversity vs. relevance tradeoff in [0,1], by default 0.5.
        filter : Optional[Dict[str, Any]], optional
            LangChain metadata filter, if used.

        Returns
        -------
        List[Document]
        """
        vs = self.load_store(name)
        return vs.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
        )
    


    # ---------------------------------------------------------------------
    # Public API: Merge / Stats
    # ---------------------------------------------------------------------
    def merge_stores(
        self,
        target_name: str,
        source_names: Sequence[str],
        *,
        overwrite: bool = False,
        save: bool = True,
    ) -> None:
        """
        Merge multiple existing stores into a single target store.

        Parameters
        ----------
        target_name : str
            Name of the resulting/target store.
        source_names : Sequence[str]
            Names of stores to merge into the target.
        overwrite : bool, optional
            If True, allows replacing an existing target store; otherwise error.
        save : bool, optional
            Persist merged target to disk, by default True.
        """
        if not source_names:
            return

        # Prepare/initialize target
        if self.store_exists(target_name):
            if not overwrite:
                raise FileExistsError(
                    f"Target store {target_name!r} already exists. Set overwrite=True to replace."
                )
            # Load and clear by recreating an empty store
            self.delete_store(target_name)

        # Start target as a copy of the first source
        first = source_names[0]
        target_vs = self.load_store(first)

        # Important: Create a deep copy by saving+loading to isolate indices (safest route)
        tmp_target = f"__merge_tmp_{datetime.utcnow().timestamp()}__"
        tmp_dir = self._store_dir(tmp_target)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        target_vs.save_local(str(tmp_dir))
        target_vs = FAISS.load_local(
            folder_path=str(tmp_dir),
            embeddings=self._embeddings,
            allow_dangerous_deserialization=True,
        )

        # Merge in the rest
        for src in source_names[1:]:
            src_vs = self.load_store(src)
            target_vs.merge_from(src_vs)

        # Save as the official target
        if save:
            self._save_store(target_name, target_vs)

        # Cleanup tmp
        shutil.rmtree(tmp_dir, ignore_errors=True)

        # Manifest bookkeeping
        self._manifest["stores"].setdefault(target_name, {})
        self._manifest["stores"][target_name].setdefault(
            "created_at", datetime.utcnow().isoformat()
        )
        self._manifest["stores"][target_name]["updated_at"] = datetime.utcnow().isoformat()
        self._manifest["stores"][target_name]["doc_count"] = self._safe_doc_count(target_vs)
        self._manifest["stores"][target_name]["dimension"] = self._safe_dimension(target_vs)
        self._write_manifest(self._manifest)



    def store_stats(self, name: str) -> Dict[str, Any]:
        """
        Return basic statistics about a store.

        Parameters
        ----------
        name : str
            Store name.

        Returns
        -------
        Dict[str, Any]
            Keys: name, path, doc_count, dimension, embedding_model, normalize_L2, updated_at
        """
        vs = self.load_store(name)
        meta = self._manifest.get("stores", {}).get(name, {})
        return {
            "name": name,
            "path": str(self._store_dir(name)),
            "doc_count": self._safe_doc_count(vs),
            "dimension": self._safe_dimension(vs),
            "embedding_model": meta.get("embedding_model", self.embedding_model),
            "normalize_L2": meta.get("normalize_L2", self.normalize_L2),
            "updated_at": meta.get("updated_at"),
            "created_at": meta.get("created_at"),
        }

    # ---------------------------------------------------------------------
    # Convenience helpers
    # ---------------------------------------------------------------------
    def embed_query(self, query: str) -> List[float]:
        """
        Compute an embedding for a query string using the configured OpenAI model.

        Parameters
        ----------
        query : str

        Returns
        -------
        List[float]
        """
        return self._embeddings.embed_query(query)

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Compute embeddings for multiple texts.

        Parameters
        ----------
        texts : Sequence[str]

        Returns
        -------
        List[List[float]]
        """
        return self._embeddings.embed_documents(list(texts))
    








if __name__ == "__main__":

    from dotenv import load_dotenv

    load_dotenv()

    fm = FaissManager(root_dir="LOCAL")

    fm.create_store(name="local_vec_store", overwrite=True)



