"""
Chroma Vector Database Data Access Layer

Provides semantic search interfaces for all 6 Chroma collections loaded by
data/load_sample_data.py. Each method maps to a specific collection and
returns structured results with documents, metadata, and relevance scores.

Collections (from load_sample_data.py):
  1. policies               - Policy documents (plan details, premiums, deductibles)
  2. procedures             - CPT procedure codes and descriptions
  3. diagnoses              - ICD-10 diagnosis codes and descriptions
  4. faqs                   - Frequently asked questions and answers
  5. clinical_guidelines    - Clinical guidelines and treatment protocols
  6. regulations            - Regulatory documents (collection created, data TBD)
"""

import logging
from typing import Any, Dict, List, Optional

from databases.connections import get_chroma, ChromaConnection

logger = logging.getLogger(__name__)


class ChromaVectorDataAccess:
    """
    Unified data-access layer for all Chroma vector collections.

    Every public method:
      - Accepts a natural-language query string.
      - Accepts optional metadata filters (``where``).
      - Returns a list of dicts with keys: id, document, metadata, distance.
    """

    # ------------------------------------------------------------------ #
    #  Collection name constants (must match load_sample_data.py exactly) #
    # ------------------------------------------------------------------ #
    POLICIES = "policies"
    PROCEDURES = "procedures"
    DIAGNOSES = "diagnoses"
    FAQS = "faqs"
    CLINICAL_GUIDELINES = "clinical_guidelines"
    REGULATIONS = "regulations"

    ALL_COLLECTIONS = [
        POLICIES, PROCEDURES, DIAGNOSES,
        FAQS, CLINICAL_GUIDELINES, REGULATIONS,
    ]

    def __init__(self, chroma_conn: Optional[ChromaConnection] = None):
        self._conn = chroma_conn or get_chroma()

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _query_collection(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run a semantic search against *collection_name* and return a
        normalised list of result dicts.

        Each result dict contains:
          - id       (str)   – document ID inside the collection
          - document (str)   – the matched text chunk
          - metadata (dict)  – metadata stored alongside the document
          - distance (float) – cosine distance (lower = more similar)
        """
        try:
            collection = self._conn.get_collection(collection_name)
        except Exception:
            logger.warning(
                "Collection '%s' not found; returning empty results.",
                collection_name,
            )
            return []

        query_kwargs: Dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where
        if where_document:
            query_kwargs["where_document"] = where_document

        try:
            raw = collection.query(**query_kwargs)
        except Exception as exc:
            logger.error(
                "Chroma query failed on '%s': %s", collection_name, exc
            )
            return []

        results: List[Dict[str, Any]] = []
        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            results.append({
                "id": doc_id,
                "document": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "distance": dists[i] if i < len(dists) else 1.0,
            })

        return results

    def _get_by_id(
        self, collection_name: str, doc_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a single document by its ID."""
        try:
            collection = self._conn.get_collection(collection_name)
            result = collection.get(ids=[doc_id], include=["documents", "metadatas"])
            if result["ids"]:
                return {
                    "id": result["ids"][0],
                    "document": result["documents"][0] if result["documents"] else "",
                    "metadata": result["metadatas"][0] if result["metadatas"] else {},
                }
        except Exception as exc:
            logger.error(
                "Chroma get-by-id failed on '%s' id='%s': %s",
                collection_name, doc_id, exc,
            )
        return None

    # ================================================================== #
    #  1. POLICIES COLLECTION                                             #
    # ================================================================== #

    def search_policies(
        self,
        query: str,
        n_results: int = 5,
        plan_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over policy documents.

        Args:
            query:     Natural-language query (e.g. "What is my deductible?")
            n_results: Maximum results to return.
            plan_type: Optional filter – HMO, PPO, EPO, POS.
            status:    Optional filter – ACTIVE, EXPIRED.

        Returns:
            List of result dicts (id, document, metadata, distance).
        """
        where = {}
        if plan_type:
            where["planType"] = plan_type
        if status:
            where["status"] = status
        return self._query_collection(
            self.POLICIES, query, n_results, where=where or None
        )

    def get_policy_by_id(self, policy_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific policy document by its policyId."""
        return self._get_by_id(self.POLICIES, policy_id)

    # ================================================================== #
    #  2. PROCEDURES COLLECTION                                           #
    # ================================================================== #

    def search_procedures(
        self,
        query: str,
        n_results: int = 5,
        category: Optional[str] = None,
        requires_prior_auth: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over CPT procedure documents.

        Args:
            query:               Natural-language query (e.g. "knee surgery").
            n_results:           Maximum results to return.
            category:            Optional filter – OFFICE_VISIT, DIAGNOSTIC,
                                 LABORATORY, SURGICAL, EMERGENCY.
            requires_prior_auth: Optional filter – True / False.

        Returns:
            List of result dicts.
        """
        where = {}
        if category:
            where["category"] = category
        if requires_prior_auth is not None:
            where["requiresPriorAuth"] = str(requires_prior_auth)
        return self._query_collection(
            self.PROCEDURES, query, n_results, where=where or None
        )

    def get_procedure_by_cpt(self, cpt_code: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific procedure document by CPT code."""
        return self._get_by_id(self.PROCEDURES, cpt_code)

    # ================================================================== #
    #  3. DIAGNOSES COLLECTION                                            #
    # ================================================================== #

    def search_diagnoses(
        self,
        query: str,
        n_results: int = 5,
        severity: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over ICD-10 diagnosis documents.

        Args:
            query:    Natural-language query (e.g. "high blood pressure").
            n_results: Maximum results to return.
            severity: Optional filter – MILD, MODERATE, SEVERE.
            category: Optional filter.

        Returns:
            List of result dicts.
        """
        where = {}
        if severity:
            where["severity"] = severity
        if category:
            where["category"] = category
        return self._query_collection(
            self.DIAGNOSES, query, n_results, where=where or None
        )

    def get_diagnosis_by_icd(self, icd_code: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific diagnosis document by ICD-10 code."""
        return self._get_by_id(self.DIAGNOSES, icd_code)

    # ================================================================== #
    #  4. FAQS COLLECTION                                                 #
    # ================================================================== #

    def search_faqs(
        self,
        query: str,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over FAQ documents.

        Args:
            query:     Natural-language question from user or CSR.
            n_results: Maximum results to return.

        Returns:
            List of result dicts.  Each document has format
            "Q: <question>\\nA: <answer>".
        """
        return self._query_collection(self.FAQS, query, n_results)

    def get_faq_by_id(self, faq_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific FAQ by its ID (e.g. 'faq_0')."""
        return self._get_by_id(self.FAQS, faq_id)

    # ================================================================== #
    #  5. CLINICAL GUIDELINES COLLECTION                                  #
    # ================================================================== #

    def search_clinical_guidelines(
        self,
        query: str,
        n_results: int = 5,
        guideline_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over clinical guidelines.

        Args:
            query:          Natural-language query (e.g. "knee arthroscopy
                            requirements").
            n_results:      Maximum results to return.
            guideline_type: Optional filter on metadata ``type`` field
                            (e.g. "clinical_guideline").

        Returns:
            List of result dicts.
        """
        where = {}
        if guideline_type:
            where["type"] = guideline_type
        return self._query_collection(
            self.CLINICAL_GUIDELINES, query, n_results,
            where=where or None,
        )

    def get_guideline_by_id(self, guideline_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific guideline by its ID (e.g. 'guideline_0')."""
        return self._get_by_id(self.CLINICAL_GUIDELINES, guideline_id)

    # ================================================================== #
    #  6. REGULATIONS COLLECTION                                          #
    # ================================================================== #

    def search_regulations(
        self,
        query: str,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over regulatory documents.

        Note: The ``regulations`` collection is created by the data loader
        but may not contain documents until regulatory content is loaded.

        Args:
            query:     Natural-language query.
            n_results: Maximum results to return.

        Returns:
            List of result dicts (may be empty if collection has no data).
        """
        return self._query_collection(self.REGULATIONS, query, n_results)

    # ================================================================== #
    #  Cross-collection search                                            #
    # ================================================================== #

    def search_all_collections(
        self,
        query: str,
        n_results_per_collection: int = 3,
        collections: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run the same semantic query across multiple (or all) collections
        and return results grouped by collection name.

        Args:
            query:                      Natural-language query.
            n_results_per_collection:   Max results per collection.
            collections:                Subset of collection names to search.
                                        Defaults to ALL_COLLECTIONS.

        Returns:
            Dict mapping collection name → list of result dicts.
        """
        target = collections or self.ALL_COLLECTIONS
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for coll_name in target:
            grouped[coll_name] = self._query_collection(
                coll_name, query, n_results_per_collection
            )
        return grouped

    # ================================================================== #
    #  Collection metadata / health                                       #
    # ================================================================== #

    def get_collection_stats(self) -> Dict[str, int]:
        """
        Return the document count for every known collection.

        Returns:
            Dict mapping collection name → document count.
        """
        stats: Dict[str, int] = {}
        for name in self.ALL_COLLECTIONS:
            try:
                coll = self._conn.get_collection(name)
                stats[name] = coll.count()
            except Exception:
                stats[name] = 0
        return stats

    def health_check(self) -> Dict[str, Any]:
        """
        Quick health check: verifies the Chroma client is reachable and
        returns per-collection document counts.

        Returns:
            Dict with ``status`` ("healthy" | "unhealthy"), ``collections``,
            and optional ``error`` message.
        """
        try:
            self._conn.connect()
            stats = self.get_collection_stats()
            return {
                "status": "healthy",
                "collections": stats,
                "total_documents": sum(stats.values()),
            }
        except Exception as exc:
            return {
                "status": "unhealthy",
                "error": str(exc),
            }


# ====================================================================== #
#  Module-level singleton                                                 #
# ====================================================================== #

_chroma_data_access: Optional[ChromaVectorDataAccess] = None


def get_chroma_data_access() -> ChromaVectorDataAccess:
    """Return (or create) the module-level ChromaVectorDataAccess singleton."""
    global _chroma_data_access
    if _chroma_data_access is None:
        _chroma_data_access = ChromaVectorDataAccess()
    return _chroma_data_access
