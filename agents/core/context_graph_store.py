"""
ContextGraphStore
=================
A LangGraph BaseCheckpointSaver backed by the CSIP Context Graph (Neo4j).

thread_id convention
--------------------
    thread_id = session_id   (always — no compound IDs)

The ContextGraphStore uses the CG's HAS_FOLLOW_UP chain and two navigation
properties (chainDepth, rootSessionId) set by link_follow_up_session() to
reconstruct the full conversation history without loading everything at once.

Batched retrieval algorithm
---------------------------
Given the current session at chainDepth N:

    N = 0  (fresh session, no prior): return None — no checkpoint to restore

    N <= 11 (short chain):
        Load sessions 0..N-1 verbatim and return as restored messages.

    N > 11 (long chain):
        Part A — Session at chainDepth 0 (root) verbatim.
        Part B — Sessions at chainDepth 1..N-11, processed in sliding
                 batches of 10.  Each batch is loaded, converted to text,
                 prepended with the prior summary, then semantically
                 compressed.  This keeps memory bounded to ~10 sessions
                 at any point regardless of chain length.
        Part C — Sessions at chainDepth N-10..N-1 verbatim (most recent
                 10 prior sessions, uncompressed for exact entity IDs).

    Final restored messages:
        Part A messages + [SystemMessage(SummaryM)] + Part C messages

put() stores ONLY the current session's messages — no prior context.
This keeps each Session node compact regardless of chain depth.

Neo4j index (add to schema migration):
    CREATE INDEX session_chain IF NOT EXISTS
    FOR (s:Session) ON (s.rootSessionId, s.chainDepth)
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Iterator, List, Optional, Sequence, Tuple

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    messages_from_dict,
    messages_to_dict,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    empty_checkpoint,
)

logger = logging.getLogger(__name__)

_CG_MESSAGES_KEY  = "conversationMessages"
_BATCH_SIZE       = 10   # sessions per compression batch
_VERBATIM_RECENT  = 10   # most recent prior sessions kept verbatim


class ContextGraphStore(BaseCheckpointSaver):
    """
    LangGraph BaseCheckpointSaver backed by the CSIP Context Graph (Neo4j).

    Uses chainDepth and rootSessionId properties on Session nodes — set by
    link_follow_up_session() — to perform efficient batched retrieval of
    conversation history without full graph traversal.
    """

    def __init__(self) -> None:
        from agents.core.context_graph import get_context_graph_manager
        self._cg = get_context_graph_manager()
        super().__init__()

    # =========================================================================
    # BaseCheckpointSaver — sync interface
    # =========================================================================

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """
        Called by LangGraph before graph.invoke() when a thread_id is present.

        Reads chainDepth from the current session to determine how to retrieve
        prior conversation history, then applies the batched reconstruction
        algorithm.  Returns None for a fresh session (chainDepth = 0).
        """
        session_id = config.get("configurable", {}).get("thread_id", "")
        if not session_id:
            return None

        try:
            messages = self._reconstruct_messages(session_id)
            if not messages:
                return None

            checkpoint = empty_checkpoint()
            checkpoint["channel_values"] = {"messages": messages}
            checkpoint["id"] = session_id

            return CheckpointTuple(
                config=config,
                checkpoint=checkpoint,
                metadata=CheckpointMetadata(source="input", step=0, writes={}),
                parent_config=None,
                pending_writes=[],
            )
        except Exception as exc:
            logger.warning(
                "ContextGraphStore.get_tuple(%s) failed (returning None): %s",
                session_id, exc,
            )
            return None

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """
        Called automatically by LangGraph at the end of graph.invoke().

        Writes ONLY the current session's messages to its Session node.
        Prior session context is never stored here — it is reconstructed
        on demand by get_tuple() via the HAS_FOLLOW_UP chain.
        """
        session_id = config.get("configurable", {}).get("thread_id", "")
        if not session_id:
            return config

        try:
            channel_values = checkpoint.get("channel_values", {})
            raw_messages   = channel_values.get("messages", [])
            tool_results   = channel_values.get("tool_results", {})

            # Keep only the CURRENT session's messages.
            # raw_messages may contain prior context restored by get_tuple() —
            # we must filter it out to keep each Session node compact.
            current_messages = _extract_current_session_messages(raw_messages)

            # Inject SystemMessages for tool results (structured entity data)
            if tool_results:
                sys_msgs = _tool_results_to_system_messages(tool_results)
                if current_messages and isinstance(current_messages[-1], AIMessage):
                    current_messages = (
                        current_messages[:-1] + sys_msgs + [current_messages[-1]]
                    )
                else:
                    current_messages.extend(sys_msgs)

            if current_messages:
                self._save_messages(session_id, current_messages)

        except Exception as exc:
            logger.warning(
                "ContextGraphStore.put(%s) failed (non-fatal): %s",
                session_id, exc,
            )

        checkpoint_id = checkpoint.get("id", str(uuid.uuid4()))
        return {
            **config,
            "configurable": {
                **config.get("configurable", {}),
                "checkpoint_id": checkpoint_id,
                "checkpoint_ns": config.get("configurable", {}).get("checkpoint_ns", ""),
            },
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        pass

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        return iter([])

    # =========================================================================
    # Async variants
    # =========================================================================

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_tuple, config)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.put, config, checkpoint, metadata, new_versions
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        pass

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        return
        yield

    # =========================================================================
    # Batched history reconstruction
    # =========================================================================

    def _reconstruct_messages(self, session_id: str) -> List[BaseMessage]:
        """
        Reconstruct prior conversation history using the chainDepth algorithm.

        Reads chainDepth and rootSessionId from the current session node,
        then applies the appropriate retrieval strategy:
            N = 0  → fresh session, return []
            N ≤ 11 → verbatim load of sessions 0..N-1
            N > 11 → Part A (root verbatim) + Part B (compressed middle)
                     + Part C (recent 10 verbatim)
        """
        # Step 1: get current session metadata
        meta = self._get_session_meta(session_id)
        if not meta:
            return []

        chain_depth     = meta.get("chainDepth", 0) or 0
        root_session_id = meta.get("rootSessionId", "")

        if chain_depth == 0:
            # Fresh session — no prior context
            return []

        N = chain_depth  # depth of current session = number of prior sessions

        if N <= _VERBATIM_RECENT + 1:
            # Short chain — load all prior sessions verbatim
            return self._load_sessions_in_range(
                root_session_id, depth_start=0, depth_end=N - 1
            )

        # Long chain — three-part reconstruction
        # Part A: root session verbatim (chainDepth = 0)
        part_a = self._load_sessions_in_range(root_session_id, 0, 0)

        # Part B: compressed middle (chainDepth 1 .. N-_VERBATIM_RECENT-1)
        middle_end = N - _VERBATIM_RECENT - 1
        part_b_summary = self._compress_middle(root_session_id, 1, middle_end)

        # Part C: most recent _VERBATIM_RECENT prior sessions verbatim
        part_c = self._load_sessions_in_range(
            root_session_id,
            depth_start=N - _VERBATIM_RECENT,
            depth_end=N - 1,
        )

        result = list(part_a)
        if part_b_summary:
            result.append(
                SystemMessage(content=f"[Conversation summary]: {part_b_summary}")
            )
        result.extend(part_c)
        return result

    def _compress_middle(
        self,
        root_session_id: str,
        depth_start: int,
        depth_end: int,
    ) -> str:
        """
        Compress sessions between depth_start and depth_end in batches of 10.

        Each batch is loaded, converted to text, prepended with the rolling
        summary from prior batches, then semantically compressed.  Memory
        at any point: one batch of messages + the running summary string.
        """
        try:
            from agents.core.context_compressor import get_semantic_compressor
            compressor = get_semantic_compressor()
        except Exception as exc:
            logger.warning(
                "ContextGraphStore: semantic compressor unavailable: %s", exc
            )
            return ""

        summary = ""
        batch_start = depth_start

        while batch_start <= depth_end:
            batch_end = min(batch_start + _BATCH_SIZE - 1, depth_end)
            batch_msgs = self._load_sessions_in_range(
                root_session_id, batch_start, batch_end
            )
            if not batch_msgs:
                batch_start = batch_end + 1
                continue

            batch_text = _messages_to_text(batch_msgs)
            if summary:
                # Prepend prior summary so compressor retains cross-batch refs
                batch_text = summary + "\n\n" + batch_text

            try:
                summary = compressor.compress_text(batch_text)
            except Exception as exc:
                logger.warning(
                    "ContextGraphStore: compression batch %d-%d failed: %s",
                    batch_start, batch_end, exc,
                )
                # Fall back to raw text for this batch — don't lose content
                summary = batch_text

            logger.debug(
                "ContextGraphStore: compressed batch depth %d-%d "
                "(summary len=%d)",
                batch_start, batch_end, len(summary),
            )
            batch_start = batch_end + 1

        return summary

    # =========================================================================
    # CG query helpers
    # =========================================================================

    def _get_session_meta(self, session_id: str) -> Optional[dict]:
        """Fetch chainDepth and rootSessionId for a session."""
        try:
            result = self._cg.cg_data_access.conn.execute_query(
                """
                MATCH (s:Session {sessionId: $sessionId})
                RETURN s.chainDepth    AS chainDepth,
                       s.rootSessionId AS rootSessionId
                """,
                {"sessionId": session_id},
            )
            return result[0] if result else None
        except Exception as exc:
            logger.warning("_get_session_meta(%s) failed: %s", session_id, exc)
            return None

    def _load_sessions_in_range(
        self,
        root_session_id: str,
        depth_start: int,
        depth_end: int,
    ) -> List[BaseMessage]:
        """
        Load and deserialise messages from all sessions in a chainDepth range.

        Ordered by chainDepth ASC so earlier sessions appear first.
        Uses the composite index on (rootSessionId, chainDepth).
        """
        try:
            # Handle root session (chainDepth=0) which has rootSessionId=null
            if depth_start == 0 and depth_end == 0:
                result = self._cg.cg_data_access.conn.execute_query(
                    f"""
                    MATCH (s:Session {{sessionId: $rootSessionId}})
                    WHERE s.{_CG_MESSAGES_KEY} IS NOT NULL
                    RETURN s.{_CG_MESSAGES_KEY} AS messages
                    """,
                    {"rootSessionId": root_session_id},
                )
            else:
                result = self._cg.cg_data_access.conn.execute_query(
                    f"""
                    MATCH (s:Session {{rootSessionId: $rootSessionId}})
                    WHERE s.chainDepth >= $depthStart
                      AND s.chainDepth <= $depthEnd
                      AND s.{_CG_MESSAGES_KEY} IS NOT NULL
                    RETURN s.{_CG_MESSAGES_KEY} AS messages
                    ORDER BY s.chainDepth ASC
                    """,
                    {
                        "rootSessionId": root_session_id,
                        "depthStart":    depth_start,
                        "depthEnd":      depth_end,
                    },
                )
        except Exception as exc:
            logger.warning(
                "_load_sessions_in_range(%s, %d, %d) failed: %s",
                root_session_id, depth_start, depth_end, exc,
            )
            return []

        messages: List[BaseMessage] = []
        for row in result:
            raw = row.get("messages", "")
            if raw:
                try:
                    messages.extend(messages_from_dict(json.loads(raw)))
                except Exception as exc:
                    logger.warning(
                        "Failed to deserialise messages for depth %d-%d: %s",
                        depth_start, depth_end, exc,
                    )
        return messages

    def _save_messages(self, session_id: str, messages: List[BaseMessage]) -> None:
        """Serialise messages to JSON and write to CG Session node."""
        try:
            serialised = json.dumps(messages_to_dict(messages))
            self._cg.cg_data_access.conn.execute_query(
                f"""
                MATCH (s:Session {{sessionId: $sessionId}})
                SET s.{_CG_MESSAGES_KEY} = $messages
                """,
                {"sessionId": session_id, "messages": serialised},
            )
        except Exception as exc:
            logger.warning("_save_messages(%s) failed: %s", session_id, exc)


# =============================================================================
# Module-level helpers
# =============================================================================

def _extract_current_session_messages(
    raw_messages: List[Any],
) -> List[BaseMessage]:
    """
    Extract only the current session's messages from the full state list.

    state["messages"] after checkpointer restore contains:
        [prior session messages...] + [current HumanMessage]

    After the graph runs it contains:
        [prior session messages...] + [current HumanMessage] + [current AIMessage]

    We only want the current session's own messages — the last HumanMessage
    onward — to avoid storing prior context redundantly on the new session node.
    """
    msgs = [m for m in raw_messages if isinstance(m, BaseMessage)]
    if not msgs:
        return []

    # Find the last HumanMessage — that is where the current session starts
    last_human_idx = None
    for i, msg in enumerate(msgs):
        if isinstance(msg, HumanMessage):
            last_human_idx = i

    if last_human_idx is None:
        return msgs

    return msgs[last_human_idx:]


def _tool_results_to_system_messages(tool_results: dict) -> List[SystemMessage]:
    """Convert tool_results to SystemMessages carrying structured entity data."""
    msgs = []
    for worker_name, result in tool_results.items():
        if not isinstance(result, dict):
            continue
        content = result.get("tool_raw_output") or result.get("output", "")
        if content:
            msgs.append(
                SystemMessage(
                    content=f"[{worker_name} result]: {content}",
                    name=worker_name,
                )
            )
    return msgs


def _messages_to_text(messages: List[BaseMessage]) -> str:
    """Convert a message list to a plain text block for compression."""
    parts = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            parts.append(f"CSR: {msg.content}")
        elif isinstance(msg, AIMessage):
            parts.append(f"System: {msg.content}")
        elif isinstance(msg, SystemMessage):
            parts.append(f"Data: {msg.content}")
    return "\n".join(parts)


# =============================================================================
# Singleton
# =============================================================================

_store: Optional[ContextGraphStore] = None


def get_context_graph_store() -> ContextGraphStore:
    """Return the singleton ContextGraphStore instance."""
    global _store
    if _store is None:
        _store = ContextGraphStore()
        logger.info(
            "ContextGraphStore initialised (Neo4j-backed LangGraph CheckpointSaver)"
        )
    return _store
