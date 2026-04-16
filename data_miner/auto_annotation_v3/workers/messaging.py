"""Redis Streams messaging layer for the auto_annotation_v3 pipeline.

Provides :class:`RedisMessageBroker` — a thin async wrapper around Redis Streams
consumer-groups used to pass :class:`~..contracts.StageMessage` envelopes between
the detect → evaluate → refine → done stages.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import redis.asyncio as aioredis

logger = logging.getLogger("data_miner.auto_annotation_v3.messaging")


# ---------------------------------------------------------------------------
# StreamConfig
# ---------------------------------------------------------------------------


@dataclass
class StreamConfig:
    """Maps stage names to Redis stream keys.

    Can be constructed directly or built from
    :class:`~..config.RedisConfig`.streams dict.
    """

    detect: str = "stream:detect"
    evaluate: str = "stream:evaluate"
    refine: str = "stream:refine"
    finalize: str = "stream:finalize"
    done: str = "stream:done"
    dead_letter: str = "stream:dead_letter"

    def stream_for_stage(self, stage: str) -> str:
        """Return the Redis stream key for *stage*, falling back to ``stream:<stage>``."""
        return getattr(self, stage, f"stream:{stage}")

    @classmethod
    def from_dict(cls, mapping: dict[str, str]) -> "StreamConfig":
        """Build from a ``{stage: stream_key}`` dict (e.g. from RedisConfig.streams)."""
        return cls(
            detect=mapping.get("detect", "stream:detect"),
            evaluate=mapping.get("evaluate", "stream:evaluate"),
            refine=mapping.get("refine", "stream:refine"),
            finalize=mapping.get("finalize", "stream:finalize"),
            done=mapping.get("done", "stream:done"),
            dead_letter=mapping.get("dead_letter", "stream:dead_letter"),
        )


# ---------------------------------------------------------------------------
# RedisMessageBroker
# ---------------------------------------------------------------------------


class RedisMessageBroker:
    """Async Redis Streams message broker for the annotation pipeline.

    Manages consumer groups, message submission, consumption, and
    dead-letter routing across the five pipeline streams.

    Parameters
    ----------
    redis_url:
        Redis connection URL, e.g. ``"redis://localhost:6379/0"``.
    consumer_group:
        Name of the consumer group shared by all workers.
    stream_config:
        Mapping of stage names to stream keys.  Defaults to
        :class:`StreamConfig` with ``"stream:<stage>"`` keys.
    """

    _ALL_STAGES: tuple[str, ...] = (
        "detect",
        "evaluate",
        "refine",
        "finalize",
        "done",
        "dead_letter",
    )

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        consumer_group: str = "aa_v3",
        stream_config: StreamConfig | None = None,
    ) -> None:
        self.redis_url = redis_url
        self.consumer_group = consumer_group
        self.streams = stream_config or StreamConfig()
        self._redis: aioredis.Redis | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the Redis connection and ensure all consumer groups exist."""
        self._redis = aioredis.from_url(self.redis_url, decode_responses=True)
        await self._ensure_consumer_groups()
        logger.info(
            "Connected to Redis at %s, consumer_group=%s",
            self.redis_url,
            self.consumer_group,
        )

    async def close(self) -> None:
        """Gracefully close the Redis connection."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
            logger.debug("Redis connection closed")

    async def __aenter__(self) -> "RedisMessageBroker":
        await self.connect()
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_redis(self) -> aioredis.Redis:
        if self._redis is None:
            raise RuntimeError(
                "RedisMessageBroker is not connected — call connect() first."
            )
        return self._redis

    async def _ensure_consumer_groups(self) -> None:
        """Create consumer groups for every stream (idempotent via BUSYGROUP guard).

        Uses ``id="$"`` so that a freshly-created group only sees messages
        submitted *after* this point.  This prevents stale messages from
        prior jobs being replayed when a new pipeline run connects to
        streams that still contain old entries.  Safe because the
        :class:`~.submitter.JobSubmitter` always (re-)submits messages for
        the current job after ``connect()``.
        """
        r = self._require_redis()
        for stage in self._ALL_STAGES:
            stream = self.streams.stream_for_stage(stage)
            try:
                await r.xgroup_create(
                    stream, self.consumer_group, id="$", mkstream=True
                )
                logger.debug("Created consumer group '%s' on %s", self.consumer_group, stream)
            except aioredis.ResponseError as exc:
                if "BUSYGROUP" not in str(exc):
                    raise
                # Group already exists — leave its read position as-is.
                # Resetting to "$" here would skip undelivered messages
                # from a parallel pipeline sharing this consumer group.

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def submit(self, stage: str, message: dict) -> str:
        """Append *message* to the stream for *stage*.

        The dict is JSON-encoded into a single ``"data"`` field so that
        nested structures survive the Redis flat-dict serialisation.

        Returns the Redis message ID assigned by the server.
        """
        r = self._require_redis()
        stream = self.streams.stream_for_stage(stage)
        flat = {"data": json.dumps(message)}
        msg_id: str = await r.xadd(stream, flat)
        logger.debug("Submitted to %s id=%s", stream, msg_id)
        return msg_id

    # ------------------------------------------------------------------
    # Consuming
    # ------------------------------------------------------------------

    async def read(
        self,
        stage: str,
        consumer_name: str,
        count: int = 1,
        block_ms: int = 5000,
    ) -> list[tuple[str, dict]]:
        """Fetch up to *count* new messages from *stage* for *consumer_name*.

        Uses ``XREADGROUP`` with ``>`` so only undelivered messages are
        returned.  Blocks for up to *block_ms* milliseconds if the stream
        is empty.

        Returns a list of ``(message_id, parsed_message_dict)`` tuples.
        """
        r = self._require_redis()
        stream = self.streams.stream_for_stage(stage)
        results = await r.xreadgroup(
            self.consumer_group,
            consumer_name,
            {stream: ">"},
            count=count,
            block=block_ms,
        )
        if not results:
            return []

        messages: list[tuple[str, dict]] = []
        for _stream_name, entries in results:
            for msg_id, fields in entries:
                data: dict = json.loads(fields["data"])
                messages.append((msg_id, data))
        return messages

    # ------------------------------------------------------------------
    # Acknowledgement
    # ------------------------------------------------------------------

    async def ack(self, stage: str, message_id: str) -> None:
        """Acknowledge *message_id* in the consumer group for *stage*."""
        r = self._require_redis()
        stream = self.streams.stream_for_stage(stage)
        await r.xack(stream, self.consumer_group, message_id)
        logger.debug("Acked %s on %s", message_id, stream)

    # ------------------------------------------------------------------
    # Dead-letter
    # ------------------------------------------------------------------

    async def send_to_dead_letter(
        self,
        stage: str,
        message_id: str,
        message: dict,
        error: str,
    ) -> None:
        """Forward a permanently-failed message to the dead-letter stream.

        Appends a wrapper envelope to ``stream:dead_letter`` containing the
        original stage, message ID, payload, and error string, then acks the
        original message so it leaves the pending set.
        """
        await self.submit(
            "dead_letter",
            {
                "original_stage": stage,
                "original_message_id": message_id,
                "message": message,
                "error": str(error),
            },
        )
        await self.ack(stage, message_id)
        logger.warning(
            "Moved message %s from stage '%s' to dead-letter: %s",
            message_id,
            stage,
            error,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    async def stream_length(self, stage: str) -> int:
        """Return the total number of entries in the stream for *stage*."""
        r = self._require_redis()
        stream = self.streams.stream_for_stage(stage)
        return await r.xlen(stream)

    async def count_by_job(self, stage: str, job_id: str) -> int:
        """Count entries in *stage*'s stream that belong to *job_id*.

        Reads all entries (``XRANGE 0 +``) and counts those whose
        ``data.job_id`` matches. For pipelines with < ~10k images per
        stream this is fast enough for a 5-second poll; for larger
        volumes a secondary index (sorted set keyed by job_id) would
        be more efficient.
        """
        r = self._require_redis()
        stream = self.streams.stream_for_stage(stage)
        count = 0
        # XRANGE returns all entries; we parse just enough to check job_id.
        entries = await r.xrange(stream)
        for _msg_id, fields in entries:
            try:
                data = json.loads(fields.get("data", "{}"))
                if data.get("job_id") == job_id:
                    count += 1
            except (json.JSONDecodeError, TypeError):
                continue
        return count

    async def pending_count(self, stage: str) -> int:
        """Return the number of pending (unacknowledged) messages for the consumer group."""
        r = self._require_redis()
        stream = self.streams.stream_for_stage(stage)
        try:
            info = await r.xpending(stream, self.consumer_group)
            # redis-py returns a dict with a "pending" key; older builds return a list.
            if isinstance(info, dict):
                return info.get("pending", 0)
            # list format: [count, min_id, max_id, consumers]
            return info[0] if info else 0
        except Exception:
            logger.debug("Could not fetch pending count for %s", stage, exc_info=True)
            return 0

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: "AutoAnnotationV3Config") -> "RedisMessageBroker":  # noqa: F821
        """Construct from an :class:`~..config.AutoAnnotationV3Config` instance."""
        redis_cfg = config.redis
        url = f"redis://{redis_cfg.host}:{redis_cfg.port}/{redis_cfg.db}"
        stream_cfg = StreamConfig.from_dict(redis_cfg.streams)
        return cls(
            redis_url=url,
            consumer_group=redis_cfg.consumer_group,
            stream_config=stream_cfg,
        )
