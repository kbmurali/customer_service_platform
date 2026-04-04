#!/usr/bin/env python3
"""
Metrics Persistence Service — multiprocess-safe.

Periodically reads Prometheus metrics and persists them to Redis for API
access.  Supports uvicorn multi-worker deployments via:

  1. **prometheus_client multiprocess mode** — when PROMETHEUS_MULTIPROC_DIR
     is set, each worker writes metrics to shared mmap files.  The persister
     uses ``MultiProcessCollector`` to read the aggregated view.

  2. **Redis leader election** — only one worker pushes metrics to Redis
     and runs the evaluation pipeline.  The leader acquires a short-lived
     Redis lock (``internal:metrics_leader``) each cycle.  If a leader dies, another
     worker picks up the lock within one cycle (30s).
"""

import os
import time
import logging
import threading
import redis
from typing import Dict, Any

from prometheus_client import (
    CollectorRegistry,
    REGISTRY,
    generate_latest,
)

# Multiprocess support — only imported when the env var is set
_MULTIPROC_DIR = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
_USE_MULTIPROC = bool(_MULTIPROC_DIR)
if _USE_MULTIPROC:
    try:
        from prometheus_client.multiprocess import MultiProcessCollector
    except ImportError:
        _USE_MULTIPROC = False

# Evaluation pipeline (soft import — missing deps don't crash the persister)
try:
    from observability.evaluation_pipeline import run_evaluation_cycle
    _EVAL_PIPELINE_AVAILABLE = True
except Exception:
    _EVAL_PIPELINE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Leader lock config
_LEADER_KEY = "internal:metrics_leader"
_LEADER_TTL = 45   # seconds — slightly longer than push interval


class MetricsPersister:
    """Persists Prometheus metrics to Redis — multiprocess-safe."""

    def __init__(self, redis_host: str = "redis", redis_port: int = 6379, redis_db: int = 3):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True
        )
        self._worker_id = f"pid-{os.getpid()}"
        logger.info(
            "MetricsPersister [%s] connected to Redis %s:%s DB%s (multiproc=%s)",
            self._worker_id, redis_host, redis_port, redis_db, _USE_MULTIPROC,
        )

    def _acquire_leader(self) -> bool:
        """Try to become the leader for this push cycle (Redis SETNX)."""
        try:
            acquired = self.redis_client.set(
                _LEADER_KEY, self._worker_id,
                nx=True, ex=_LEADER_TTL,
            )
            if acquired:
                return True
            # Check if we are already the leader
            current = self.redis_client.get(_LEADER_KEY)
            if current == self._worker_id:
                self.redis_client.expire(_LEADER_KEY, _LEADER_TTL)
                return True
            return False
        except Exception as e:
            logger.debug("Leader acquisition failed: %s", e)
            return True  # Fail open — push anyway if Redis is down

    def collect_and_persist(self):
        """Collect metrics from all workers and persist to Redis."""
        if not self._acquire_leader():
            return

        try:
            metrics_data = {}

            if _USE_MULTIPROC:
                registry = CollectorRegistry()
                MultiProcessCollector(registry)
                source = registry
            else:
                source = REGISTRY

            for metric in source.collect():
                metric_name = metric.name

                if metric_name.startswith('python_') or metric_name.startswith('process_'):
                    continue

                for sample in metric.samples:
                    sample_name = sample.name
                    sample_value = sample.value
                    sample_labels = sample.labels

                    if sample_labels:
                        label_str = ",".join([f"{k}={v}" for k, v in sorted(sample_labels.items())])
                        redis_key = f"metrics:{sample_name}:{label_str}"
                    else:
                        redis_key = f"metrics:{sample_name}"

                    self.redis_client.setex(redis_key, 86400, str(sample_value))
                    metrics_data[redis_key] = sample_value

            self._store_aggregated_metrics()
            self.update_user_token_tallies()
            if _EVAL_PIPELINE_AVAILABLE:
                run_evaluation_cycle(redis_client=self.redis_client)

            logger.info(
                "MetricsPersister [%s] persisted %d metrics to Redis",
                self._worker_id, len(metrics_data),
            )

        except Exception as e:
            logger.error("Failed to persist metrics: %s", e)

    def _store_aggregated_metrics(self):
        """Store aggregated metrics for common API queries."""
        try:
            key_groups = {
                "input_validation_failures": "metrics:input_validation_failures_total:*",
                "authorization_denials": "metrics:authorization_denials_total:*",
                "memory_security_scrubs": "metrics:memory_security_scrubs_total:*",
                "output_validation_failures": "metrics:output_validation_failures_total:*",
                "requests_blocked": "metrics:requests_blocked_total:*",
                "user_queries_total": "metrics:user_queries_total:*",
                "successful_resolutions": "metrics:successful_resolutions_total:*",
            }
            for agg_name, pattern in key_groups.items():
                keys = self.redis_client.keys(pattern)
                total = sum([float(self.redis_client.get(k) or 0) for k in keys])
                self.redis_client.setex(f"metrics:{agg_name}", 86400, str(int(total)))

            logger.debug("Stored aggregated metrics")
        except Exception as e:
            logger.error("Failed to store aggregated metrics: %s", e)

        # Persist security control counters from audit_logs to MySQL
        # so the security heatmap survives Prometheus TSDB expiry.
        self._persist_security_counters()

    def _persist_security_counters(self):
        """
        Aggregate security events from audit_logs by action type for the
        current hour and store them in Redis with a ``security_hourly:``
        prefix.  This provides a permanent audit-log-backed source for the
        security heatmap that does not depend on in-process Prometheus
        counters or their 15-day TSDB retention.

        Throttled to once per hour via a Redis key.
        """
        _THROTTLE_KEY = "metrics:security_counters:last_persist"
        try:
            if self.redis_client.exists(_THROTTLE_KEY):
                return
        except Exception:
            pass  # Redis unavailable — skip

        try:
            from databases.connections import get_mysql
            from datetime import datetime, timezone, timedelta

            mysql = get_mysql()
            now = datetime.now(timezone.utc)
            hour_start = now.replace(minute=0, second=0, microsecond=0)
            hour_end = hour_start + timedelta(hours=1)
            date_str = hour_start.strftime("%Y-%m-%d")
            hour_val = hour_start.hour

            # Map audit_log actions to security heatmap categories
            action_map = {
                "INPUT_VALIDATION_BLOCK":   "nemo_blocks",
                "AUTHORIZATION_DENIED":     "rbac_denials",
                "RATE_LIMIT_EXCEEDED":      "rate_limited",
                "INPUT_SANITIZED":          "nh3_sanitized",
                "PII_SCRUBBED":             "presidio_vaulted",
                "OUTPUT_VALIDATION_BLOCK":  "output_validation",
                "APPROVAL_REQUESTED":       "approval_requests",
                "REPLAY_REJECTED":          "replay_rejected",
            }

            rows = mysql.execute_query("""
                SELECT action, COUNT(*) AS cnt
                FROM audit_logs
                WHERE timestamp >= %s AND timestamp < %s
                  AND action IN (%s)
                GROUP BY action
            """.replace(
                "IN (%s)",
                "IN (" + ",".join(["%s"] * len(action_map)) + ")"
            ), (hour_start, hour_end, *action_map.keys()))

            counters = {v: 0 for v in action_map.values()}
            for row in (rows or []):
                action = row.get("action", "")
                mapped = action_map.get(action)
                if mapped:
                    counters[mapped] = int(row.get("cnt", 0))

            # Store hourly counters in Redis (permanent key, no TTL — accumulates)
            redis_key = f"security_hourly:{date_str}:{hour_val:02d}"
            import json
            self.redis_client.set(redis_key, json.dumps(counters))

            # Set throttle
            self.redis_client.setex(_THROTTLE_KEY, 3600, "1")

            logger.debug("Persisted security counters for %s hour %d", date_str, hour_val)

        except ImportError:
            logger.debug("MySQL not available for security counter persistence")
        except Exception as exc:
            logger.error("Failed to persist security counters: %s", exc)

    def update_user_token_tallies(self):
        """Update per-user token usage counters in Redis for budget enforcement."""
        try:
            budget_controller = None
            try:
                from agents.core.budget_controller import get_budget_controller
                budget_controller = get_budget_controller()
            except Exception:
                pass

            if budget_controller is None:
                return

            user_token_keys = self.redis_client.keys("metrics:token_usage_total:*")
            for key in user_token_keys:
                try:
                    parts = key.split(":")
                    if len(parts) >= 4:
                        user_id = parts[3]
                        tokens = int(self.redis_client.get(key) or 0)
                        if tokens > 0:
                            budget_controller.record_usage(user_id, tokens)
                except Exception as inner:
                    logger.debug("update_user_token_tallies: key=%s error=%s", key, inner)

        except Exception as exc:
            logger.error("Failed to update user token tallies: %s", exc)

    def get_metric(self, metric_name: str, labels: Dict[str, str] = None) -> float:
        """Get a specific metric value from Redis."""
        try:
            if labels:
                label_str = ",".join([f"{k}={v}" for k, v in sorted(labels.items())])
                redis_key = f"metrics:{metric_name}:{label_str}"
            else:
                redis_key = f"metrics:{metric_name}"

            value = self.redis_client.get(redis_key)
            return float(value) if value else 0.0
        except Exception as e:
            logger.error("Failed to get metric %s: %s", metric_name, e)
            return 0.0

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all persisted metrics from Redis."""
        try:
            all_keys = self.redis_client.keys("metrics:*")
            metrics = {}
            for key in all_keys:
                value = self.redis_client.get(key)
                metrics[key.replace("metrics:", "")] = float(value) if value else 0.0
            return metrics
        except Exception as e:
            logger.error("Failed to get all metrics: %s", e)
            return {}


def start_background_pusher(interval_seconds: int = 30) -> None:
    """
    Start the MetricsPersister push loop in a daemon thread.

    Call this once from each worker's lifespan startup.  In a multi-worker
    deployment, all workers start this thread but only the Redis leader
    actually pushes — the others skip silently via _acquire_leader().

    The thread is a daemon so it terminates automatically when the process
    exits — no cleanup is required in the lifespan teardown.
    """
    redis_host = os.getenv("REDIS_HOST", "redis")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_db = int(os.getenv("REDIS_METRICS_DB", "3"))

    persister = MetricsPersister(redis_host, redis_port, redis_db)

    def _loop():
        persister.collect_and_persist()
        while True:
            time.sleep(interval_seconds)
            persister.collect_and_persist()

    thread = threading.Thread(target=_loop, daemon=True, name="metrics-pusher")
    thread.start()
    logger.info(
        "Metrics background pusher started [pid-%s] (interval=%ds multiproc=%s)",
        os.getpid(), interval_seconds, _USE_MULTIPROC,
    )
