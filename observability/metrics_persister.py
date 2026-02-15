#!/usr/bin/env python3
"""
Metrics Persistence Service
Periodically reads Prometheus metrics and persists them to Redis for API access
"""

import time
import logging
import redis
from prometheus_client import REGISTRY
from typing import Dict, Any
import schedule
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsPersister:
    """Persists Prometheus metrics to Redis"""
    
    def __init__(self, redis_host: str = "redis", redis_port: int = 6379, redis_db: int = 3):
        """Initialize metrics persister"""
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True
        )
        logger.info(f"Connected to Redis at {redis_host}:{redis_port} DB{redis_db}")
    
    def collect_and_persist(self):
        """Collect metrics from Prometheus registry and persist to Redis"""
        try:
            metrics_data = {}
            
            # Iterate through all metrics in the registry
            for metric in REGISTRY.collect():
                metric_name = metric.name
                
                # Skip internal Prometheus metrics
                if metric_name.startswith('python_') or metric_name.startswith('process_'):
                    continue
                
                # Handle different metric types
                for sample in metric.samples:
                    sample_name = sample.name
                    sample_value = sample.value
                    sample_labels = sample.labels
                    
                    # Create Redis key
                    if sample_labels:
                        # Include labels in the key
                        label_str = ",".join([f"{k}={v}" for k, v in sorted(sample_labels.items())])
                        redis_key = f"metrics:{sample_name}:{label_str}"
                    else:
                        redis_key = f"metrics:{sample_name}"
                    
                    # Store in Redis with 24-hour expiration
                    self.redis_client.setex(redis_key, 86400, str(sample_value))
                    metrics_data[redis_key] = sample_value
            
            # Store aggregated totals for common queries
            self._store_aggregated_metrics()
            
            logger.info(f"Persisted {len(metrics_data)} metrics to Redis")
            
        except Exception as e:
            logger.error(f"Failed to persist metrics: {e}")
    
    def _store_aggregated_metrics(self):
        """Store aggregated metrics for common API queries"""
        try:
            # Get all keys for specific metrics
            input_validation_keys = self.redis_client.keys("metrics:input_validation_failures_total:*")
            authorization_keys = self.redis_client.keys("metrics:authorization_denials_total:*")
            memory_security_keys = self.redis_client.keys("metrics:memory_security_scrubs_total:*")
            output_validation_keys = self.redis_client.keys("metrics:output_validation_failures_total:*")
            requests_blocked_keys = self.redis_client.keys("metrics:requests_blocked_total:*")
            user_queries_keys = self.redis_client.keys("metrics:user_queries_total:*")
            successful_resolutions_keys = self.redis_client.keys("metrics:successful_resolutions_total:*")
            
            # Sum up values
            input_validation_total = sum([float(self.redis_client.get(k) or 0) for k in input_validation_keys])
            authorization_total = sum([float(self.redis_client.get(k) or 0) for k in authorization_keys])
            memory_security_total = sum([float(self.redis_client.get(k) or 0) for k in memory_security_keys])
            output_validation_total = sum([float(self.redis_client.get(k) or 0) for k in output_validation_keys])
            requests_blocked_total = sum([float(self.redis_client.get(k) or 0) for k in requests_blocked_keys])
            user_queries_total = sum([float(self.redis_client.get(k) or 0) for k in user_queries_keys])
            successful_resolutions_total = sum([float(self.redis_client.get(k) or 0) for k in successful_resolutions_keys])
            
            # Store aggregated values
            self.redis_client.setex("metrics:input_validation_failures", 86400, str(int(input_validation_total)))
            self.redis_client.setex("metrics:authorization_denials", 86400, str(int(authorization_total)))
            self.redis_client.setex("metrics:memory_security_scrubs", 86400, str(int(memory_security_total)))
            self.redis_client.setex("metrics:output_validation_failures", 86400, str(int(output_validation_total)))
            self.redis_client.setex("metrics:requests_blocked", 86400, str(int(requests_blocked_total)))
            self.redis_client.setex("metrics:user_queries_total", 86400, str(int(user_queries_total)))
            self.redis_client.setex("metrics:successful_resolutions", 86400, str(int(successful_resolutions_total)))
            
            logger.debug("Stored aggregated metrics")
            
        except Exception as e:
            logger.error(f"Failed to store aggregated metrics: {e}")
    
    def get_metric(self, metric_name: str, labels: Dict[str, str] = None) -> float:
        """Get a specific metric value from Redis"""
        try:
            if labels:
                label_str = ",".join([f"{k}={v}" for k, v in sorted(labels.items())])
                redis_key = f"metrics:{metric_name}:{label_str}"
            else:
                redis_key = f"metrics:{metric_name}"
            
            value = self.redis_client.get(redis_key)
            return float(value) if value else 0.0
            
        except Exception as e:
            logger.error(f"Failed to get metric {metric_name}: {e}")
            return 0.0
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all persisted metrics from Redis"""
        try:
            all_keys = self.redis_client.keys("metrics:*")
            metrics = {}
            
            for key in all_keys:
                value = self.redis_client.get(key)
                metrics[key.replace("metrics:", "")] = float(value) if value else 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get all metrics: {e}")
            return {}


def run_persister():
    """Run the metrics persister service"""
    logger.info("========================================")
    logger.info("Metrics Persistence Service Starting")
    logger.info("========================================")
    
    # Get Redis configuration from environment
    redis_host = os.getenv("REDIS_HOST", "redis")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_db = int(os.getenv("REDIS_METRICS_DB", "3"))
    
    # Initialize persister
    persister = MetricsPersister(redis_host, redis_port, redis_db)
    
    # Schedule periodic persistence (every 30 seconds)
    schedule.every(30).seconds.do(persister.collect_and_persist)
    
    logger.info("Metrics persister scheduled to run every 30 seconds")
    
    # Run immediately on startup
    persister.collect_and_persist()
    
    # Run scheduler
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Metrics persister stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in metrics persister: {e}")
            time.sleep(5)


if __name__ == "__main__":
    run_persister()
