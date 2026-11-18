"""
CASTER Phase 3 -- thin Redis queue wrapper. Bridges the Phase 1/2
CV+classification pipeline (producer) and Phase 3's LLM worker (consumer),
so a slow LLM call never blocks frame processing (see
CASTER_Phase3_DeepDive.docx Section 6.1).

Requires: pip install redis
Requires: a running Redis instance (docker run -p 6379:6379 redis, or
          `redis-server` locally -- free, no cloud dependency)
"""
import json
import os

import redis

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_QUEUE_KEY = "caster:scene_json_events"


class SceneEventQueue:
    def __init__(self, host=None, port=None, key=None):
        self.client = redis.Redis(
            host=host or REDIS_HOST, port=port or REDIS_PORT, decode_responses=True,
        )
        self.key = key or REDIS_QUEUE_KEY

    def push(self, scene_json: dict):
        """Called by the Phase 1/2 pipeline whenever build_scene_json() has
        produced an event that already passed Phase 2's confidence gate."""
        self.client.rpush(self.key, json.dumps(scene_json, default=str))

    def pop(self, timeout: int = 5):
        """Called by the Phase 3 LLM worker. Blocks up to `timeout` seconds;
        returns None on timeout so the caller's loop can check for shutdown
        signals periodically."""
        result = self.client.blpop(self.key, timeout=timeout)
        if result is None:
            return None
        _, raw = result
        return json.loads(raw)

    def queue_length(self) -> int:
        return self.client.llen(self.key)
