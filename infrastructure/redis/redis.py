from typing import TypeVar
from redis import Redis
from core.config.config import nested_config as config
T = TypeVar('T')


class RedisClient:
    def __init__(self):
        self.redis_url = config["redis"]["url"]
        self._redis = None

    @property
    def redis(self):
        if self._redis is None:
            self._redis = Redis.from_url(self.redis_url, decode_responses=False)
        return self._redis


redis_client = RedisClient()