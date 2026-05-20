"""Redis / Valkey 连接封装。

负责缓存、分布式锁、队列、令牌桶限流等功能，是任务调度和进度刷新链路的
关键基础设施之一。
"""

#

import asyncio
import logging
import json
import uuid

import valkey as redis
from common.decorator import singleton
from common import settings
from valkey.lock import Lock

REDIS = {}
try:
    REDIS = settings.decrypt_database_config(name="redis")
except Exception:
    try:
        REDIS = settings.get_base_config("redis", {})
    except Exception:
        REDIS = {}


class RedisMsg:
    """对 Redis Stream 消息做的轻量包装。"""
    def __init__(self, consumer, queue_name, group_name, msg_id, message):
        self.__consumer = consumer
        self.__queue_name = queue_name
        self.__group_name = group_name
        self.__msg_id = msg_id
        self.__message = json.loads(message["message"])

    def ack(self):
        """确认消息已消费。"""
        try:
            self.__consumer.xack(self.__queue_name, self.__group_name, self.__msg_id)
            return True
        except Exception as e:
            logging.warning("[EXCEPTION]ack" + str(self.__queue_name) + "||" + str(e))
        return False

    def get_message(self):
        """返回反序列化后的消息体。"""
        return self.__message

    def get_msg_id(self):
        """返回消息 ID。"""
        return self.__msg_id


@singleton
class RedisDB:
    lua_delete_if_equal = None
    lua_token_bucket = None
    LUA_DELETE_IF_EQUAL_SCRIPT = """
        local current_value = redis.call('get', KEYS[1])
        if current_value and current_value == ARGV[1] then
            redis.call('del', KEYS[1])
            return 1
        end
        return 0
    """

    LUA_TOKEN_BUCKET_SCRIPT = """
        -- KEYS[1] = rate limit key
        -- ARGV[1] = capacity
        -- ARGV[2] = rate
        -- ARGV[3] = now
        -- ARGV[4] = cost

        local key       = KEYS[1]
        local capacity  = tonumber(ARGV[1])
        local rate      = tonumber(ARGV[2])
        local now       = tonumber(ARGV[3])
        local cost      = tonumber(ARGV[4])

        local data = redis.call("HMGET", key, "tokens", "timestamp")
        local tokens = tonumber(data[1])
        local last_ts = tonumber(data[2])

        if tokens == nil then
            tokens = capacity
            last_ts = now
        end

        local delta = math.max(0, now - last_ts)
        tokens = math.min(capacity, tokens + delta * rate)

        if tokens < cost then
            return {0, tokens}
        end

        tokens = tokens - cost

        redis.call("HMSET", key,
            "tokens", tokens,
            "timestamp", now
        )

        redis.call("EXPIRE", key, math.ceil(capacity / rate * 2))

        return {1, tokens}
    """

    def __init__(self):
        self.REDIS = None
        self.config = REDIS
        self.__open__()

    def register_scripts(self) -> None:
        """把 Lua 脚本注册到当前 Redis 连接。"""
        cls = self.__class__
        client = self.REDIS
        cls.lua_delete_if_equal = client.register_script(cls.LUA_DELETE_IF_EQUAL_SCRIPT)
        cls.lua_token_bucket = client.register_script(cls.LUA_TOKEN_BUCKET_SCRIPT)

    def __open__(self):
        """根据配置建立 Redis 连接。"""
        try:
            conn_params = {
                "host": self.config["host"].split(":")[0],
                "port": int(self.config.get("host", ":6379").split(":")[1]),
                "db": int(self.config.get("db", 1)),
                "decode_responses": True,
            }
            username = self.config.get("username")
            if username:
                conn_params["username"] = username
            password = self.config.get("password")
            if password:
                conn_params["password"] = password

            self.REDIS = redis.StrictRedis(**conn_params)

            self.register_scripts()
        except Exception as e:
            logging.warning(f"Redis can't be connected. Error: {str(e)}")
        return self.REDIS

    def health(self):
        """做一次最小读写测试，检查 Redis 是否健康。"""
        self.REDIS.ping()
        a, b = "xx", "yy"
        self.REDIS.set(a, b, 3)

        if self.REDIS.get(a) == b:
            return True
        return False

    def info(self):
        """返回常用 Redis 运行时信息。"""
        info = self.REDIS.info()
        return {
            'redis_version': info["redis_version"],
            'server_mode': info["server_mode"] if "server_mode" in info else info.get("redis_mode", ""),
            'used_memory': info["used_memory_human"],
            'total_system_memory': info["total_system_memory_human"],
            'mem_fragmentation_ratio': info["mem_fragmentation_ratio"],
            'connected_clients': info["connected_clients"],
            'blocked_clients': info["blocked_clients"],
            'instantaneous_ops_per_sec': info["instantaneous_ops_per_sec"],
            'total_commands_processed': info["total_commands_processed"]
        }

    def is_alive(self):
        """判断连接对象是否已经建立。"""
        return self.REDIS is not None

    def exist(self, k):
        if not self.REDIS:
            return None
        try:
            return self.REDIS.exists(k)
        except Exception as e:
            logging.warning("RedisDB.exist " + str(k) + " got exception: " + str(e))
            self.__open__()

    def get(self, k):
        if not self.REDIS:
            return None
        try:
            return self.REDIS.get(k)
        except Exception as e:
            logging.warning("RedisDB.get " + str(k) + " got exception: " + str(e))
            self.__open__()

    def set_obj(self, k, obj, exp=3600):
        try:
            self.REDIS.set(k, json.dumps(obj, ensure_ascii=False), exp)
            return True
        except Exception as e:
            logging.warning("RedisDB.set_obj " + str(k) + " got exception: " + str(e))
            self.__open__()
        return False

    def set(self, k, v, exp=3600):
        try:
            self.REDIS.set(k, v, exp)
            return True
        except Exception as e:
            logging.warning("RedisDB.set " + str(k) + " got exception: " + str(e))
            self.__open__()
        return False

    def sadd(self, key: str, member: str):
        try:
            self.REDIS.sadd(key, member)
            return True
        except Exception as e:
            logging.warning("RedisDB.sadd " + str(key) + " got exception: " + str(e))
            self.__open__()
        return False

    def srem(self, key: str, member: str):
        try:
            self.REDIS.srem(key, member)
            return True
        except Exception as e:
            logging.warning("RedisDB.srem " + str(key) + " got exception: " + str(e))
            self.__open__()
        return False

    def smembers(self, key: str):
        try:
            res = self.REDIS.smembers(key)
            return res
        except Exception as e:
            logging.warning(
                "RedisDB.smembers " + str(key) + " got exception: " + str(e)
            )
            self.__open__()
        return None

    def zadd(self, key: str, member: str, score: float):
        try:
            self.REDIS.zadd(key, {member: score})
            return True
        except Exception as e:
            logging.warning("RedisDB.zadd " + str(key) + " got exception: " + str(e))
            self.__open__()
        return False

    def zcount(self, key: str, min: float, max: float):
        try:
            res = self.REDIS.zcount(key, min, max)
            return res
        except Exception as e:
            logging.warning("RedisDB.zcount " + str(key) + " got exception: " + str(e))
            self.__open__()
        return 0

    def zpopmin(self, key: str, count: int):
        try:
            res = self.REDIS.zpopmin(key, count)
            return res
        except Exception as e:
            logging.warning("RedisDB.zpopmin " + str(key) + " got exception: " + str(e))
            self.__open__()
        return None

    def zrangebyscore(self, key: str, min: float, max: float):
        try:
            res = self.REDIS.zrangebyscore(key, min, max)
            return res
        except Exception as e:
            logging.warning(
                "RedisDB.zrangebyscore " + str(key) + " got exception: " + str(e)
            )
            self.__open__()
        return None

    def zremrangebyscore(self, key: str, min: float, max: float):
        try:
            res = self.REDIS.zremrangebyscore(key, min, max)
            return res
        except Exception as e:
            logging.warning(
                f"RedisDB.zremrangebyscore {key} got exception: {e}"
            )
            self.__open__()
        return 0

    def incrby(self, key: str, increment: int):
        return self.REDIS.incrby(key, increment)

    def decrby(self, key: str, decrement: int):
        return self.REDIS.decrby(key, decrement)

    def generate_auto_increment_id(self, key_prefix: str = "id_generator", namespace: str = "default",
                                   increment: int = 1, ensure_minimum: int | None = None) -> int:
        redis_key = f"{key_prefix}:{namespace}"

        try:
            # 使用 pipeline，把这组操作尽量放在同一个原子批次里执行。
            pipe = self.REDIS.pipeline()

            # 先判断 key 是否已经存在。
            pipe.exists(redis_key)

            # 根据是否要求最小值，走“读取后修正”或直接自增。
            if ensure_minimum is not None:
                # 如果业务要求一个最小起始值，就先按最小值校正。
                pipe.get(redis_key)
                results = pipe.execute()

                if results[0] == 0:  # key 还不存在
                    start_id = max(1, ensure_minimum)
                    pipe.set(redis_key, start_id)
                    pipe.execute()
                    return start_id
                else:
                    current = int(results[1])
                    if current < ensure_minimum:
                        pipe.set(redis_key, ensure_minimum)
                        pipe.execute()
                        return ensure_minimum

            # 执行自增。
            next_id = self.REDIS.incrby(redis_key, increment)

            # 如果是第一次生成，则补一个更合理的初始值。
            if next_id == increment:
                self.REDIS.set(redis_key, 1 + increment)
                return 1 + increment

            return next_id

        except Exception as e:
            logging.warning("RedisDB.generate_auto_increment_id got exception: " + str(e))
            self.__open__()
        return -1

    def get_or_create_secret_key(self, key_name: str, new_value: str) -> str:
        """
        原子地获取一个已有密钥；如果不存在，就创建并返回它。

        这个方法保证在并发调用场景下，最终只会有一个值真正写入 Redis，
        其余调用方都会拿到同一个结果。
        """
        # 先尝试直接读取已有值。
        existing_value = self.REDIS.get(key_name)
        if existing_value is not None:
            logging.debug("Retrieved existing key from Redis")
            return existing_value

        # 用 SETNX 做“仅当不存在时才写入”，这是这里的并发保护核心。
        # SETNX 返回 True 表示本次成功写入，False 表示别的请求已经先写了。
        if self.REDIS.setnx(key_name, new_value):
            logging.info("Successfully created new secret key in Redis")
            return new_value

        # 如果 SETNX 失败，说明并发期间已经有别的进程创建了这个 key。
        # 这时再读一次，把最终落库的值返回给调用方。
        final_key = self.REDIS.get(key_name)
        if final_key is None:
            # 极少数情况下可能刚好遇到竞争窗口，再递归重试一次。
            logging.warning("Key disappeared during concurrent access, retrying...")
            return self.get_or_create_secret_key(key_name, new_value)

        logging.debug("Retrieved key created by another process")
        return final_key

    def transaction(self, key, value, exp=3600):
        try:
            pipeline = self.REDIS.pipeline(transaction=True)
            pipeline.set(key, value, exp, nx=True)
            pipeline.execute()
            return True
        except Exception as e:
            logging.warning(
                "RedisDB.transaction " + str(key) + " got exception: " + str(e)
            )
            self.__open__()
        return False

    def queue_product(self, queue, message) -> bool:
        for _ in range(3):
            try:
                payload = {"message": json.dumps(message)}
                self.REDIS.xadd(queue, payload)
                return True
            except Exception as e:
                logging.exception(
                    "RedisDB.queue_product " + str(queue) + " got exception: " + str(e)
                )
                self.__open__()
        return False

    def queue_consumer(self, queue_name, group_name, consumer_name, msg_id=b">") -> RedisMsg:
        """https://redis.io/docs/latest/commands/xreadgroup/"""
        for _ in range(3):
            try:

                try:
                    group_info = self.REDIS.xinfo_groups(queue_name)
                    if not any(gi["name"] == group_name for gi in group_info):
                        self.REDIS.xgroup_create(queue_name, group_name, id="0", mkstream=True)
                except redis.exceptions.ResponseError as e:
                    if "no such key" in str(e).lower():
                        self.REDIS.xgroup_create(queue_name, group_name, id="0", mkstream=True)
                    elif "busygroup" in str(e).lower():
                        logging.warning("Group already exists, continue.")
                        pass
                    else:
                        raise

                args = {
                    "groupname": group_name,
                    "consumername": consumer_name,
                    "count": 1,
                    "block": 5,
                    "streams": {queue_name: msg_id},
                }
                messages = self.REDIS.xreadgroup(**args)
                if not messages:
                    return None
                stream, element_list = messages[0]
                if not element_list:
                    return None
                msg_id, payload = element_list[0]
                res = RedisMsg(self.REDIS, queue_name, group_name, msg_id, payload)
                return res
            except Exception as e:
                if str(e) == 'no such key':
                    pass
                else:
                    logging.exception(
                        "RedisDB.queue_consumer "
                        + str(queue_name)
                        + " got exception: "
                        + str(e)
                    )
                    self.__open__()
        return None

    def get_unacked_iterator(self, queue_names: list[str], group_name, consumer_name):
        try:
            for queue_name in queue_names:
                try:
                    group_info = self.REDIS.xinfo_groups(queue_name)
                except Exception as e:
                    if str(e) == 'no such key':
                        logging.warning(f"RedisDB.get_unacked_iterator queue {queue_name} doesn't exist")
                        continue
                if not any(gi["name"] == group_name for gi in group_info):
                    logging.warning(f"RedisDB.get_unacked_iterator queue {queue_name} group {group_name} doesn't exist")
                    continue
                current_min = 0
                while True:
                    payload = self.queue_consumer(queue_name, group_name, consumer_name, current_min)
                    if not payload:
                        break
                    current_min = payload.get_msg_id()
                    logging.info(f"RedisDB.get_unacked_iterator {queue_name} {consumer_name} {current_min}")
                    yield payload
        except Exception:
            logging.exception(
                "RedisDB.get_unacked_iterator got exception: "
            )
            self.__open__()

    def get_pending_msg(self, queue, group_name):
        try:
            messages = self.REDIS.xpending_range(queue, group_name, '-', '+', 10)
            return messages
        except Exception as e:
            if 'No such key' not in (str(e) or ''):
                logging.warning(
                    "RedisDB.get_pending_msg " + str(queue) + " got exception: " + str(e)
                )
        return []

    def requeue_msg(self, queue: str, group_name: str, msg_id: str):
        for _ in range(3):
            try:
                messages = self.REDIS.xrange(queue, msg_id, msg_id)
                if messages:
                    self.REDIS.xadd(queue, messages[0][1])
                    self.REDIS.xack(queue, group_name, msg_id)
            except Exception as e:
                logging.warning(
                    "RedisDB.get_pending_msg " + str(queue) + " got exception: " + str(e)
                )
                self.__open__()

    def queue_info(self, queue, group_name) -> dict | None:
        for _ in range(3):
            try:
                groups = self.REDIS.xinfo_groups(queue)
                for group in groups:
                    if group["name"] == group_name:
                        return group
            except Exception as e:
                logging.warning(
                    "RedisDB.queue_info " + str(queue) + " got exception: " + str(e)
                )
                self.__open__()
        return None

    def delete_if_equal(self, key: str, expected_value: str) -> bool:
        """
        Do following atomically:
        Delete a key if its value is equals to the given one, do nothing otherwise.
        """
        return bool(self.lua_delete_if_equal(keys=[key], args=[expected_value], client=self.REDIS))

    def delete(self, key) -> bool:
        try:
            self.REDIS.delete(key)
            return True
        except Exception as e:
            logging.warning("RedisDB.delete " + str(key) + " got exception: " + str(e))
            self.__open__()
        return False


REDIS_CONN = RedisDB()


class RedisDistributedLock:
    def __init__(self, lock_key, lock_value=None, timeout=10, blocking_timeout=1):
        self.lock_key = lock_key
        if lock_value:
            self.lock_value = lock_value
        else:
            self.lock_value = str(uuid.uuid4())
        self.timeout = timeout
        self.lock = Lock(REDIS_CONN.REDIS, lock_key, timeout=timeout, blocking_timeout=blocking_timeout)

    def acquire(self):
        REDIS_CONN.delete_if_equal(self.lock_key, self.lock_value)
        return self.lock.acquire(token=self.lock_value)

    async def spin_acquire(self):
        REDIS_CONN.delete_if_equal(self.lock_key, self.lock_value)
        while True:
            if self.lock.acquire(token=self.lock_value):
                break
            await asyncio.sleep(10)

    def release(self):
        REDIS_CONN.delete_if_equal(self.lock_key, self.lock_value)
