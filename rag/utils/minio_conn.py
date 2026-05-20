"""MinIO 连接封装。

负责对象存储的读写、预签名 URL、桶删除和单桶/多桶兼容逻辑，
是文档、图片等二进制资源的主要存储适配层。
"""

#

import logging
import ssl
import time
from minio import Minio
from minio.commonconfig import CopySource
from minio.error import S3Error, ServerError, InvalidResponseError
from io import BytesIO
import urllib3
from common.decorator import singleton
from common import settings


def _build_minio_http_client():
    """按配置构造可选的 urllib3 HTTP 客户端。

    当 `MINIO.verify=false` 时，允许使用自签名证书。
    """
    verify = settings.MINIO.get("verify", True)
    if verify is True or verify == "true" or verify == "1":
        return None
    return urllib3.PoolManager(cert_reqs=ssl.CERT_NONE)


@singleton
class RAG_MedQAMinio:
    def __init__(self):
        self.conn = None
        # 把空字符串归一成 None，确保“单桶模式未启用”时状态清晰。
        self.bucket = settings.MINIO.get('bucket', None) or None
        self.prefix_path = settings.MINIO.get('prefix_path', None) or None
        self.__open__()

    @staticmethod
    def use_default_bucket(method):
        def wrapper(self, bucket, *args, **kwargs):
            # 如果配置了默认物理桶，则逻辑桶名会变成对象 key 的前缀。
            original_bucket = bucket
            actual_bucket = self.bucket if self.bucket else bucket
            if self.bucket:
                # 把原始逻辑桶名继续往下传，供路径改写装饰器使用。
                kwargs['_orig_bucket'] = original_bucket
            return method(self, actual_bucket, *args, **kwargs)

        return wrapper

    @staticmethod
    def use_prefix_path(method):
        def wrapper(self, bucket, fnm, *args, **kwargs):
            # 若启用了默认物理桶，则优先用原始逻辑桶名构造对象路径前缀。
            orig_bucket = kwargs.pop('_orig_bucket', None)

            if self.prefix_path:
                # 若配置了统一前缀，则路径形如 <prefix>/<logic-bucket>/<filename>。
                if orig_bucket:
                    fnm = f"{self.prefix_path}/{orig_bucket}/{fnm}"
                else:
                    fnm = f"{self.prefix_path}/{fnm}"
            else:
                # 未配置前缀时，如果存在逻辑桶名，则仍把它折叠进对象路径里。
                if orig_bucket and bucket == self.bucket:
                    fnm = f"{orig_bucket}/{fnm}"

            return method(self, bucket, fnm, *args, **kwargs)

        return wrapper

    def __open__(self):
        try:
            if self.conn:
                self.__close__()
        except Exception:
            pass

        try:
            secure = settings.MINIO.get("secure", False)
            if isinstance(secure, str):
                secure = secure.lower() in ("true", "1", "yes")
            http_client = _build_minio_http_client()
            self.conn = Minio(
                settings.MINIO["host"],
                access_key=settings.MINIO["user"],
                secret_key=settings.MINIO["password"],
                secure=secure,
                region=settings.MINIO.get("region", None) or None,
                http_client=http_client,
            )
        except Exception:
            logging.exception(
                "Fail to connect %s " % settings.MINIO["host"])

    def __close__(self):
        del self.conn
        self.conn = None

    def health(self):
        """检查 MinIO 是否可用。"""
        try:
            if self.bucket:
                # 单桶模式下只检查桶是否存在，不做有副作用的写入测试。
                exists = self.conn.bucket_exists(self.bucket)

                # 历史上这里做过两件事：
                # - 通过写入 `_health_check` 校验写权限
                # - 缺桶时自动创建存储桶

                return exists
            else:
                # 多桶模式下，列桶即可判断服务连通性。
                self.conn.list_buckets()
                return True
        except (S3Error, ServerError, InvalidResponseError):
            return False
        except Exception as e:
            logging.warning(f"Unexpected error in MinIO health check: {e}")
            return False

    @use_default_bucket
    @use_prefix_path
    def put(self, bucket, fnm, binary, tenant_id=None):
        for _ in range(3):
            try:
                # 多桶模式下缺桶时会自动补建；单桶模式则直接写入既有物理桶。
                if not self.bucket and not self.conn.bucket_exists(bucket):
                    self.conn.make_bucket(bucket)

                r = self.conn.put_object(bucket, fnm,
                                         BytesIO(binary),
                                         len(binary)
                                         )
                return r
            except Exception:
                logging.exception(f"Fail to put {bucket}/{fnm}:")
                self.__open__()
                time.sleep(1)

    @use_default_bucket
    @use_prefix_path
    def rm(self, bucket, fnm, tenant_id=None):
        try:
            self.conn.remove_object(bucket, fnm)
        except Exception:
            logging.exception(f"Fail to remove {bucket}/{fnm}:")

    @use_default_bucket
    @use_prefix_path
    def get(self, bucket, filename, tenant_id=None):
        for _ in range(1):
            try:
                r = self.conn.get_object(bucket, filename)
                return r.read()
            except Exception:
                logging.exception(f"Fail to get {bucket}/{filename}")
                self.__open__()
                time.sleep(1)
        return

    @use_default_bucket
    @use_prefix_path
    def obj_exist(self, bucket, filename, tenant_id=None):
        try:
            if not self.conn.bucket_exists(bucket):
                return False
            if self.conn.stat_object(bucket, filename):
                return True
            else:
                return False
        except S3Error as e:
            if e.code in ["NoSuchKey", "NoSuchBucket", "ResourceNotFound"]:
                return False
        except Exception:
            logging.exception(f"obj_exist {bucket}/{filename} got exception")
            return False

    @use_default_bucket
    def bucket_exists(self, bucket):
        try:
            if not self.conn.bucket_exists(bucket):
                return False
            else:
                return True
        except S3Error as e:
            if e.code in ["NoSuchKey", "NoSuchBucket", "ResourceNotFound"]:
                return False
        except Exception:
            logging.exception(f"bucket_exist {bucket} got exception")
            return False

    @use_default_bucket
    @use_prefix_path
    def get_presigned_url(self, bucket, fnm, expires, tenant_id=None):
        for _ in range(10):
            try:
                return self.conn.get_presigned_url("GET", bucket, fnm, expires)
            except Exception:
                logging.exception(f"Fail to get_presigned {bucket}/{fnm}:")
                self.__open__()
                time.sleep(1)
        return

    @use_default_bucket
    def remove_bucket(self, bucket, **kwargs):
        orig_bucket = kwargs.pop('_orig_bucket', None)
        try:
            if self.bucket:
                # 单桶模式下，只删除当前业务前缀下的对象。
                prefix = ""
                if self.prefix_path:
                    prefix = f"{self.prefix_path}/"
                if orig_bucket:
                    prefix += f"{orig_bucket}/"

                # 先列出该前缀下的所有对象，再逐个删除。
                objects_to_delete = self.conn.list_objects(bucket, prefix=prefix, recursive=True)
                for obj in objects_to_delete:
                    self.conn.remove_object(bucket, obj.object_name)
                # 这里不要删除物理桶本身，避免影响共享桶中的其他数据。
            else:
                if self.conn.bucket_exists(bucket):
                    objects_to_delete = self.conn.list_objects(bucket, recursive=True)
                    for obj in objects_to_delete:
                        self.conn.remove_object(bucket, obj.object_name)
                    self.conn.remove_bucket(bucket)
        except Exception:
            logging.exception(f"Fail to remove bucket {bucket}")

    def _resolve_bucket_and_path(self, bucket, fnm):
        if self.bucket:
            if self.prefix_path:
                fnm = f"{self.prefix_path}/{bucket}/{fnm}"
            else:
                fnm = f"{bucket}/{fnm}"
            bucket = self.bucket
        elif self.prefix_path:
            fnm = f"{self.prefix_path}/{fnm}"
        return bucket, fnm

    def copy(self, src_bucket, src_path, dest_bucket, dest_path):
        try:
            src_bucket, src_path = self._resolve_bucket_and_path(src_bucket, src_path)
            dest_bucket, dest_path = self._resolve_bucket_and_path(dest_bucket, dest_path)

            if not self.conn.bucket_exists(dest_bucket):
                self.conn.make_bucket(dest_bucket)

            try:
                self.conn.stat_object(src_bucket, src_path)
            except Exception as e:
                logging.exception(f"Source object not found: {src_bucket}/{src_path}, {e}")
                return False

            self.conn.copy_object(
                dest_bucket,
                dest_path,
                CopySource(src_bucket, src_path),
            )
            return True

        except Exception:
            logging.exception(f"Fail to copy {src_bucket}/{src_path} -> {dest_bucket}/{dest_path}")
            return False

    def move(self, src_bucket, src_path, dest_bucket, dest_path):
        try:
            if self.copy(src_bucket, src_path, dest_bucket, dest_path):
                self.rm(src_bucket, src_path)
                return True
            else:
                logging.error(f"Copy failed, move aborted: {src_bucket}/{src_path}")
                return False
        except Exception:
            logging.exception(f"Fail to move {src_bucket}/{src_path} -> {dest_bucket}/{dest_path}")
            return False
