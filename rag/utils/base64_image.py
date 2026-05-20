
"""图片与存储标识互转辅助。

负责把内存中的图片对象上传到对象存储，并把图片 ID 回填到 chunk；
也负责根据图片 ID 再反查回 PIL Image。
"""

import base64
import logging
from functools import partial
from io import BytesIO

from PIL import Image



from common.misc_utils import thread_pool_exec
from rag.utils.lazy_image import open_image_for_processing

test_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAIAAAD/gAIDAAAA6ElEQVR4nO3QwQ3AIBDAsIP9d25XIC+EZE8QZc18w5l9O+AlZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBT+IYAHHLHkdEgAAAABJRU5ErkJggg=="
test_image = base64.b64decode(test_image_base64)


async def image2id(d: dict, storage_put_func: partial, objname: str, bucket: str = "imagetemps"):
    """把字典中的图片上传到对象存储，并回填 `img_id`。"""
    import logging
    from io import BytesIO
    from rag.svr.task_executor import minio_limiter

    if "image" not in d:
        return
    if not d["image"]:
        del d["image"]
        return

    def encode_image():
        """把图片对象标准化编码成 JPEG 二进制。"""
        with BytesIO() as buf:
            img, close_after = open_image_for_processing(d["image"], allow_bytes=False)

            if isinstance(img, bytes):
                buf.write(img)
                buf.seek(0)
                return buf.getvalue()

            if not isinstance(img, Image.Image):
                return None

            if img.mode in ("RGBA", "P"):
                orig_img = img
                img = img.convert("RGB")
                if close_after:
                    try:
                        orig_img.close()
                    except Exception:
                        pass

            try:
                img.save(buf, format="JPEG")
                buf.seek(0)
                return buf.getvalue()
            except OSError as e:
                logging.warning(f"Saving image exception: {e}")
                return None
            finally:
                if close_after:
                    try:
                        img.close()
                    except Exception:
                        pass

    jpeg_binary = await thread_pool_exec(encode_image)
    if jpeg_binary is None:
        del d["image"]
        return

    async with minio_limiter:
        await thread_pool_exec(
            lambda: storage_put_func(bucket=bucket, fnm=objname, binary=jpeg_binary)
        )

    d["img_id"] = f"{bucket}-{objname}"

    if not isinstance(d["image"], bytes):
        d["image"].close()
    del d["image"]


def id2image(image_id: str | None, storage_get_func: partial):
    """根据 `bucket-object` 形式的图片 ID 取回 PIL Image。"""
    if not image_id:
        return
    arr = image_id.split("-")
    if len(arr) != 2:
        return
    bkt, nm = image_id.split("-")
    try:
        blob = storage_get_func(bucket=bkt, fnm=nm)
        if not blob:
            return
        return Image.open(BytesIO(blob))
    except Exception as e:
        logging.exception(e)
