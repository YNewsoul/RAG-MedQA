"""惰性图片封装。

`LazyImage` 会把多个图片 blob 延迟到真正访问时再解码/拼接，
适合文档解析和检索阶段减少不必要的图像开销。
"""

import logging
from io import BytesIO

from PIL import Image

from rag.nlp import concat_img


class LazyImage:
    """惰性图片对象，可在需要时再转成真正的 PIL Image。"""
    def __init__(self, blobs, source=None):
        self._blobs = [b for b in (blobs or []) if b]
        self.source = source
        self._pil = None

    def __bool__(self):
        return bool(self._blobs)

    def to_pil(self):
        """把内部 blob 列表解码并拼成一个 PIL Image。"""
        if self._pil is not None:
            try:
                self._pil.load()
                return self._pil
            except Exception:
                try:
                    self._pil.close()
                except Exception:
                    pass
                self._pil = None
        res_img = None
        for blob in self._blobs:
            try:
                image = Image.open(BytesIO(blob)).convert("RGB")
            except Exception as e:
                logging.info(f"LazyImage: skip bad image blob: {e}")
                continue

            if res_img is None:
                res_img = image
                continue

            new_img = concat_img(res_img, image)
            if new_img is not res_img:
                try:
                    res_img.close()
                except Exception:
                    pass
            try:
                image.close()
            except Exception:
                pass
            res_img = new_img

        self._pil = res_img
        return self._pil

    def to_pil_detached(self):
        """返回 PIL Image，并把内部缓存 ownership 交给调用方。"""
        pil = self.to_pil()
        self._pil = None
        return pil

    def close(self):
        """显式释放内部缓存的 PIL Image。"""
        if self._pil is not None:
            try:
                self._pil.close()
            except Exception:
                pass
            self._pil = None
        return None

    def __getattr__(self, name):
        pil = self.to_pil()
        if pil is None:
            raise AttributeError(name)
        return getattr(pil, name)

    def __array__(self, dtype=None):
        import numpy as np

        pil = self.to_pil()
        if pil is None:
            return np.array([], dtype=dtype)
        return np.array(pil, dtype=dtype)

    def __enter__(self):
        return self.to_pil()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    @staticmethod
    def merge(a, b):
        """合并两个 LazyImage，把它们的 blob 列表串起来。"""
        a_blobs = a._blobs if isinstance(a, LazyImage) else []
        b_blobs = b._blobs if isinstance(b, LazyImage) else []
        combined = a_blobs + b_blobs
        if not combined:
            return None
        merged = LazyImage(combined)
        return merged


LazyDocxImage = LazyImage


def ensure_pil_image(img):
    """尽量把输入统一成 PIL Image。"""
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, LazyImage):
        return img.to_pil()
    return None


def is_image_like(img):
    """判断对象是否可被当作图片处理。"""
    return isinstance(img, Image.Image) or isinstance(img, LazyImage)


def open_image_for_processing(img, *, allow_bytes=False):
    """为后续处理打开图片，并返回 `(image, 是否需要调用方关闭)`。"""
    if isinstance(img, Image.Image):
        return img, False
    if isinstance(img, LazyImage):
        return img.to_pil_detached(), True
    if allow_bytes and isinstance(img, (bytes, bytearray)):
        try:
            pil = Image.open(BytesIO(img)).convert("RGB")
            return pil, True
        except Exception as e:
            logging.info(f"open_image_for_processing: bad bytes: {e}")
            return None, False
    return img, False
