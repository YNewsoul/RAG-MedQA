"""独立于项目原有链路的 MinerU 调用层。"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


class MinerURunner:
    """用 CLI 或 HTTP API 调用 MinerU，并返回 Markdown 文本。

    为什么单独抽这一层：

    1. 让上层 planning 不关心“MinerU 是本机命令还是远程服务”
    2. 便于以后替换成别的 PDF 结构化工具
    3. 让“获取 markdown”与“markdown 如何分块”职责分离
    """

    def __init__(
        self,
        mode: str = "auto",
        api_url: str = "",
        server_url: str = "",
        output_dir: str = "",
        delete_output: bool = True,
    ) -> None:
        # `mode=auto` 时，优先尝试 API；若未提供 API 地址，再走 CLI。
        self.mode = (mode or "auto").lower()
        self.api_url = (api_url or "").rstrip("/")
        self.server_url = (server_url or "").rstrip("/")
        self.output_dir = output_dir or ""
        self.delete_output = delete_output

    def check_installation(self) -> tuple[bool, str]:
        """检查当前可用的 MinerU 调用方式。

        返回 `(是否可用, 使用方式)`，供上层在真正开始 materialize 前做提示。
        """

        if self.api_url or self.server_url:
            return True, "api"
        cli = self._find_cli()
        if cli:
            return True, cli
        return False, "unavailable"

    def _find_cli(self) -> str | None:
        """探测当前环境中可用的 MinerU 命令。

        支持两个常见入口：
        - `mineru`
        - `magic-pdf`
        """
        for cmd in ("mineru", "magic-pdf"):
            try:
                result = subprocess.run(
                    [cmd, "--version"],
                    capture_output=True,
                    timeout=10,
                    text=True,
                )
                if result.returncode == 0:
                    return cmd
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        return None

    def _parse_via_cli(self, pdf_path: str, out_dir: str, parse_method: str) -> str:
        """通过本机命令执行 MinerU，并从输出目录中提取 markdown。"""

        cli = self._find_cli()
        if not cli:
            raise RuntimeError(
                "MinerU CLI not found. Please install mineru/magic-pdf or use API mode."
            )

        args = [cli, "-p", pdf_path, "-o", out_dir, "-m", parse_method or "auto"]
        logging.info("MinerU CLI: %s", " ".join(args))
        result = subprocess.run(args, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            raise RuntimeError(
                f"MinerU CLI exited with code {result.returncode}:\n{result.stderr}"
            )

        # MinerU 不同版本的输出目录结构可能不同，
        # 所以这里先按常见路径找，再兜底全目录搜索 markdown。
        pdf_stem = Path(pdf_path).stem
        candidates = [
            Path(out_dir) / pdf_stem / "auto" / f"{pdf_stem}.md",
            Path(out_dir) / pdf_stem / f"{pdf_stem}.md",
            Path(out_dir) / f"{pdf_stem}.md",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.read_text(encoding="utf-8")

        md_files = sorted(Path(out_dir).rglob("*.md"))
        if md_files:
            return md_files[0].read_text(encoding="utf-8")

        raise FileNotFoundError(f"MinerU output markdown not found under {out_dir!r}")

    def _parse_via_api(self, binary: bytes, parse_method: str) -> str:
        """通过 HTTP API 调用 MinerU 服务端。"""

        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("requests package is required for MinerU API mode") from exc

        def extract_markdown(payload: Any) -> str | None:
            """尽量从 MinerU 返回体里提取 markdown 文本。

            兼容几类常见返回：
            1. 直接返回 `md_content` / `markdown`
            2. 返回 `results -> <filename> -> md_content`
            3. 返回 `data` / `result` 再嵌套一层
            """

            if isinstance(payload, str):
                return payload

            if isinstance(payload, dict):
                for field in ("md_content", "markdown", "content"):
                    value = payload.get(field)
                    if isinstance(value, str) and value.strip():
                        return value

                nested = payload.get("results")
                if isinstance(nested, dict):
                    for item in nested.values():
                        extracted = extract_markdown(item)
                        if extracted:
                            return extracted

                for field in ("result", "data"):
                    extracted = extract_markdown(payload.get(field))
                    if extracted:
                        return extracted

            if isinstance(payload, list):
                for item in payload:
                    extracted = extract_markdown(item)
                    if extracted:
                        return extracted

            return None

        base = self.api_url or self.server_url
        errors: list[str] = []
        for endpoint in ("/file_parse", "/predict", "/parse", "/api/v1/parse"):
            url = f"{base}{endpoint}"
            try:
                resp = requests.post(
                    url,
                    files={"files": ("document.pdf", binary, "application/pdf")},
                    data={
                        "parse_method": parse_method or "auto",
                        "return_md": "true",
                    },
                    timeout=1800,
                )
                if resp.status_code != 200:
                    errors.append(f"{url}: HTTP {resp.status_code} {resp.text[:300]}")
                    continue
                data = resp.json()

                markdown_text = extract_markdown(data)
                if markdown_text:
                    return markdown_text

                # 新版本 MinerU 常见写法是先返回 task/result_url，再去拿结果。
                result_url = data.get("result_url") if isinstance(data, dict) else None
                if isinstance(result_url, str) and result_url:
                    result_resp = requests.get(result_url, timeout=1800)
                    if result_resp.status_code == 200:
                        result_data = result_resp.json()
                        markdown_text = extract_markdown(result_data)
                        if markdown_text:
                            return markdown_text

                return str(data)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{url}: {exc}")
        raise RuntimeError(
            f"All MinerU API endpoints failed for {base!r}. Errors:\n" + "\n".join(errors)
        )

    def parse_to_markdown(
        self,
        filepath: Path,
        parse_method: str = "auto",
        binary: bytes | None = None,
    ) -> str:
        """执行一次 MinerU 解析，并返回 Markdown 文本。

        这层只保证“拿到 markdown 文本”：
        - 不负责后续分块
        - 不负责噪声清洗
        - 不负责切成 logical document
        """

        # 自动模式下，如果提供了 API 地址，则优先使用 API。
        # 原因是远程 API 往往比本机 CLI 更稳定，也更适合批量任务。
        use_api = self.mode == "api" or (self.mode == "auto" and (self.api_url or self.server_url))
        tmp_pdf: str | None = None
        tmp_out_dir: str | None = None
        try:
            if use_api:
                if binary is None:
                    binary = filepath.read_bytes()
                return self._parse_via_api(binary, parse_method)

            # CLI 模式要求有磁盘上的 PDF 文件。
            # 如果上层只给了二进制，就先临时写到一个 pdf 文件里。
            if filepath.exists():
                pdf_path = str(filepath)
            else:
                if binary is None:
                    raise ValueError("Either filepath or binary must be provided.")
                fd, tmp_pdf = tempfile.mkstemp(suffix=".pdf")
                os.close(fd)
                Path(tmp_pdf).write_bytes(binary)
                pdf_path = tmp_pdf

            # 如果外部没指定固定输出目录，就给这次调用单独开一个临时目录。
            # 这样多个文件并发或重复执行时不容易互相污染。
            out_dir = self.output_dir
            if not out_dir:
                tmp_out_dir = tempfile.mkdtemp(prefix="mineru_pdf_")
                out_dir = tmp_out_dir
            return self._parse_via_cli(pdf_path, out_dir, parse_method)
        finally:
            # 临时文件和临时输出目录都在这里统一清理。
            # 如果用户传了固定 output_dir，则不会清掉那个固定目录。
            if tmp_pdf and os.path.exists(tmp_pdf):
                try:
                    os.unlink(tmp_pdf)
                except OSError:
                    pass
            if self.delete_output and tmp_out_dir and os.path.exists(tmp_out_dir):
                shutil.rmtree(tmp_out_dir, ignore_errors=True)
