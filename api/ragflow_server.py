

"""RAG-MedQA 服务器启动入口文件。

本文件负责启动整个 RAG-MedQA 应用服务，包括：
1. 初始化日志系统
2. 初始化数据库表和初始数据
3. 配置运行时参数
4. 启动后台进度更新线程
5. 启动 HTTP 服务器

环境变量:
    RAG_MedQA_DEBUGPY_LISTEN: debugpy 监听端口，默认 0（不启用）

命令行参数:
    --version: 显示版本信息并退出
    --debug: 启用调试模式
    --init-superuser: 初始化超级用户
"""

print("Start RAG-MedQA server...")

import time
start_ts = time.time()  # 记录启动时间戳，用于计算初始化耗时

import logging
import os
import signal
import sys
import threading
import uuid
import faulthandler  # 用于捕获崩溃信息

# 应用核心模块导入
from api.apps import app  # Quart 应用实例
from api.db.runtime_config import RuntimeConfig  # 运行时配置管理
from api.db.services.document_service import DocumentService  # 文档服务
from common.file_utils import get_project_base_directory  # 获取项目基础目录
from common import settings  # 全局设置
from api.db.db_models import init_database_tables as init_web_db  # 数据库表初始化
from api.db.init_data import init_web_data, init_superuser  # 初始数据初始化
from common.versions import get_RAG_MedQA_version  # 获取版本号
from common.config_utils import show_configs  # 显示配置信息
from common.log_utils import init_root_logger  # 初始化根日志器
from rag.utils.redis_conn import RedisDistributedLock  # Redis 分布式锁

# 全局停止事件，用于优雅关闭
stop_event = threading.Event()

# debugpy 调试端口配置（环境变量）
RAG_MedQA_DEBUGPY_LISTEN = int(os.environ.get('RAG_MedQA_DEBUGPY_LISTEN', "0"))

def update_progress():
    """后台进度更新线程函数。

    定期更新文档处理进度，使用 Redis 分布式锁确保多实例部署时
    只有一个实例执行进度更新，避免重复操作。

    执行周期：每6秒检查一次
    锁超时时间：60秒
    """
    # 生成唯一锁值，用于标识当前实例
    lock_value = str(uuid.uuid4())
    # 创建 Redis 分布式锁
    redis_lock = RedisDistributedLock("update_progress", lock_value=lock_value, timeout=60)
    logging.info(f"update_progress lock_value: {lock_value}")

    # 循环执行直到收到停止信号
    while not stop_event.is_set():
        try:
            # 尝试获取锁
            if redis_lock.acquire():
                # 获取锁成功，执行进度更新
                DocumentService.update_progress()
                redis_lock.release()
        except Exception:
            # 捕获并记录异常
            logging.exception("update_progress exception")
        finally:
            # 确保释放锁
            try:
                redis_lock.release()
            except Exception:
                logging.exception("update_progress exception")
            # 等待6秒后再次执行
            stop_event.wait(6)

def signal_handler(sig, frame):
    """信号处理器，用于优雅关闭服务器。

    处理 SIGINT (Ctrl+C) 和 SIGTERM 信号，执行以下操作：
    1. 关闭所有 MCP 会话
    2. 设置停止事件，通知后台线程退出
    3. 等待1秒后退出进程

    Args:
        sig: 信号编号
        frame: 当前栈帧
    """
    logging.info("Received interrupt signal, shutting down...")
    shutdown_all_mcp_sessions()  # 关闭所有 MCP 会话
    stop_event.set()  # 设置停止信号
    stop_event.wait(1)  # 等待后台线程退出
    sys.exit(0)

if __name__ == '__main__':
    """主入口函数，启动 RAG-MedQA 服务器。"""
    # 启用故障处理器，捕获崩溃信息
    faulthandler.enable()
    # 初始化根日志器
    init_root_logger("RAG-MedQA_server")
    # 打印启动 Logo
    logging.info(r"""
        ____   ___    ______ ______ __
       / __ \ /   |  / ____// ____// /____  _      __
      / /_/ // /| | / / __ / /_   / // __ \| | /| / /
     / _, _// ___ |/ /_/ // __/  / // /_/ /| |/ |/ /
    /_/ |_|/_/  |_|\____//_/    /_/ \____/ |__/|__/

    """)
    # 记录版本信息
    logging.info(f'RAG-MedQA version: {get_RAG_MedQA_version()}')
    # 记录项目基础目录
    logging.info(f'project base: {get_project_base_directory()}')
    # 显示配置信息
    show_configs()
    # 初始化设置
    settings.init_settings()
    # 打印 RAG 设置
    settings.print_rag_settings()

    # 如果配置了 debugpy 端口，启用远程调试
    if RAG_MedQA_DEBUGPY_LISTEN > 0:
        logging.info(f"debugpy listen on {RAG_MedQA_DEBUGPY_LISTEN}")
        import debugpy
        debugpy.listen(("0.0.0.0", RAG_MedQA_DEBUGPY_LISTEN))

    # ==================== 数据库初始化 ====================
    # 初始化数据库表
    init_web_db()
    # 初始化初始数据
    init_web_data()

    # ==================== 命令行参数解析 ====================
    import argparse
    parser = argparse.ArgumentParser(description="RAG-MedQA Server")
    parser.add_argument(
        "--version", default=False, help="Show RAG-MedQA version", action="store_true"
    )
    parser.add_argument(
        "--debug", default=False, help="Enable debug mode", action="store_true"
    )
    parser.add_argument(
        "--init-superuser", default=False, help="Initialize superuser", action="store_true"
    )
    args = parser.parse_args()

    # 如果请求版本信息，打印并退出
    if args.version:
        print(get_RAG_MedQA_version())
        sys.exit(0)

    # 如果请求初始化超级用户
    if args.init_superuser:
        init_superuser()

    # 设置调试模式
    RuntimeConfig.DEBUG = args.debug
    if RuntimeConfig.DEBUG:
        logging.info("Running in debug mode")

    # ==================== 运行时配置 ====================
    RuntimeConfig.init_env()  # 初始化环境变量
    RuntimeConfig.init_config(
        JOB_SERVER_HOST=settings.HOST_IP, 
        HTTP_PORT=settings.HOST_PORT
    )

    # ==================== 信号处理注册 ====================
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # 终止信号

    # ==================== 后台线程启动 ====================
    def delayed_start_update_progress():
        """延迟启动进度更新线程。

        使用延迟启动确保其他组件已初始化完成。
        """
        logging.info("Starting update_progress thread (delayed)")
        # 创建守护线程执行进度更新
        t = threading.Thread(target=update_progress, daemon=True)
        t.start()

    # 在调试模式下，需要检查是否是 Werkzeug 重载后的主进程
    if RuntimeConfig.DEBUG:
        if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
            threading.Timer(1.0, delayed_start_update_progress).start()
    else:
        threading.Timer(1.0, delayed_start_update_progress).start()

    # ==================== HTTP 服务器启动 ====================
    try:
        # 记录初始化完成耗时
        logging.info(f"RAG-MedQA server is ready after {time.time() - start_ts:.2f}s initialization.")
        # 启动 Quart 应用
        # use_reloader: 调试模式下启用自动重载
        # debug: 禁用 Quart 内置调试器（使用外部 debugpy）
        app.run(
            host=settings.HOST_IP, 
            port=settings.HOST_PORT, 
            use_reloader=RuntimeConfig.DEBUG, 
            debug=False
        )
    except Exception as e:
        # 捕获未处理的异常，记录日志并强制退出
        logging.exception(f"Unhandled exception: {e}")
        stop_event.set()
        stop_event.wait(1)
        os.kill(os.getpid(), signal.SIGKILL)