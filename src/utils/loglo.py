"""自定義 Logger - 使用 Adapter 模式包裝 logging.Logger
PURPOSE:
1. print to console, 搭配 systemctl 也能夠儲存 log 檔
2. log msg to files, 預設會在你執行路徑的旁邊的 logs 資料夾內
  - 可用 TimedRotatingFileHandler 來針對不同天數分檔
3. 用 python env 設定 log level

2026.01.30 by Panda
"""

import logging
import os
import sys
import traceback
from logging.handlers import TimedRotatingFileHandler

from typing import Any

from dotenv import dotenv_values

# usage e.g.
"""
from putils.logger import getUniqueLogger
log = getUniqueLogger(__file__)
log.d("I'm log.", "sth else", 123)
# will print `I'm log. sth else 123`
"""

FMT_CONSOLE_DATE = "%y%m%d_%H:%M:%S"
LOG_FOLDER = "./logs"

SHOW_CONSOLE = True
SAVE_LOG = False

SRC_ROOT = os.path.join(os.getcwd(), "src")


def extract_src_frames(exc_info: tuple[type, BaseException, Any]) -> str:
    """回傳所有在專案根目錄下的 traceback frame"""
    tb_list = traceback.extract_tb(exc_info[2])
    project_frames = [frame for frame in tb_list if frame.filename.startswith(SRC_ROOT)]

    if project_frames:
        return "".join(traceback.format_list(project_frames))
    else:
        # fallback: 最後一行通常是錯誤拋出處
        return "".join(traceback.format_list(tb_list[-1:]))


class LogAdapter:
    """Logger Adapter - 包裝 logging.Logger 提供簡短方法名"""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def dt(self, *args, stacklevel: int = 1) -> None:
        """debug with type - 顯示每個參數的類型

        Example:
            log.dt(my_obj, 123, "hello")
            # output: <MyClass>... <int>123 <str>hello
        """
        parts = [f"<{type(a).__name__}>{a}" for a in args]
        msg = " ".join(parts)
        self._logger.debug(msg, stacklevel=stacklevel + 1)

    def d(self, *args, stacklevel: int = 1) -> None:
        """debug"""
        msg = " ".join(str(a) for a in args)
        self._logger.debug(msg, stacklevel=stacklevel + 1)

    def i(self, *args, stacklevel: int = 1) -> None:
        """info"""
        msg = " ".join(str(a) for a in args)
        self._logger.info(msg, stacklevel=stacklevel + 1)

    def w(self, *args, stacklevel: int = 1) -> None:
        """warning"""
        msg = " ".join(str(a) for a in args)
        self._logger.warning(msg, stacklevel=stacklevel + 1)

    def e(self, *args, stacklevel: int = 1) -> None:
        """error"""
        msg = " ".join(str(a) for a in args)
        self._logger.error(msg, stacklevel=stacklevel + 1)

    def et(self, *args, stacklevel: int = 1) -> None:
        """error with traceback"""
        tb_str = extract_src_frames(sys.exc_info())
        msg = " ".join(str(a) for a in args)
        self._logger.error(f"未預期錯誤: {msg}\n{tb_str}", stacklevel=stacklevel + 1)

    def c(self, *args, stacklevel: int = 1) -> None:
        """critical"""
        msg = " ".join(str(a) for a in args)
        self._logger.critical(msg, stacklevel=stacklevel + 1)

    # 相容原本 logging.Logger 的方法
    def debug(self, *args, stacklevel: int = 1, **kwargs) -> None:
        msg = " ".join(str(a) for a in args)
        self._logger.debug(msg, stacklevel=stacklevel + 1, **kwargs)

    def info(self, *args, stacklevel: int = 1, **kwargs) -> None:
        msg = " ".join(str(a) for a in args)
        self._logger.info(msg, stacklevel=stacklevel + 1, **kwargs)

    def warning(self, *args, stacklevel: int = 1, **kwargs) -> None:
        msg = " ".join(str(a) for a in args)
        self._logger.warning(msg, stacklevel=stacklevel + 1, **kwargs)

    def error(self, *args, stacklevel: int = 1, **kwargs) -> None:
        msg = " ".join(str(a) for a in args)
        self._logger.error(msg, stacklevel=stacklevel + 1, **kwargs)

    def critical(self, *args, stacklevel: int = 1, **kwargs) -> None:
        msg = " ".join(str(a) for a in args)
        self._logger.critical(msg, stacklevel=stacklevel + 1, **kwargs)

    def fatal(self, *args, stacklevel: int = 1, **kwargs) -> None:
        msg = " ".join(str(a) for a in args)
        self._logger.fatal(msg, stacklevel=stacklevel + 2, **kwargs)


class ColorFormatter(logging.Formatter):
    """帶顏色的 log formatter"""

    COLOR_CODES = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[1;31m",  # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLOR_CODES.get(record.levelno, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


# 用 .env 讀取 log level（只讀取一次）
_env_values = dotenv_values(".env")
_LOG_LEVEL_MAP = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}
_DEFAULT_LOG_LEVEL = _LOG_LEVEL_MAP.get(
    str(_env_values.get("LOG_LEVEL", "")), logging.DEBUG
)

# 快取 LogAdapter 實例
_adapters: dict[str, LogAdapter] = {}


def getUniqueLogger(
    name: str = __file__,
    show_console: bool = SHOW_CONSOLE,
    save_log: bool = SAVE_LOG,
    log_folder: str = LOG_FOLDER,
) -> LogAdapter:
    """取得或創建指定名稱的 LogAdapter

    Args:
        name: logger 名稱（建議用 __file__）
        show_console: 是否輸出到 console
        save_log: 是否儲存到檔案
        log_folder: log 檔案資料夾路徑

    Returns:
        LogAdapter 實例
    """
    # 已有快取就直接回傳
    if name in _adapters:
        return _adapters[name]

    # 創建新的 logger
    _logger = logging.getLogger(name)
    _logger.propagate = False  # 不要傳遞到 logger root

    # 避免重複添加 handler
    if not _logger.hasHandlers():
        if show_console:
            consoleh = logging.StreamHandler(sys.stdout)
            format_log = (
                "%(asctime)s %(filename)s:%(lineno)d.%(funcName)-8s "
                "%(levelname)-.1s %(message)s"
            )
            consoleh.setFormatter(ColorFormatter(format_log, datefmt=FMT_CONSOLE_DATE))
            _logger.addHandler(consoleh)

        if save_log:
            os.makedirs(log_folder, exist_ok=True)
            format_log = (
                "%(asctime)s %(filename)s:%(lineno)d.%(funcName)s "
                "%(levelname)s %(message)s"
            )
            logfile_path = os.path.join(log_folder, "log")
            fileh = TimedRotatingFileHandler(
                logfile_path, when="midnight", backupCount=365
            )
            fileh.suffix = "%Y-%m-%d.log"
            fileh.setFormatter(logging.Formatter(format_log))
            _logger.addHandler(fileh)

        _logger.setLevel(_DEFAULT_LOG_LEVEL)

    adapter = LogAdapter(_logger)
    _adapters[name] = adapter
    return adapter


if __name__ == "__main__":
    log = getUniqueLogger(__file__)

    log.debug("I'm log.")
    log.info("I'm log.")
    log.warning("I'm log.")
    log.error("I'm log.")
    log.fatal("I'm log.")

    the_var_name = "I'm log."
    appended_text = "sth else"
    appended_int = 123
    log.d(the_var_name, appended_text, appended_int)
    log.i("I'm log.", appended_text, appended_int)
    log.w("I'm log.", appended_text, appended_int)
    log.e("I'm log.", appended_text, appended_int)
    log.c("critical test")
