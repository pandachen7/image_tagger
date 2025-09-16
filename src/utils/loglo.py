"""
PURPOSE:
1. print to console, 搭配systemctl也能夠儲存log檔
2. log msg to files, 預設會在你執行路徑的旁邊的logs資料夾內
  - 可用TimedRotatingFileHandler來針對不同天數分檔
3. 用python env設定log level

2025.09.12 by Panda
"""

import inspect
import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

from dotenv import dotenv_values

# usage e.g.
"""
from lib.loglo import getUniqueLogger
log = getUniqueLogger(__file__)
log.d("I'm log.", "sth else", 123)
# will print `I'm log. sth else 123`
"""
FMT_CONSOLE_DATE = "%y%m%d_%H:%M:%S"
LOG_FOLDER = "./logs"

SHOW_CONSOLE = True
SAVE_LOG = False


class CustomLogger(logging.Logger):
    def d(self, *args, stacklevel=1):
        prefix = ""
        frame = inspect.currentframe().f_back
        for var_name, var_val in frame.f_locals.items():
            if var_val is args[0]:
                prefix = f"{var_name}<{type(var_val).__name__}> "
        msg = f"{prefix}" + " ".join(str(a) for a in args)
        super().debug(msg, stacklevel=stacklevel + 1)

    def i(self, *args, stacklevel=1):
        msg = " ".join(str(a) for a in args)
        super().info(msg, stacklevel=stacklevel + 1)

    def w(self, *args, stacklevel=1):
        msg = " ".join(str(a) for a in args)
        super().warning(msg, stacklevel=stacklevel + 1)

    def e(self, *args, stacklevel=1):
        msg = " ".join(str(a) for a in args)
        super().error(msg, stacklevel=stacklevel + 1)

    def c(self, *args, stacklevel=1):
        msg = " ".join(str(a) for a in args)
        super().critical(msg, stacklevel=stacklevel + 1)


logging.setLoggerClass(CustomLogger)


class ColorFormatter(logging.Formatter):
    COLOR_CODES = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[1;31m",  # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLOR_CODES.get(record.levelno, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


# 如果用systemctl, console的訊息會直接紀錄到你設定的log路徑中, 因此通常可以不用改動
# 但如果希望用TimedRotatingFileHandler來命名不同的log檔名, 那還是以save_log為主
def getUniqueLogger(
    filepath=__file__,
    show_console=SHOW_CONSOLE,
    save_log=SAVE_LOG,
    log_folder=LOG_FOLDER,
    **whatever,  # 對應舊的params, 不用理會的kwargs
):
    # 已有register就直接回傳
    if filepath in logging.Logger.manager.loggerDict:
        return logging.getLogger(filepath)

    # 務必用filepath, 以免同檔名互相干擾
    _logger = logging.getLogger(filepath)

    # fastapi似乎會影響這裡的handler, 所以先除掉
    # if _logger.hasHandlers():
    #     _logger.handlers.clear()
    _logger.propagate = False  # 不要傳遞到logger root

    if show_console:
        consoleh = logging.StreamHandler(sys.stdout)

        format_log = "%(asctime)s %(filename)s:%(lineno)d.%(funcName)-8s %(levelname)-.1s %(message)s"

        # consoleh.setFormatter(logging.Formatter(format_log, datefmt=FMT_CONSOLE_DATE))
        consoleh.setFormatter(ColorFormatter(format_log, datefmt=FMT_CONSOLE_DATE))
        _logger.addHandler(consoleh)

    if save_log:
        if not os.path.exists(log_folder):
            os.makedirs(log_folder, exist_ok=True)

        format_log = (
            "%(asctime)s %(filename)s:%(lineno)d.%(funcName)s %(levelname)s %(message)s"
        )

        logfile_path = os.path.join(log_folder, "log")
        fileh = TimedRotatingFileHandler(logfile_path, when="midnight", backupCount=365)
        fileh.suffix = "%Y-%m-%d.log"
        # 10秒一次, 保留5筆的話: ('./logs/log.out', when='S', interval=10, backupCount=5)
        fileh.setFormatter(logging.Formatter(format_log))
        _logger.addHandler(fileh)

    # 用'.env'設定log level, 你的`.env`檔案會在專案根目錄, 預計文字內容只有一行如下
    """
    LOG_LEVEL=DEBUG
    """
    penv = dotenv_values(".env")
    if "LOG_LEVEL" in penv:
        if penv["LOG_LEVEL"] == "CRITICAL":
            _logger.setLevel(level=logging.CRITICAL)
        elif penv["LOG_LEVEL"] == "ERROR":
            _logger.setLevel(level=logging.ERROR)
        elif penv["LOG_LEVEL"] == "WARNING":
            _logger.setLevel(level=logging.WARNING)
        elif penv["LOG_LEVEL"] == "INFO":
            _logger.setLevel(level=logging.INFO)
        else:
            _logger.setLevel(level=logging.DEBUG)
    else:
        _logger.setLevel(level=logging.DEBUG)

    # abbreviation, use case e.g. log.d("I'm log.")
    # 舊寫法, 透過CustomLogger取代
    # _logger.d = _logger.debug
    # _logger.i = _logger.info
    # _logger.w = _logger.warning
    # _logger.e = _logger.error
    # _logger.c = _logger.critical

    # # a little annoying, should take off them
    # if whatever:
    #     print("should not set params: ", *whatever)

    return _logger


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
    log.c()
