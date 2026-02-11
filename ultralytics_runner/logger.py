# -*- coding: utf-8 -*-
"""
📦 core/common/logger.py

일관된 로깅 설정 + 회전 핸들러 + 컬러 콘솔 출력 지원

사용 예시:
    from core.common.logger import get_logger
    logger = get_logger(__name__)

    from core.common.logger import get_logger

    # 회전 안 함 (기본)
    logger = get_logger(__name__)

    # 크기 기준 회전 로그
    logger = get_logger(__name__, log_file="logs/train.log", use_rotation="size", max_bytes=2 * 1024 * 1024)

    # 날짜 기준 회전 로그
    logger = get_logger(__name__, use_rotation="time", backup_count=7)

"""

import logging
import sys
from pathlib import Path
from tqdm import tqdm
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False


# 색상 매핑
LEVEL_COLORS = {
    "DEBUG": Fore.LIGHTBLACK_EX,
    "INFO": Fore.WHITE,
    "WARNING": Fore.YELLOW,
    "ERROR": Fore.RED,
    "CRITICAL": Fore.MAGENTA,
}


class ColoredFormatter(logging.Formatter):
    """콘솔용 컬러 포맷터"""
    def format(self, record):
        msg = super().format(record)
        if COLORAMA_AVAILABLE and record.levelname in LEVEL_COLORS:
            color = LEVEL_COLORS[record.levelname]
            msg = f"{color}{msg}{Style.RESET_ALL}"
        return msg


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def get_logger(
    name: str = __name__,
    log_file: str = "logs/project.log",
    use_rotation: str = "size",  # "size", "time", or None
    max_bytes: int = 5 * 1024 * 1024,  # 5MB
    backup_count: int = 3
) -> logging.Logger:
    """일관된 로깅 설정 반환

    Args:
        name (str): 로거 이름
        log_file (str): 로그 파일 경로
        use_rotation (str): "size" | "time" | None
        max_bytes (int): 파일 크기 기준 회전 시 최대 크기
        backup_count (int): 백업 보존 개수

    Returns:
        logging.Logger: 설정된 로거
    """
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        # Formatter (공통 포맷)
        log_format = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"

        # tqdm-safe + 컬러 콘솔 핸들러로 대체
        tqdm_handler = TqdmLoggingHandler()
        tqdm_handler.setLevel(logging.DEBUG)
        tqdm_handler.setFormatter(ColoredFormatter(log_format, date_format))
        logger.addHandler(tqdm_handler)

        # 로그 디렉토리 생성
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # 파일 핸들러 설정
        if use_rotation == "size":
            file_handler = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
        elif use_rotation == "time":
            file_handler = TimedRotatingFileHandler(log_path, when="midnight", backupCount=backup_count, encoding="utf-8")
            file_handler.suffix = "%Y-%m-%d"
        else:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")

        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        logger.addHandler(file_handler)

    return logger
