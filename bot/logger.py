"""Loglama ve (isteğe bağlı) Telegram bildirimi."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import requests

from .config import ROOT


def get_logger(name: str = "bot") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    log_dir = Path(os.getenv("BOT_LOG_DIR", ROOT / "logs"))   # testler ayrı klasöre yazar
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "bot.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


class Notifier:
    def __init__(self, token: str = "", chat_id: str = ""):
        self.token, self.chat_id = token, chat_id
        self.log = get_logger()

    def send(self, text: str) -> None:
        self.log.info(text)
        if not (self.token and self.chat_id):
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": text},
                timeout=10,
            )
        except Exception as e:  # bildirim hatası botu durdurmamalı
            self.log.warning(f"Telegram bildirimi gönderilemedi: {e}")
