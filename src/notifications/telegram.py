"""Centralized Telegram notification client.

Migrated from experiments/telegram_bot.py with proper logging,
auto-splitting for long messages, and dry-run support.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 4096


class TelegramNotifier:
    """Sends messages to a Telegram chat via the Bot API."""

    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> None:
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_configured(self) -> bool:
        """True when both token and chat_id are set."""
        return bool(self.token) and bool(self.chat_id)

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    def send_message(
        self,
        text: str,
        parse_mode: str = "HTML",
    ) -> bool:
        """Send a single message (must be <= 4096 chars).

        Returns True on success, False otherwise.  When unconfigured the
        message is printed to the log as a dry-run.
        """
        if not self.is_configured:
            logger.info("[DRY RUN] Would send Telegram message:\n%s", text)
            return False

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }

        try:
            response = requests.post(url, json=payload, timeout=15)
            if response.status_code == 200:
                logger.info("Telegram message sent (%d chars)", len(text))
                return True
            logger.error(
                "Telegram API error %d: %s", response.status_code, response.text
            )
            return False
        except requests.RequestException as exc:
            logger.error("Telegram send failed: %s", exc)
            return False

    def send_long_message(
        self,
        text: str,
        parse_mode: str = "HTML",
    ) -> bool:
        """Send a message, auto-splitting at newline boundaries if > 4096 chars."""
        if len(text) <= MAX_MESSAGE_LENGTH:
            return self.send_message(text, parse_mode=parse_mode)

        parts = _split_message(text, MAX_MESSAGE_LENGTH)
        return self.send_parts(parts, parse_mode=parse_mode)

    def send_parts(
        self,
        parts: list[str],
        parse_mode: str = "HTML",
    ) -> bool:
        """Send a list of message parts sequentially.

        Returns True if ALL parts were sent successfully.
        """
        if not parts:
            return True

        all_ok = True
        for i, part in enumerate(parts, 1):
            if not part.strip():
                continue
            logger.debug("Sending part %d/%d (%d chars)", i, len(parts), len(part))
            ok = self.send_message(part, parse_mode=parse_mode)
            if not ok:
                all_ok = False
        return all_ok

    def test_connection(self) -> bool:
        """Verify bot token is valid."""
        if not self.token:
            logger.warning("No Telegram token configured")
            return False

        url = f"https://api.telegram.org/bot{self.token}/getMe"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("ok"):
                    bot_name = data["result"].get("username")
                    logger.info("Connected to Telegram bot: @%s", bot_name)
                    return True
            logger.error("Telegram connection failed: %s", response.text)
            return False
        except requests.RequestException as exc:
            logger.error("Telegram connection error: %s", exc)
            return False


def _split_message(text: str, max_len: int) -> list[str]:
    """Split *text* into chunks of at most *max_len* chars on newline boundaries."""
    if len(text) <= max_len:
        return [text]

    parts: list[str] = []
    current_lines: list[str] = []
    current_len = 0

    for line in text.split("\n"):
        line_len = len(line) + 1  # +1 for the newline
        if current_len + line_len > max_len and current_lines:
            parts.append("\n".join(current_lines))
            current_lines = []
            current_len = 0
        current_lines.append(line)
        current_len += line_len

    if current_lines:
        parts.append("\n".join(current_lines))

    return parts
