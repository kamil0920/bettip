"""Centralized HF Hub retry utilities with exponential backoff.

All CI jobs sharing parallel HF Hub access should use these wrappers
instead of raw huggingface_hub calls to handle 429 rate limits gracefully.

Retry config: 5 retries, exponential backoff 60s -> 120s -> 240s -> 300s -> 300s
with 0-30s random jitter to prevent thundering herd from parallel CI jobs.
"""

import logging
import os
import random
import time
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 5
DEFAULT_BASE_DELAY = 60
DEFAULT_MAX_DELAY = 300
DEFAULT_JITTER = 30


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is retryable (429 or 5xx)."""
    msg = str(exc).lower()

    # Check for HfHubHTTPError or requests.HTTPError status codes
    status_code = getattr(exc, "response", None)
    if status_code is not None:
        status_code = getattr(status_code, "status_code", None)
    if status_code is not None:
        if status_code == 429 or status_code >= 500:
            return True

    # Fallback: check message content
    if "429" in msg or "rate limit" in msg or "too many requests" in msg:
        return True
    if "500" in msg or "502" in msg or "503" in msg or "504" in msg:
        return True
    if "server error" in msg or "service unavailable" in msg:
        return True

    return False


def hf_retry(
    fn: Callable,
    *args: Any,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    jitter: float = DEFAULT_JITTER,
    **kwargs: Any,
) -> Any:
    """Call fn with exponential backoff on 429/5xx errors.

    Args:
        fn: Function to call.
        *args: Positional arguments for fn.
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay cap in seconds.
        jitter: Maximum random jitter in seconds added to each delay.
        **kwargs: Keyword arguments for fn.

    Returns:
        The return value of fn.
    """
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries or not _is_retryable(e):
                raise

            delay = min(base_delay * (2**attempt), max_delay)
            delay += random.uniform(0, jitter)
            logger.warning(
                "HF Hub call failed (attempt %d/%d): %s. Retrying in %.0fs...",
                attempt + 1,
                max_retries + 1,
                e,
                delay,
            )
            time.sleep(delay)

    # Should not be reached, but satisfy type checker
    raise RuntimeError("hf_retry exhausted all retries")


def _get_defaults() -> tuple[str, str | None]:
    """Get default repo_id and token from environment."""
    repo_id = os.getenv("HF_REPO_ID", "czlowiekZplanety/bettip-data")
    token = os.getenv("HF_TOKEN")
    return repo_id, token


def download_file(
    filename: str,
    local_dir: str = ".",
    repo_id: str | None = None,
    token: str | None = None,
    **kwargs: Any,
) -> str:
    """hf_hub_download with retry.

    Args:
        filename: Path within the repo to download.
        local_dir: Local directory to download into.
        repo_id: HF repo ID (defaults to env HF_REPO_ID).
        token: HF token (defaults to env HF_TOKEN).
        **kwargs: Extra kwargs passed to hf_hub_download.

    Returns:
        Local path to the downloaded file.
    """
    from huggingface_hub import hf_hub_download

    default_repo, default_token = _get_defaults()
    repo_id = repo_id or default_repo
    token = token or default_token

    return hf_retry(
        hf_hub_download,
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        token=token,
        local_dir=local_dir,
        **kwargs,
    )


def download_snapshot(
    patterns: list[str],
    local_dir: str = ".",
    repo_id: str | None = None,
    token: str | None = None,
    **kwargs: Any,
) -> str:
    """snapshot_download with retry.

    Args:
        patterns: File patterns to download (allow_patterns).
        local_dir: Local directory to download into.
        repo_id: HF repo ID (defaults to env HF_REPO_ID).
        token: HF token (defaults to env HF_TOKEN).
        **kwargs: Extra kwargs passed to snapshot_download.

    Returns:
        Local path to the downloaded snapshot.
    """
    from huggingface_hub import snapshot_download

    default_repo, default_token = _get_defaults()
    repo_id = repo_id or default_repo
    token = token or default_token

    return hf_retry(
        snapshot_download,
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=patterns,
        token=token,
        **kwargs,
    )


def upload_file(
    path_or_fileobj: str,
    path_in_repo: str,
    repo_id: str | None = None,
    token: str | None = None,
    **kwargs: Any,
) -> Any:
    """HfApi.upload_file with retry.

    Args:
        path_or_fileobj: Local file path to upload.
        path_in_repo: Destination path within the repo.
        repo_id: HF repo ID (defaults to env HF_REPO_ID).
        token: HF token (defaults to env HF_TOKEN).
        **kwargs: Extra kwargs passed to HfApi.upload_file.

    Returns:
        The upload result from HfApi.
    """
    from huggingface_hub import HfApi

    default_repo, default_token = _get_defaults()
    repo_id = repo_id or default_repo
    token = token or default_token

    api = HfApi(token=token)
    return hf_retry(
        api.upload_file,
        path_or_fileobj=path_or_fileobj,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        **kwargs,
    )


def upload_folder(
    folder_path: str,
    path_in_repo: str,
    repo_id: str | None = None,
    token: str | None = None,
    **kwargs: Any,
) -> Any:
    """HfApi.upload_folder with retry.

    Args:
        folder_path: Local folder path to upload.
        path_in_repo: Destination path within the repo.
        repo_id: HF repo ID (defaults to env HF_REPO_ID).
        token: HF token (defaults to env HF_TOKEN).
        **kwargs: Extra kwargs passed to HfApi.upload_folder.

    Returns:
        The upload result from HfApi.
    """
    from huggingface_hub import HfApi

    default_repo, default_token = _get_defaults()
    repo_id = repo_id or default_repo
    token = token or default_token

    api = HfApi(token=token)
    return hf_retry(
        api.upload_folder,
        folder_path=folder_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        **kwargs,
    )
