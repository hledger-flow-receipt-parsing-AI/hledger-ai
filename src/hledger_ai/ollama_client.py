"""Thin wrapper around the Ollama REST API."""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, Dict, Optional

import requests
from typeguard import typechecked

log = logging.getLogger(__name__)


@typechecked
class OllamaClient:
    """Minimal Ollama client using the REST API.

    No ``ollama`` pip package required — the HTTP API is stable and
    trivial.
    """

    def __init__(
        self, *, base_url: str = "http://localhost:11434"
    ) -> None:
        # PR-1: receipts are only sent to a server on this machine.
        from hledger_ai.privacy import assert_local_url

        assert_local_url(url=base_url)
        self._base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the Ollama server is reachable."""
        try:
            r = requests.get(
                f"{self._base_url}/", timeout=3
            )
            return r.status_code == 200
        except requests.ConnectionError:
            return False

    def ensure_model(self, *, model: str) -> bool:
        """Return True if *model* is already pulled locally."""
        try:
            r = requests.get(
                f"{self._base_url}/api/tags", timeout=5
            )
            r.raise_for_status()
            models = r.json().get("models", [])
            return any(m.get("name", "").startswith(model) for m in models)
        except (requests.ConnectionError, requests.HTTPError):
            return False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate_structured(
        self,
        *,
        model: str,
        prompt: str,
        image_path: Optional[str] = None,
        schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Call Ollama with structured-output (JSON schema enforcement).

        Returns the parsed JSON dict.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": schema,
            "options": {
                "num_predict": 2048,
                "temperature": 0,
                "seed": 42,
            },
        }

        if image_path is not None:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            payload["images"] = [img_b64]

        log.debug("Ollama request: model=%s", model)
        r = requests.post(
            f"{self._base_url}/api/generate",
            json=payload,
            timeout=120,
        )
        r.raise_for_status()

        body = r.json()
        response_text = body.get("response", "")

        # Qwen3 models use "thinking" mode by default — the
        # structured JSON may end up in the ``thinking`` field
        # with ``response`` left empty.
        if not response_text.strip():
            thinking = body.get("thinking", "")
            if thinking.strip():
                log.debug(
                    "response empty; using thinking field instead"
                )
                response_text = thinking

        if not response_text.strip():
            log.warning(
                "Ollama returned empty response for model=%s",
                model,
            )
            return {}

        return json.loads(response_text)

    def generate_text(
        self,
        *,
        model: str,
        prompt: str,
    ) -> str:
        """Plain text generation (no schema enforcement)."""
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "seed": 42,
            },
        }
        r = requests.post(
            f"{self._base_url}/api/generate",
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        body = r.json()
        text = body.get("response", "")

        # Qwen3 thinking mode: output may be in ``thinking``
        if not text.strip():
            thinking = body.get("thinking", "")
            if thinking.strip():
                text = thinking

        return text
