"""Client for the qvac runtime (HW-8): a Node sidecar that runs models
through the qvac SDK, next to the Ollama runtime.

Requests and replies are JSON lines over the sidecar's stdin and stdout.
Inference runs in a sandbox without network access when the system offers
one (PR-1); only installing and downloading models needs the network.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from typing import Any, Callable, Dict, List, Optional, Sequence

from typeguard import typechecked

log = logging.getLogger(__name__)

# The SDK itself holds no private data, so all finance roots share it.
DEFAULT_QVAC_DIR = os.path.expanduser("~/.hledger-ai/runtimes/qvac")
QVAC_SDK = "@qvac/sdk@0.24.0"
MARK = "@@hledger-ai@@ "
_SIDECAR_SOURCE = os.path.join(os.path.dirname(__file__), "qvac_sidecar.mjs")
_SIDECAR_NAME = "hledger_sidecar.mjs"
_SANDBOX = ["unshare", "-rn"]  # new user and network namespace: no network


class QvacError(RuntimeError):
    """The qvac runtime is missing, stopped or reported an error."""


@typechecked
def sandbox_available() -> bool:
    """Whether a process can be started without network access."""
    if shutil.which(_SANDBOX[0]) is None:
        return False
    try:
        return (
            subprocess.run(
                _SANDBOX + ["true"], capture_output=True, timeout=10
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


@typechecked
class QvacClient:
    def __init__(
        self,
        *,
        qvac_dir: str = DEFAULT_QVAC_DIR,
        offline: bool = True,
        command: Optional[Sequence[str]] = None,
    ) -> None:
        """*offline*: start the sidecar without network access (inference).
        *command* replaces the sidecar command line (tests)."""
        self._dir = os.path.expanduser(qvac_dir)
        self._offline = offline
        self._command = list(command) if command else None
        self._process: Optional[subprocess.Popen] = None
        self._next_id = 0
        self.starts = 0

    def is_installed(self) -> bool:
        return self._command is not None or (
            shutil.which("node") is not None
            and os.path.isdir(
                os.path.join(self._dir, "node_modules", "@qvac", "sdk")
            )
        )

    def command_line(self, *, sandbox_available: bool) -> List[str]:
        """How the sidecar is started."""
        if self._command is not None:
            return self._command
        command = ["node", os.path.join(self._dir, _SIDECAR_NAME)]
        if self._offline and sandbox_available:
            return _SANDBOX + command
        return command

    def _start(self) -> subprocess.Popen:
        if not self.is_installed():
            raise QvacError(
                f"qvac is not installed in {self._dir}"
                " (run hledger-ai --install-qvac)"
            )
        if self._command is None:
            shutil.copyfile(
                _SIDECAR_SOURCE, os.path.join(self._dir, _SIDECAR_NAME)
            )
        sandboxed = self._offline and sandbox_available()
        if self._offline and not sandboxed and self._command is None:
            log.warning("No network sandbox available; qvac runs without it")
        self.starts += 1
        return subprocess.Popen(
            self.command_line(sandbox_available=sandboxed),
            cwd=self._dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def request(self, *, op: str, **fields: Any) -> Dict[str, Any]:
        """Send one request and wait for its reply."""
        if self._process is None or self._process.poll() is not None:
            self._process = self._start()
        self._next_id += 1
        message = {"id": self._next_id, "op": op, **fields}
        try:
            assert self._process.stdin and self._process.stdout
            self._process.stdin.write(json.dumps(message) + "\n")
            self._process.stdin.flush()
            for line in self._process.stdout:
                if not line.startswith(MARK):
                    continue  # the SDK's own log output
                reply = json.loads(line[len(MARK) :])
                if reply.get("id") != self._next_id:
                    continue
                if "error" in reply:
                    raise QvacError(str(reply["error"]))
                return reply
        except (BrokenPipeError, OSError) as error:
            raise QvacError(f"qvac sidecar stopped: {error}") from error
        raise QvacError("qvac sidecar stopped before answering")

    def ocr(self, *, image_path: str) -> List[Dict[str, Any]]:
        """Text blocks of an image: ``text``, ``bbox`` (x1, y1, x2, y2) and
        ``confidence`` each."""
        return list(self.request(op="ocr", image=image_path)["blocks"])

    def close(self) -> None:
        if self._process is None or self._process.poll() is not None:
            return
        try:
            assert self._process.stdin
            self._process.stdin.write(json.dumps({"op": "close"}) + "\n")
            self._process.stdin.flush()
            self._process.wait(timeout=20)
        except (OSError, subprocess.SubprocessError):
            self._process.kill()


@typechecked
def install_qvac(
    *,
    qvac_dir: str,
    disk_free_gb: float,
    needed_gb: float,
    out: Callable[[str], None],
    run: Callable[..., Any] = subprocess.run,
    download_models: bool = True,
) -> bool:
    """Install the qvac SDK and its OCR models. Returns False, before
    anything is downloaded, when the disk is too small (HW-6)."""
    if disk_free_gb < needed_gb:
        out(
            f"qvac needs {needed_gb:.1f} GB of disk space; only"
            f" {disk_free_gb:.1f} GB is available. Nothing was installed."
        )
        return False
    folder = os.path.expanduser(qvac_dir)
    os.makedirs(folder, exist_ok=True)
    package = os.path.join(folder, "package.json")
    if not os.path.isfile(package):
        with open(package, "w", encoding="utf-8") as handle:
            json.dump({"name": "hledger-ai-qvac", "type": "module"}, handle)
    run(["npm", "install", "--prefix", folder, QVAC_SDK], check=True)
    if download_models:
        # The only step that needs the network: fetch the model files.
        client = QvacClient(qvac_dir=folder, offline=False)
        try:
            client.request(op="warmup")
        finally:
            client.close()
    out(f"qvac installed in {folder}")
    return True
