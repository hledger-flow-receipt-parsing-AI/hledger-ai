"""The qvac runtime: a Node sidecar spoken to over stdin/stdout (HW-8)."""

import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import pytest

from hledger_ai.ocr.qvac_engine import QvacOcrEngine
from hledger_ai.runtimes.qvac_client import QvacClient, QvacError

# A stand-in sidecar: speaks the same line protocol as qvac_sidecar.mjs and
# prints the kind of log noise the real SDK writes to stdout.
FAKE_SIDECAR = textwrap.dedent("""
    import json, sys
    MARK = "@@hledger-ai@@ "
    print("[QVACRegistryClient] [INFO] Blob download complete", flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request["op"] == "close":
            break
        print("some log line {not json}", flush=True)
        if request["op"] == "ocr" and "broken" in request["image"]:
            reply = {"id": request["id"], "error": "cannot decode image"}
        elif request["op"] == "ocr":
            reply = {"id": request["id"], "blocks": [
                {"text": "TOTAAL", "bbox": [36, 392, 136, 420], "confidence": 0.99},
                {"text": "11,54", "bbox": [226, 394, 310, 423], "confidence": 0.7},
            ]}
        else:
            reply = {"id": request["id"], "ok": True}
        print(MARK + json.dumps(reply), flush=True)
    """)


@pytest.fixture
def client(tmp_path: Path):
    script = tmp_path / "fake_sidecar.py"
    script.write_text(FAKE_SIDECAR)
    client = QvacClient(
        qvac_dir=str(tmp_path), command=[sys.executable, str(script)]
    )
    yield client
    client.close()


def test_ocr_reply_is_found_between_the_sdks_own_log_lines(client) -> None:
    blocks = client.ocr(image_path="/photos/receipt.jpg")

    assert [b["text"] for b in blocks] == ["TOTAAL", "11,54"]


def test_one_sidecar_serves_many_requests(client) -> None:
    first = client.ocr(image_path="/photos/a.jpg")
    second = client.ocr(image_path="/photos/b.jpg")

    assert first == second
    assert client.starts == 1


def test_error_from_the_sidecar_becomes_an_exception(client) -> None:
    with pytest.raises(QvacError, match="cannot decode image"):
        client.ocr(image_path="/photos/broken.jpg")


def test_sidecar_that_dies_is_reported_not_waited_for(tmp_path: Path) -> None:
    script = tmp_path / "dies.py"
    script.write_text("import sys; sys.exit(3)\n")
    client = QvacClient(
        qvac_dir=str(tmp_path), command=[sys.executable, str(script)]
    )

    with pytest.raises(QvacError, match="stopped"):
        client.ocr(image_path="/photos/a.jpg")


def test_missing_installation_is_reported_as_not_installed(
    tmp_path: Path,
) -> None:
    client = QvacClient(qvac_dir=str(tmp_path / "nothing-here"))

    assert client.is_installed() is False
    with pytest.raises(QvacError, match="not installed"):
        client.ocr(image_path="/photos/a.jpg")


def test_engine_turns_word_blocks_into_ocr_rows(client, tmp_path: Path) -> None:
    from PIL import Image

    image = tmp_path / "receipt.png"
    Image.new("RGB", (600, 900), "white").save(image)
    engine = QvacOcrEngine(engine_id="qvac-ocr-latin", client=client)

    result = engine.read(image_path=str(image), image_sha256="abc")

    assert result.engine == "qvac-ocr-latin"
    assert result.row_texts() == ["TOTAAL 11,54"]
    assert (result.width, result.height) == (600, 900)
    assert result.lines[1].score == 0.7


def test_inference_runs_without_network_when_the_sandbox_exists(
    tmp_path: Path,
) -> None:
    client = QvacClient(qvac_dir=str(tmp_path), offline=True)

    command = client.command_line(sandbox_available=True)
    unsandboxed = client.command_line(sandbox_available=False)

    assert command[:2] == ["unshare", "-rn"]
    assert command[-1].endswith("hledger_sidecar.mjs")
    assert unsandboxed[0] != "unshare"


@pytest.mark.slow
def test_real_qvac_reads_a_printed_receipt(tmp_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    from hledger_ai.runtimes.qvac_client import DEFAULT_QVAC_DIR

    real = QvacClient(qvac_dir=DEFAULT_QVAC_DIR)
    if not real.is_installed():
        pytest.skip("qvac is not installed")
    image = Image.new("RGB", (600, 300), "white")
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 26
    )
    ImageDraw.Draw(image).text(
        (40, 40), "TOTAAL      11,54", fill="black", font=font
    )
    path = tmp_path / "receipt.png"
    image.save(path)
    try:
        result = QvacOcrEngine(engine_id="qvac-ocr-latin", client=real).read(
            image_path=str(path), image_sha256="abc"
        )
    finally:
        real.close()

    assert "TOTAAL" in result.row_texts()[0]


def test_install_is_refused_before_downloading_when_the_disk_is_too_small(
    tmp_path: Path,
) -> None:
    from hledger_ai.runtimes.qvac_client import install_qvac

    commands: List[List[str]] = []
    lines: List[str] = []

    done = install_qvac(
        qvac_dir=str(tmp_path / "qvac"),
        disk_free_gb=2.0,
        needed_gb=3.5,
        run=lambda command, **_: commands.append(command),
        out=lines.append,
    )

    assert done is False
    assert commands == []
    assert "3.5" in lines[0] and "2.0" in lines[0]


def test_install_fetches_the_pinned_sdk_into_the_shared_folder(
    tmp_path: Path,
) -> None:
    from hledger_ai.runtimes.qvac_client import install_qvac

    commands: List[List[str]] = []

    done = install_qvac(
        qvac_dir=str(tmp_path / "qvac"),
        disk_free_gb=100.0,
        needed_gb=3.5,
        run=lambda command, **_: commands.append(command),
        out=lambda line: None,
        download_models=False,
    )

    assert done is True
    assert commands == [
        [
            "npm",
            "install",
            "--prefix",
            str(tmp_path / "qvac"),
            "@qvac/sdk@0.24.0",
        ]
    ]
    assert (tmp_path / "qvac" / "package.json").exists()
