from __future__ import annotations

import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent
sys.path.insert(0, str(TESTS_DIR))

FIXTURES = TESTS_DIR / "fixtures"

#: Header of a published GGUF model (Qwen2.5-0.5B-Instruct-GGUF, q2_k).  Every
#: metadata value is the real one, byte for byte, including the 2509-character
#: chat template; only the three large vocabulary arrays were truncated and the
#: tensor table shortened so the file can live in the repository.
REAL_GGUF = FIXTURES / "qwen2.5-0.5b-instruct-header.gguf"


@pytest.fixture
def real_gguf() -> Path:
    assert REAL_GGUF.exists(), f"missing fixture {REAL_GGUF}"
    return REAL_GGUF


@pytest.fixture
def qwen_template(real_gguf: Path) -> str:
    from gguf_template_doctor import parse_header

    return parse_header(real_gguf).metadata["tokenizer.chat_template"]
