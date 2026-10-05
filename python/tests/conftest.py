"""python/tests/conftest.py: shared fixtures for the toks package tests.

The tests import the INSTALLED toks (a wheel): run them from the repository root with
    uv run --no-project --with build/wheels/<wheel> --with pytest --with tokenizers==0.23.2 pytest python/tests
Tokenizer files come from $TOKS_TOKENIZER_CACHE (default ~/.cache/toks/tokenizers, tools/corpora/fetch_tokenizers.py);
a missing file is a skip, never a pass.
"""
import json
import os
import sys

import pytest

import toks

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOKENIZERS = os.path.expanduser(os.environ.get("TOKS_TOKENIZER_CACHE", "~/.cache/toks/tokenizers"))
REPORT = os.environ.get("TOKS_PY_REPORT")       # a json file the parity tests add their counts to

sys.path.append(os.path.join(ROOT, "python"))  # toks_oracle; appended, so `toks` stays the installed wheel


def tok_path(name: str) -> str:
    p = os.path.join(TOKENIZERS, name)
    if not os.path.isfile(p):
        pytest.skip(f"{p} is missing (tools/corpora/fetch_tokenizers.py)")
    return p


def report(key: str, counts: dict) -> None:
    if not REPORT:
        return
    data = {}
    if os.path.exists(REPORT):
        with open(REPORT) as f:
            data = json.load(f)
    data[key] = counts
    with open(REPORT, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)


def pytest_report_header(config):
    return [f"toks {toks.__version__} from {os.path.dirname(toks.__file__)}", f"tokenizer files: {TOKENIZERS}"]


@pytest.fixture(scope="session")
def gpt2_path():
    return tok_path("gpt2")


@pytest.fixture(scope="session")
def gpt2(gpt2_path):
    return toks.Tokenizer.from_file(gpt2_path)
