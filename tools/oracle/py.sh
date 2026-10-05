#!/bin/sh
# tools/oracle/py.sh: python with the pinned reference stack of docs/models/kimi.md (python 3.12, tiktoken
# 0.14.0, transformers 5.18.0), through uv.   tools/oracle/py.sh tests/parity/oracle_tiktoken.py --self-check
# transformers' remote-code module cache goes under this checkout's build/ (lab hosts: work only under
# ~/toks-ci/<branch>/); on a lab host also point TOKS_KIMI_DIR at build/kimik3 (oracle_tiktoken.py --fetch).
root=$(cd "$(dirname "$0")/../.." && pwd)
HF_MODULES_CACHE="${HF_MODULES_CACHE:-$root/build/hf_modules}"
export HF_MODULES_CACHE
exec uv run -q --python 3.12 --with tiktoken==0.14.0 --with transformers==5.18.0 python "$@"
