"""python/setup.py: the toks extension module (PEP 517 via setuptools; pyproject.toml has the metadata).

    uv build --wheel python --out-dir build/wheels          # from the repository root

The library is built by the repository's own Makefile (the same objects `make` builds: c core + this ISA's
asm kernels, its flags, its tier dispatch) into build/python/<platform>/libtoks.a and linked into the
extension statically, so the wheel is self-contained. Only PyInit__toks is exported: the toks_* symbols
stay inside the module (Linux --exclude-libs, macOS -exported_symbol), whatever else the process loads.
"""
import os
import platform
import re
import subprocess
import sys
import sysconfig

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# PEP 639 license-files are read relative to this directory, so the repository's texts are copied here first
# (the copies are gitignored); the wheel then carries them under toks-*.dist-info/licenses/.
for _name in ("LICENSE", "LICENSING.md", "THIRD_PARTY_NOTICES.md"):
    _src = os.path.join(ROOT, _name)
    if os.path.isfile(_src):
        with open(_src, "rb") as _f:
            _data = _f.read()
        with open(os.path.join(HERE, _name), "wb") as _f:
            _f.write(_data)


def _version():
    """TOKS_VERSION of include/toks.h: the wheel's version is the library's, never a second number"""
    with open(os.path.join(ROOT, "include", "toks.h"), encoding="utf-8") as f:
        m = re.search(r'^#define TOKS_VERSION +"([0-9]+\.[0-9]+\.[0-9]+)"', f.read(), re.M)
    if m is None:
        raise RuntimeError("toks: no TOKS_VERSION in include/toks.h")
    return m.group(1)


def _isa():
    m = platform.machine().lower()
    return "arm64" if m in ("arm64", "aarch64") else "x86_64" if m in ("x86_64", "amd64") else m


class BuildExt(build_ext):
    def build_extension(self, ext):
        if not os.path.isfile(os.path.join(ROOT, "include", "toks.h")):
            raise RuntimeError(f"toks: build the wheel from the toks repository (no include/toks.h under {ROOT})")
        plat = sysconfig.get_platform().replace("-", "_").replace(".", "_")
        bdir = f"build/python/{plat}"
        env = dict(os.environ)
        env.setdefault("CC", "clang")
        if sys.platform == "darwin":
            env.setdefault("MACOSX_DEPLOYMENT_TARGET", sysconfig.get_config_var("MACOSX_DEPLOYMENT_TARGET") or "11.0")
        jobs = str(min(8, os.cpu_count() or 1))
        subprocess.run(["make", "-C", ROOT, "-j", jobs, f"BUILD_DIR={bdir}", f"{bdir}/libtoks.a"], check=True, env=env)
        lib = os.path.join(ROOT, bdir, "libtoks.a")
        ext.extra_objects = [lib]
        ext.depends = [lib]                 # setuptools relinks only when a source or a dependency is newer
        super().build_extension(ext)


os.environ.setdefault("CC", "clang")     # the library's compiler (LLVM) for the module too

cflags = ["-std=c11", "-O3", "-g0", "-fvisibility=hidden", "-Wall", "-Wextra", "-Werror",
          "-Wno-missing-field-initializers"]
ldflags = []
if sys.platform == "darwin":
    ldflags += ["-Wl,-exported_symbol,_PyInit__toks", "-Wl,-dead_strip"]
elif sys.platform.startswith("linux"):
    cflags += ["-fcf-protection=full"] if _isa() == "x86_64" else ["-mbranch-protection=bti"]
    ldflags += ["-Wl,--exclude-libs,ALL", "-Wl,--gc-sections", "-Wl,-z,noexecstack", "-Wl,-z,relro,-z,now"]

setup(
    version=_version(),
    ext_modules=[
        Extension(
            "toks._toks",
            sources=["toks/_toks.c"],
            include_dirs=[os.path.join(ROOT, "include")],
            extra_compile_args=cflags,
            extra_link_args=ldflags,
        )
    ],
    cmdclass={"build_ext": BuildExt},
)
