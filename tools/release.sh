#!/bin/sh
# tools/release.sh: the toks release artifacts of this host (docs/release.md), from the repository root:
#
#   build/release/toks-<version>-<os>-<isa>.tar.gz   include/{toks.h, toks.inc, toks_asm.h}, lib/libtoks.{a,so|dylib},
#                                                    LICENSE, NOTICE, LICENSING.md, THIRD_PARTY_NOTICES.md,
#                                                    MANIFEST (version, commit, compiler, kernels, sha256 of each file)
#   build/wheels/toks-<version>-cp3XX-*.whl          python/build.sh (every CPython, tested)
#
# The library in the tarball is the one `make test` just tested (same BUILD_DIR). On a lab host the tree has no
# .git (tools/remote.sh): pass TOKS_COMMIT=$(git rev-parse HEAD). At a tag, the tag must be v<TOKS_VERSION>.
#
#   tools/remote.sh <host> "TOKS_COMMIT=$(git rev-parse HEAD) sh tools/release.sh"
set -eu
v=$(sed -n 's/^#define TOKS_VERSION  *"\([0-9.]*\)".*/\1/p' include/toks.h)
[ -n "$v" ] || { echo "release: no TOKS_VERSION in include/toks.h" >&2; exit 1; }
commit=${TOKS_COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}
if tag=$(git describe --tags --exact-match 2>/dev/null); then
    [ "$tag" = "v$v" ] || { echo "release: HEAD is tag $tag but toks.h says $v" >&2; exit 1; }
    [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "release: $tag with local changes" >&2; exit 1; }
fi
cc=${CC:-clang}
export UV_PYTHON_PREFERENCE=only-managed            # python only through uv, kept under build/ (python/build.sh)
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$PWD/build/uv-python}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/build/uv-cache}"
if command -v sha256sum >/dev/null 2>&1; then sum=sha256sum; else sum="shasum -a 256"; fi
case "$(uname -s)" in Darwin) os=macos; so=libtoks.dylib ;; *) os=linux; so=libtoks.so ;; esac
case "$(uname -m)" in arm64|aarch64) isa=arm64 ;; *) isa=x86_64 ;; esac
name=toks-$v-$os-$isa
obj=build/release/obj-$os-$isa
out=build/release/$name
rm -rf "$out" "$out.tar.gz"
${TASKSET:-} make -j "${JOBS:-8}" BUILD_DIR="$obj" lib test
mkdir -p "$out/include" "$out/lib"
cp include/toks.h "$out/include/"
uv run -q --no-project --python 3.13 python tools/gen/asm_inc.py "$out/include" "$cc" >/dev/null
cp "$obj/libtoks.a" "$obj/$so" "$out/lib/"
cp LICENSE NOTICE LICENSING.md THIRD_PARTY_NOTICES.md "$out/"
{
    echo "toks $v"
    echo "commit $commit"
    echo "target $os-$isa"
    echo "cc $($cc --version | head -1)"
    echo "kernels $(cat "$obj/have.txt")"
    # the .so's glibc floor: the newest symbol version it binds (libtoks.a binds none until it is linked)
    [ "$os" != linux ] || echo "glibc $(llvm-objdump -T "$obj/$so" | sed -n 's/.*GLIBC_\([0-9.]*\).*/\1/p' | sort -t. -k1,1n -k2,2n | tail -1)"
    (cd "$out" && find include lib LICENSE NOTICE LICENSING.md THIRD_PARTY_NOTICES.md -type f | sort | while read -r f; do
        echo "sha256 $($sum "$f" | cut -d' ' -f1) $f"
    done)
} > "$out/MANIFEST"
COPYFILE_DISABLE=1 tar -C build/release -czf "$out.tar.gz" "$name"   # no macOS ._ xattr files
echo "== $out.tar.gz"
cat "$out/MANIFEST"
[ "${WHEELS:-1}" = 0 ] || sh python/build.sh
