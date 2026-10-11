#!/bin/sh
# tests/proof/install.sh: user-local Frama-C for the proof package (SPEC §14.1, docs/proof.md). Lab hosts only
# (linux aarch64 or x86_64, Ubuntu 24.04), never the control-plane mac:
#
#   tools/remote.sh <gb10 host> 'nice -n 10 taskset -c 10-13 sh tests/proof/install.sh'
#
# No sudo: the gmp headers zarith needs come from the distribution's libgmp-dev deb, unpacked with dpkg-deb -x (the
# runtime libgmp.so.10 is always present); opam is the release binary and runs with depexts and sandboxing off.
# Everything lands under $TOKS_PROOF_TOOLS (default ~/toks-ci/opam), outside every worktree, so one install serves
# every branch on the host. Idempotent: finished steps are skipped. Writes $TOKS_PROOF_TOOLS/env.sh (tests/proof/
# eva.sh sources it) and $TOKS_PROOF_TOOLS/versions.txt (the tools, the opam repository and every package of the
# switch at its version: the receipt eva.sh prints). The opam binary is checked against the release's sha256.
set -eu
OPAM_VERSION=2.6.0
OCAML_VERSION=4.14.2
FRAMAC_VERSION=33.0
JOBS=${JOBS:-4}                       # shared hosts: <= 8 processes (opam builds JOBS packages x JOBS jobs)

T=${TOKS_PROOF_TOOLS:-$HOME/toks-ci/opam}
mkdir -p "$T/dl" "$T/bin"
case $(uname -m) in        # opam_sha: sha256 of the opam-$OPAM_VERSION-<arch>-linux release asset
  aarch64|arm64) deb_arch=arm64; opam_arch=arm64; triplet=aarch64-linux-gnu
                 opam_sha=aeaeb4294a9abaa7d37844d9138230125933c648e631da2eec888b5e4ce55bde ;;
  x86_64)        deb_arch=amd64; opam_arch=x86_64; triplet=x86_64-linux-gnu
                 opam_sha=a59184447f881005dae70b2ae455c3a7e9549834a41c635c49a5a28235eca758 ;;
  *) echo "install.sh: unsupported machine $(uname -m)" >&2; exit 2 ;;
esac

# 1. gmp headers for zarith.
if [ ! -f "$T/sysroot/usr/include/$triplet/gmp.h" ] && [ ! -f "$T/sysroot/usr/include/gmp.h" ]; then
  (cd "$T/dl" && apt-get download "libgmp-dev:$deb_arch")
  dpkg-deb -x "$T"/dl/libgmp-dev_*_"$deb_arch".deb "$T/sysroot"
  ln -sf "/usr/lib/$triplet/libgmp.so.10" "$T/sysroot/usr/lib/$triplet/libgmp.so"
fi

# 2. opam, the switch, frama-c.
if [ ! -x "$T/bin/opam" ]; then
  curl -fsSL -o "$T/dl/opam" "https://github.com/ocaml/opam/releases/download/$OPAM_VERSION/opam-$OPAM_VERSION-$opam_arch-linux"
  echo "$opam_sha  $T/dl/opam" | sha256sum -c - >/dev/null || { echo "install.sh: opam's sha256 is not the release's" >&2; exit 1; }
  mv "$T/dl/opam" "$T/bin/opam" && chmod +x "$T/bin/opam"
fi
echo "$opam_sha  $T/bin/opam" | sha256sum -c - >/dev/null || { echo "install.sh: $T/bin/opam is not the release's" >&2; exit 1; }
cat > "$T/env.sh" <<EOF
export TOKS_PROOF_TOOLS="$T"
export OPAMROOT="$T/root" OPAMYES=1 OPAMCOLOR=never OPAMJOBS=$JOBS OPAMNODEPEXTS=1 OPAMNOENVNOTICE=1
export C_INCLUDE_PATH="$T/sysroot/usr/include/$triplet:$T/sysroot/usr/include\${C_INCLUDE_PATH:+:\$C_INCLUDE_PATH}"
export LIBRARY_PATH="$T/sysroot/usr/lib/$triplet\${LIBRARY_PATH:+:\$LIBRARY_PATH}"
export PATH="$T/root/toks/bin:$T/bin:\$PATH"
EOF
. "$T/env.sh"
[ -f "$OPAMROOT/config" ] || opam init --bare --disable-sandboxing -n
opam switch list -s 2>/dev/null | grep -qx toks || opam switch create toks "ocaml-base-compiler.$OCAML_VERSION"
opam option --global depext=false
# conf-graphviz is a post-dependency that only checks for `dot` (call-graph rendering, unused by eva).
opam list -i -s --switch toks conf-graphviz 2>/dev/null | grep -q conf-graphviz || opam install --switch toks --fake conf-graphviz
command -v frama-c >/dev/null 2>&1 || opam install --switch toks "frama-c.$FRAMAC_VERSION"

{
  echo "host: $(uname -srm)"
  echo "opam: $(opam --version)"
  echo "ocaml: $(opam exec --switch toks -- ocamlc -version)"
  echo "frama-c: $(frama-c -version)"
  echo "gcc (frama-c's preprocessor): $(gcc -dumpfullversion)"
  echo "opam binary sha256: $opam_sha"
  echo "opam repository: $(opam repo list --switch toks --short 2>/dev/null | tr '\n' ' ')$(opam repo list --all 2>/dev/null | awk '$1 == "default" { print $2 }')"
  echo "the switch, every package at its version:"
  opam list --switch toks --installed --columns name,version --short 2>/dev/null | sed 's/^/  /'
} > "$T/versions.txt"
cat "$T/versions.txt"
