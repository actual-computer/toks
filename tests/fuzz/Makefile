# tests/fuzz/Makefile: the libFuzzer harnesses (docs/fuzz.md; SPEC §14.2, T6, T9). From the repository root:
#
#   make fuzz                                    = make -f tests/fuzz/Makefile (build only; never part of make test)
#   make -f tests/fuzz/Makefile -j8              every harness, asan + ubsan, into build/fuzz-<os>-<isa>/
#   make -f tests/fuzz/Makefile SAN=memory -j8   the msan build (linux), into build/fuzz-msan-<os>-<isa>/
#   make -f tests/fuzz/Makefile SAN=coverage -j8 source-based coverage, for tests/fuzz/run.sh cover
#
# The library is compiled from source with coverage instrumentation (-fsanitize=fuzzer-no-link) and the sanitizers
# at -O1 with SPEC §9's strict flags; the asm kernels of this isa are assembled and dispatched as in `make`
# (TOKS_HAVE_<NAME> per src/asm/<isa>/<name>.S), so TOKS_TIER_AUTO runs them and every harness compares them with
# TOKS_TIER_SCALAR (the instrumented c twins). Needs clang with libFuzzer (LLVM 21.1.8 on the lab hosts; Apple's
# clang has none).

ifeq ($(origin CC),default)
  CC := clang
endif
HOST := $(shell $(CC) -dumpmachine)
ifneq ($(filter aarch64% arm64%,$(HOST)),)
  ISA := arm64
else
  ISA := x86_64
endif
ifneq ($(findstring apple,$(HOST))$(findstring darwin,$(HOST)),)
  OS := macos
else
  OS := linux
endif

SAN ?= address,undefined
ifeq ($(SAN),memory)
  # scalar only: msan cannot see the asm kernels' stores, so every read of a kernel's output would be a report
  FUZZ_DIR ?= build/fuzz-msan-$(OS)-$(ISA)
  SANFLAGS := -fsanitize=memory -fsanitize-memory-track-origins=2
  NOASM    := 1
else ifeq ($(SAN),coverage)
  FUZZ_DIR ?= build/fuzz-cov-$(OS)-$(ISA)
  SANFLAGS := -fprofile-instr-generate -fcoverage-mapping
else
  FUZZ_DIR ?= build/fuzz-$(OS)-$(ISA)
  # every check aborts except alignment: a misaligned caller-array access is reported once per source location and the
  # run goes on (run.sh status / regress count "runtime error" lines as findings)
  SANFLAGS := -fsanitize=$(SAN) -fno-sanitize-recover=all -fsanitize-recover=alignment
endif

LIB_C    := $(wildcard src/core/*.c src/platform/*.c src/gen/*.c src/par/*.c)
LIB_S    := $(if $(NOASM),,$(wildcard src/asm/$(ISA)/*.S))
HAVE     := $(addprefix -DTOKS_HAVE_,$(addsuffix =1,$(shell echo $(basename $(notdir $(filter src/asm/$(ISA)/k%,$(LIB_S)))) | tr a-z A-Z)))
COMMON   := -std=c17 -O1 -g -fno-omit-frame-pointer -fno-strict-aliasing -fwrapv -Wall -Wextra -Werror \
            -Iinclude -Isrc/core -Isrc/platform -pthread
LIBFLAGS := $(COMMON) -Wconversion -Wsign-conversion -fno-builtin-strlen -fno-builtin-bcmp -fvisibility=hidden $(HAVE) $(SANFLAGS) -fsanitize=fuzzer-no-link
LIB_OBJ  := $(patsubst %.c,$(FUZZ_DIR)/obj/%.o,$(LIB_C)) $(patsubst %.S,$(FUZZ_DIR)/obj/%.o,$(LIB_S))
LIB_HDRS := $(wildcard src/core/*.h src/platform/*.h include/*.h)
HDRS     := $(wildcard tests/fuzz/*.h) $(LIB_HDRS)

HARNESSES := encode pieces decode stream par load load_json
BINS      := $(addprefix $(FUZZ_DIR)/fuzz_,$(HARNESSES))
# the arena probe (arprobe.h, docs/fuzz.md §5.7): every src/core object of this build reports each toks_ar_alloc's
# site, bound and end to arprobe.o, linked into every harness; `tests/fuzz/run.sh probe` prints the table
PROBE_OBJ := $(FUZZ_DIR)/obj/tests/fuzz/arprobe.o
$(FUZZ_DIR)/obj/src/core/%.o: LIBFLAGS += -include tests/fuzz/arprobe.h

.PHONY: all clean
.SECONDARY:
all: $(BINS)

$(FUZZ_DIR)/obj/%.o: %.c $(LIB_HDRS) tests/fuzz/arprobe.h tests/fuzz/Makefile
	@mkdir -p $(dir $@)
	$(CC) $(LIBFLAGS) -c $< -o $@

$(FUZZ_DIR)/obj/%.o: %.S
	@mkdir -p $(dir $@)
	$(CC) -Iinclude -Isrc/core -Isrc/asm/$(ISA) -c $< -o $@

$(FUZZ_DIR)/fuzz_%: tests/fuzz/fuzz_%.c $(LIB_OBJ) $(PROBE_OBJ) $(HDRS) tests/fuzz/Makefile
	@mkdir -p $(dir $@)
	$(CC) $(COMMON) $(SANFLAGS) -fsanitize=fuzzer -o $@ $< $(LIB_OBJ) $(PROBE_OBJ)

clean:
	rm -rf $(FUZZ_DIR)
