# toks build. `make test` is the gate on every host; CI runs it on Blacksmith (docs/ci.md).
#
#   make                    library (static + shared) and test programs for the host, in build/<os>-<isa>/
#   make test               build and run every test program
#   make test-guard         the same programs in the guard geometry, runs 1 and 2 (docs/testing.md)
#   make TARGET=<triple>    cross build; x86_64-apple-darwin runs under rosetta on apple silicon
#   make asm TARGET=<t>     assemble every kernel for a target without its sysroot (macro-layer check)
#   make fuzz               the libFuzzer harnesses (tests/fuzz/, docs/fuzz.md): build only, lab hosts
#   BUILD_DIR=...           override the output directory (parallel worktrees / lanes)
#   tools/remote.sh <host> make -j test    the same on a benchmark machine (docs/machines.md)

ifeq ($(origin CC),default)
  CC      := clang
endif
ifeq ($(origin AR),default)
  AR      := $(if $(shell command -v llvm-ar 2>/dev/null),llvm-ar,ar)
endif
HOST      := $(shell $(CC) -dumpmachine)
TARGET    ?= $(HOST)

ifneq ($(filter aarch64% arm64%,$(TARGET)),)
  ISA := arm64
else ifneq ($(filter x86_64% amd64%,$(TARGET)),)
  ISA := x86_64
else
  $(error toks: unsupported TARGET $(TARGET))
endif
ifneq ($(findstring apple,$(TARGET))$(findstring darwin,$(TARGET)),)
  OS := macos
else ifneq ($(findstring windows,$(TARGET))$(findstring mingw,$(TARGET)),)
  OS := windows
else
  OS := linux
endif

BUILD_DIR ?= build/$(OS)-$(ISA)
TFLAGS    := $(if $(filter $(HOST),$(TARGET)),,--target=$(TARGET))

# SPEC §9's strict subset for the library; tests are c17 + warnings without the conversion pedantry.
# -fno-builtin-strlen: clang's loop-idiom pass would turn a `while (s[n]) n++` into a libc strlen call; -fno-builtin-bcmp:
# on linux it would spell memcmp(...) == 0 as bcmp (§9: libc is memcpy / memset / memcmp; tests/abi/cf_audit.py checks
# the objects).
CSTRICT   := -std=c17 -O3 -fno-strict-aliasing -fwrapv -Wall -Wextra -Wconversion -Wsign-conversion -Werror \
             -fno-builtin-strlen -fno-builtin-bcmp
CTEST     := -std=c17 -O2 -fno-strict-aliasing -fwrapv -Wall -Wextra -Werror
CPPFLAGS  := -Iinclude -Isrc/core -Isrc/platform
# GUARD (1 or 2) is set only by test-guard below: the guard build of the library and the tests (core.h toks_tab)
ifneq ($(GUARD),)
  CPPFLAGS += -DTOKS_GUARD=$(GUARD)
endif
ifeq ($(OS),linux)
  PIC := -fPIC
  ifeq ($(ISA),x86_64)
    HARDEN := -fcf-protection=full
  else
    HARDEN := -mbranch-protection=bti
  endif
else ifeq ($(OS),macos)
  PIC := -fPIC
else ifeq ($(OS),windows)
  # the ucrt marks fopen / getenv / sscanf deprecated ("use fopen_s"): a msvc-only lint that -Werror makes fatal
  CPPFLAGS += -D_CRT_SECURE_NO_WARNINGS
endif
# threads exist only in src/par (toks_par, SPEC §2.4); the core is compiled without them
ifneq ($(OS),windows)
  PTHREAD := -pthread
endif
$(BUILD_DIR)/obj/src/par/%.o: XFLAGS := $(PTHREAD)

SRC_C     := $(wildcard src/core/*.c src/platform/*.c src/gen/*.c src/par/*.c)
SRC_S     := $(wildcard src/asm/$(ISA)/*.S)
OBJ_C     := $(patsubst %.c,$(BUILD_DIR)/obj/%.o,$(SRC_C))
OBJ_S     := $(patsubst %.S,$(BUILD_DIR)/obj/%.o,$(SRC_S))
OBJS      := $(OBJ_C) $(OBJ_S)

# Every kernel source joins its dispatch (src/core/kernels.h): src/asm/<isa>/<name>.S defines TOKS_HAVE_<NAME>=1,
# the name upper-cased (k3_cl100k_neon.S -> TOKS_HAVE_K3_CL100K_NEON, k5_avx2.S -> TOKS_HAVE_K5_AVX2); the tests also
# get the source list (TOKS_ASM_SOURCES: test_tier checks every kernel in it is wired). HAVE_FILE holds both and is
# rewritten only when they change; the objects, libraries and tests depend on it, so adding or removing a kernel
# rebuilds the dispatch instead of linking a kernel that nothing calls.
HAVE      := $(addprefix -DTOKS_HAVE_,$(addsuffix =1,$(shell echo $(basename $(notdir $(filter src/asm/$(ISA)/k%,$(SRC_S)))) | tr a-z A-Z)))
ASM_LIST  := $(notdir $(SRC_S))
HAVE_FILE := $(BUILD_DIR)/have.txt
ifneq ($(shell cat $(HAVE_FILE) 2>/dev/null),$(strip $(HAVE) $(ASM_LIST)))
  $(shell mkdir -p $(BUILD_DIR) && echo '$(strip $(HAVE) $(ASM_LIST))' > $(HAVE_FILE))
endif

LIB       := $(BUILD_DIR)/libtoks.a
ifeq ($(OS),macos)
  SHLIB   := $(BUILD_DIR)/libtoks.dylib
  SHFLAGS := -dynamiclib
else
  SHLIB   := $(BUILD_DIR)/libtoks.so
  SHFLAGS := -shared
endif

TEST_COMMON := tests/common/guard.c tests/common/abicheck_$(ISA).S
TEST_SRCS   := $(wildcard tests/c/*.c)
TEST_BINS   := $(patsubst tests/c/%.c,$(BUILD_DIR)/tests/%,$(TEST_SRCS))
# test_bpe once more per asm tier with K5 / K6 kernels (src/asm/<isa>/k5_<tier>.S, k6_<tier>.S): the same checks
# against that tier's asm in place of the c twins; the binary skips on a cpu without the tier
BPE_TIERS   := $(sort $(patsubst k5_%,%,$(patsubst k6_%,%,$(basename $(filter k5_%.S k6_%.S,$(ASM_LIST))))))
BPE_BINS    := $(BPE_TIERS:%=$(BUILD_DIR)/tests/test_bpe_%)
TEST_BINS   += $(BPE_BINS)

.PHONY: all lib test test-guard guard-run asm asmcheck size fuzz clean
all: lib $(TEST_BINS)
lib: $(LIB) $(SHLIB)

# size: SPEC T10's budgets (tools/size.sh): source lines per budget, code and generated-table bytes of this build
size: $(OBJS)
	@tools/size.sh $(BUILD_DIR)

$(HAVE_FILE):
	@mkdir -p $(dir $@)
	echo '$(strip $(HAVE) $(ASM_LIST))' > $@

$(BUILD_DIR)/obj/%.o: %.c $(HAVE_FILE) Makefile
	@mkdir -p $(dir $@)
	$(CC) $(TFLAGS) $(CSTRICT) $(PIC) $(HARDEN) $(XFLAGS) -fvisibility=hidden $(CPPFLAGS) $(HAVE) -MMD -MP -c $< -o $@

$(BUILD_DIR)/obj/%.o: %.S
	@mkdir -p $(dir $@)
	$(CC) $(TFLAGS) $(PIC) $(CPPFLAGS) -Isrc/asm/$(ISA) -MMD -MP -c $< -o $@

$(LIB): $(OBJS) $(HAVE_FILE)
	@mkdir -p $(dir $@)
	rm -f $@
	$(AR) rcs $@ $(OBJS)

$(SHLIB): $(OBJS) $(HAVE_FILE)
	$(CC) $(TFLAGS) $(SHFLAGS) $(HARDEN) -o $@ $(OBJS) $(PTHREAD)

$(BUILD_DIR)/tests/%: tests/c/%.c $(TEST_COMMON) $(LIB) $(wildcard tests/c/*.inc)
	@mkdir -p $(dir $@)
	$(CC) $(TFLAGS) $(CTEST) $(HARDEN) $(CPPFLAGS) $(HAVE) '-DTOKS_ASM_SOURCES="$(ASM_LIST)"' -Itests/common -o $@ $< $(TEST_COMMON) $(LIB) $(PTHREAD)

$(BPE_BINS): $(BUILD_DIR)/tests/test_bpe_%: tests/c/test_bpe.c $(TEST_COMMON) $(LIB) $(wildcard tests/c/*.inc)
	@mkdir -p $(dir $@)
	$(CC) $(TFLAGS) $(CTEST) $(HARDEN) $(CPPFLAGS) $(HAVE) '-DTOKS_ASM_SOURCES="$(ASM_LIST)"' \
	    $(if $(filter k6_$*.S,$(ASM_LIST)),-DK6_BPE=toks_k6_bpe_$*) $(if $(filter k5_$*.S,$(ASM_LIST)),-DK5_ENCODE=toks_k5_encode_$*) \
	    '-DTEST_BPE_TIER="$*"' -DTEST_BPE_FEAT=TOKS_FEAT_$(shell echo $* | tr a-z A-Z)_TIER \
	    -Itests/common -o $@ $< $(TEST_COMMON) $(LIB) $(PTHREAD)

# test: each test binary runs as its own target, side by side under make -j, its output and exit status kept
# beside it, one pair per tier (TOKS_TIER unset is auto), so two tiers can run at once in one build dir; then
# every output in TEST_BINS order under its "== <binary>" header, and the failures listed: every binary runs either
# way. The run targets are phony (the tier and the tokenizer files are inputs make cannot see); a run removes its
# old status first, so a binary still running (or hung) has an .out and no .rc.
TEST_TIER := $(or $(TOKS_TIER),auto)
TEST_RUNS := $(TEST_BINS:%=%.run)
TEST_SAYS  = @f=; for t in $(TEST_BINS); do echo "== $$t"; cat $$t.$(TEST_TIER).out; \
	    [ "$$(cat $$t.$(TEST_TIER).rc 2>/dev/null)" = 0 ] || f="$$f $${t\#\#*/}"; done; \
	[ -z "$$f" ] || { echo "make $@ ($(TEST_TIER)): FAIL:$$f"; exit 1; }
.PHONY: $(TEST_RUNS)
test: asmcheck size $(TEST_RUNS)
	$(TEST_SAYS)

$(TEST_RUNS): %.run: %
	@rm -f $<.$(TEST_TIER).rc; $< > $<.$(TEST_TIER).out 2>&1; echo $$? > $<.$(TEST_TIER).rc
# test_stall times its input classes against en's ns per byte: it runs after every other binary of its make and
# asmcheck / size (listed first, so they start first and are long done), so their load cannot read as a stall
TEST_TIMED := $(BUILD_DIR)/tests/test_stall.run
$(TEST_TIMED): $(filter-out $(TEST_TIMED),$(TEST_RUNS)) $(if $(GUARD),,asmcheck size)

# test-guard: the guard geometry (docs/testing.md), the shipped library's tables and scratch regions each on its own
# pages: run 1 with a no-access page flush against every one's end, run 2 against its start, each the library and
# every test program built in its own directory (-DTOKS_GUARD), then run as make test runs them (TOKS_TIER as there).
# The object audits stay make test's: they read the shipped objects
test-guard:
	@$(MAKE) GUARD=1 BUILD_DIR=$(BUILD_DIR)-guard1 guard-run; r1=$$?; \
	 $(MAKE) GUARD=2 BUILD_DIR=$(BUILD_DIR)-guard2 guard-run; r2=$$?; \
	 echo "make test-guard ($(TEST_TIER)): run 1 $$([ $$r1 = 0 ] && echo pass || echo FAIL), run 2 $$([ $$r2 = 0 ] && echo pass || echo FAIL)"; \
	 [ $$r1 = 0 ] && [ $$r2 = 0 ]
guard-run: $(TEST_RUNS)
	$(TEST_SAYS)

asm: $(OBJ_S)

# fuzz: the libFuzzer harnesses (tests/fuzz/Makefile, docs/fuzz.md), build only; never part of test (clang with
# libFuzzer: the lab hosts' LLVM). Campaigns: tests/fuzz/run.sh on a lab host.
fuzz:
	$(MAKE) -f tests/fuzz/Makefile

# asmcheck (part of test): every asm source, plus the macro layer's own cases in tests/asm/<isa>, assembles for
# all three object formats of both isas. Mach-o, elf and coff each have their own local-label and cpu-extension
# rules (asm.h: LOCAL, .arch_extension), so one format passing proves nothing about the others. Then the abi lint
# (tests/abi/asm_regs_audit.py): in the x86_64 coff objects every callee-saved register a function names (win64:
# rbx rbp rdi rsi r12-r15 xmm6-15) is saved by its unwind codes; arm64 sources keep PROLOGUE / LEAF's contract.
# Then SPEC §9 / §4.1 on the built objects (tests/abi/cf_audit.py, docs/hardening.md): no indirect call, no
# address-taken function, no libc but memcpy / memset / memcmp, no recursion, no vla, nothing after load reaching
# the platform layer or libc; its teeth are tests/abi/cf_teeth.c, compiled with the library's flags.
ASM_TRIPLES_arm64  := aarch64-apple-darwin aarch64-linux-gnu aarch64-windows-msvc
ASM_TRIPLES_x86_64 := x86_64-apple-darwin x86_64-linux-gnu x86_64-pc-windows-msvc
define ASMCHECK_TRIPLE
$(BUILD_DIR)/asmcheck/$(2)/%.o: %.S
	@mkdir -p $$(dir $$@)
	$(CC) --target=$(2) $(CPPFLAGS) -Isrc/asm/$(1) -MMD -MP -c $$< -o $$@
ASMCHECK += $$(patsubst %.S,$(BUILD_DIR)/asmcheck/$(2)/%.o,$$(wildcard src/asm/$(1)/*.S tests/asm/$(1)/*.S))
endef
$(foreach i,arm64 x86_64,$(foreach t,$(ASM_TRIPLES_$(i)),$(eval $(call ASMCHECK_TRIPLE,$(i),$(t)))))
asmcheck: $(ASMCHECK) $(OBJS)
	@echo "asmcheck: $(words $(ASMCHECK)) objects (every .S x mach-o, elf, coff) assemble"
	@python3 tests/abi/asm_regs_audit.py $(BUILD_DIR)/asmcheck $(firstword $(CC))
	@python3 tests/abi/cf_audit.py $(BUILD_DIR) $(CC) $(TFLAGS) $(CSTRICT) $(PIC) $(HARDEN) $(CPPFLAGS)

clean:
	rm -rf build

-include $(OBJS:.o=.d) $(ASMCHECK:.o=.d)
