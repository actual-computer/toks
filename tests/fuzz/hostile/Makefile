# tests/fuzz/hostile/Makefile: the hostile battery's builds (docs/fuzz.md "hostile"). From the repository
# root, on a lab host (or the control-plane mac for the seconds-scale builds):
#
#   make -f tests/fuzz/hostile/Makefile                    # asan + ubsan, build/hostile-<os>-<isa>/
#   make -f tests/fuzz/hostile/Makefile SAN=               # no sanitizers (a guard build feeds this)
#   make -f tests/fuzz/hostile/Makefile FUZZ_DIR=<dir>     # another output directory
#
# The library is compiled from source exactly as tests/fuzz/Makefile compiles it (the same flags, the
# same asm dispatch, the arena probe), so the sanitizer harnesses and this battery see the same code;
# the one difference is the driver: hostile_driver.c instead of a libFuzzer entry point. Reproducing a
# finding needs no libFuzzer: the driver's argv is the repro.
#
# The guard-geometry build is the guards PR's own (make test-guard there); this file only builds the
# sanitizers' driver. To run the battery under the guard geometry, build the guard library on that
# branch and link the driver against it:
#   git -C <guards worktree> ... make GUARD=1 lib
#   clang -std=c17 -O2 -Iinclude -o build/hostile_guard tests/fuzz/hostile/hostile_driver.c <guard libtoks.a>

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
ifeq ($(SAN),)
  SANFLAGS :=
  FUZZ_DIR ?= build/hostile-$(OS)-$(ISA)
else
  # alignment recovers: a misaligned caller-array access is reported once per source location and the
  # run goes on (the battery hands toks +1..+7 misaligned out buffers on purpose)
  SANFLAGS := -fsanitize=$(SAN) -fno-sanitize-recover=all -fsanitize-recover=alignment
  FUZZ_DIR ?= build/hostile-$(OS)-$(ISA)
endif

LIB_C    := $(wildcard src/core/*.c src/platform/*.c src/gen/*.c src/par/*.c)
LIB_S    := $(wildcard src/asm/$(ISA)/*.S)
HAVE     := $(addprefix -DTOKS_HAVE_,$(addsuffix =1,$(shell echo $(basename $(notdir $(filter src/asm/$(ISA)/k%,$(LIB_S)))) | tr a-z A-Z)))
COMMON   := -std=c17 -O1 -g -fno-omit-frame-pointer -fno-strict-aliasing -fwrapv -Wall -Wextra -Werror \
            -Iinclude -Isrc/core -Isrc/platform -pthread
LIBFLAGS := $(COMMON) -Wconversion -Wsign-conversion -fno-builtin-strlen -fno-builtin-bcmp -fvisibility=hidden $(HAVE) $(SANFLAGS)
LIB_OBJ  := $(patsubst %.c,$(FUZZ_DIR)/obj/%.o,$(LIB_C)) $(patsubst %.S,$(FUZZ_DIR)/obj/%.o,$(LIB_S))
LIB_HDRS := $(wildcard src/core/*.h src/platform/*.h include/*.h)
PROBE_OBJ := $(FUZZ_DIR)/obj/tests/fuzz/arprobe.o
# the arena probe (tests/fuzz/Makefile's): every src/core object reports each toks_ar_alloc's site,
# bound and end; `run.sh probe` on the fuzz build prints it, here it guards load work per site
$(FUZZ_DIR)/obj/src/core/%.o: LIBFLAGS += -include tests/fuzz/arprobe.h

DRV := $(FUZZ_DIR)/hostile_driver

.PHONY: all clean
all: $(DRV)

$(FUZZ_DIR)/obj/%.o: %.c $(LIB_HDRS) tests/fuzz/arprobe.h tests/fuzz/hostile/Makefile
	@mkdir -p $(dir $@)
	$(CC) $(LIBFLAGS) -c $< -o $@

$(FUZZ_DIR)/obj/%.o: %.S
	@mkdir -p $(dir $@)
	$(CC) -Iinclude -Isrc/core -Isrc/asm/$(ISA) -c $< -o $@

$(DRV): tests/fuzz/hostile/hostile_driver.c $(LIB_OBJ) $(PROBE_OBJ) tests/fuzz/hostile/Makefile $(LIB_HDRS)
	@mkdir -p $(dir $@)
	$(CC) $(COMMON) $(SANFLAGS) -o $@ $< $(LIB_OBJ) $(PROBE_OBJ)

clean:
	rm -rf $(FUZZ_DIR)
