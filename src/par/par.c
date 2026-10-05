/*
 * par.c: toks_par (SPEC §0.3, §2.4, §4.10, §22), the companion layer: a small persistent worker pool with a
 * load balancer, used only where it beats one core. Threads live here and nowhere else; the core stays
 * thread-free. No semantics of its own: every id comes from toks_encode on the caller's bytes, every cut from
 * toks_split_points, the post-processor from the core (split.h). docs/usage.md (threads) is the user's view.
 *
 * The pool: n participants share one read-only context. The calling thread is participant 0 and n - 1 threads
 * are the rest, created once: n is the caller's cap (default min(4, the fast cores), never more than the cpus
 * the process may run on). Workers prefer the fast cores: on linux a worker is confined to the cpus of the
 * highest capacity class (cpu_capacity, else the highest max frequency) when the process may run on several
 * classes (a GB10's X925s, not its A725s); on apple it takes the caller's QoS class. Each participant owns a
 * scratch it grows to the longest unit it meets and keeps, so its piece caches stay warm across calls.
 *
 * A call is a batch of items (toks_par_encode is a batch of one):
 *   1. k, the participants: 1 when the call has fewer than PAR_MIN_BYTES (no clock read, no thread touched);
 *      else the cost model: est = the call's serial time (its bytes x the pool's measured cost a byte, the
 *      lower of its units' and its serial calls': c_dec), the participants' time k T(k) = est (1 + eps) + the
 *      workers' join delays + half a last unit for all but one (the idle tail) + PAR_UNIT_NS a unit, and k =
 *      the most participants that keep est / (k T(k)) >= 75%. A join delay is measured per state (5.): a
 *      worker spinning after its last job joins in about a microsecond, a sleeping one costs a kernel wake
 *      (2-250 us measured: deep idle states are slow to leave), unless calls come back to back (the last one
 *      ended within a spin): one wake then buys workers that spin for the calls after it. Compute is spent
 *      only where it pays.
 *   2. units, planned by the caller: whole items grouped to about T bytes (T = bytes / (k x PAR_UNITS_PER),
 *      within [PAR_MIN_UNIT, PAR_MAX_UNIT]), an item of >= 2 T split at toks_split_points' cuts into parts of
 *      about T (an item without cuts stays whole); over the last quarter of the bytes, units of T / 4, so the
 *      participant that runs out of work last waits for a small unit (longest first, as LPT scheduling).
 *   3. the caller wakes exactly workers 1 .. k - 1 (each sleeps on its own word: no wake-all; the rest stay
 *      asleep) and works too; every participant claims units in order from one atomic cursor (dynamic
 *      balancing: equal bytes are not equal time across scripts, cache misses and big / little cores).
 *   4. a whole item is one toks_encode straight into its own out: zero copies. A split item's first part goes
 *      straight to out + its pre-ids; part i >= 1 into a slab of the pool's staging, room b_i + 4 ids (bpe
 *      gives at most one id per byte, so K5 writes there directly and never through its bounce, and slabs are
 *      disjoint, so K5's 4-id overstore cannot touch a neighbour). The participant that encoded a part copies
 *      it once to out + P_i (P_i = the pre-ids + the item's earlier parts' counts) as soon as every earlier
 *      part of the item is done (its ids still in that core's cache), else in order after its last claim.
 *      Copies run in parallel: sources (staging) and destinations (out) never overlap, and every copy waits
 *      for part 0, whose toks_encode may overwrite out past its own ids. A part whose count passes its slab
 *      (impossible for bpe; future models) or that fails sends its item back to a serial encode.
 *   5. the caller folds what it measured into the model: the units' encode time per byte; then the
 *      participants' time the model did not foresee (k x the wall time - the rest of the model): a call of
 *      >= 2 ms of encodes files it under eps, a shorter one per worker under the state worker 1 was in (the
 *      join delay plus what a woken core costs beyond it: caches and clocks that start cold).
 *      toks_par_create seeds the per-worker costs with this host's own join delays (empty jobs, the workers
 *      spinning, then asleep): the threshold is measured per host from the first call on.
 * An idle worker spins about as long as a wake costs (the measured sleeping join delay, within [PAR_SPIN_LO,
 * PAR_SPIN_HI]), then sleeps in the kernel on its word (futex / ulock / WaitOnAddress): ski rental between
 * burning a core and paying a wake. A worker checks into a job (active += 1, then open must still be 1 and the
 * job must be the one it was asked into), so a call returns once its work is done and no straggler is inside,
 * without waiting for workers that wake late. env TOKS_PAR_EAGER=1 at toks_par_create skips the model (k = n
 * whenever the units allow): a measurement knob for tests and scaling benches.
 */
#if defined(__linux__)
#  define _GNU_SOURCE 1                    /* sched_getaffinity, pthread_attr_setaffinity_np, syscall */
#elif defined(__APPLE__)
#  define _DARWIN_C_SOURCE 1
#endif
#include "core.h"
#include "split.h"

#include <stdatomic.h>

#if defined(_WIN32)
#  include <windows.h>
#  pragma comment(lib, "synchronization.lib")
#else
#  include <pthread.h>
#  include <sched.h>
#  include <time.h>
#  include <unistd.h>
#  if defined(__linux__)
#    include <fcntl.h>
#    include <linux/futex.h>
#    include <sys/syscall.h>
#  elif defined(__APPLE__)
#    include <pthread/qos.h>
#    include <sys/sysctl.h>
/* the os's address wait (libc++'s std::atomic::wait uses it); in libSystem since macos 10.12 */
extern int __ulock_wait(uint32_t op, void *addr, uint64_t value, uint32_t timeout_us);
extern int __ulock_wake(uint32_t op, void *addr, uint64_t wake_value);
#    define TOKS_UL_COMPARE_AND_WAIT 1u
#    define TOKS_ULF_NO_ERRNO        0x01000000u
#  endif
#endif

#define PAR_DEFAULT_N  4u                  /* n_threads 0: a couple of participants (SPEC §0.3) */
#define PAR_MAX_N      1024u
#define PAR_MIN_BYTES  (16u << 10)         /* a call of fewer bytes stays on the caller: no clock, no wake */
#define PAR_MIN_UNIT   (8u << 10)          /* a unit (one claim) holds at least this many bytes */
#define PAR_MAX_UNIT   (1u << 20)          /* and at most this many where cuts allow: bounds a worker's scratch */
#define PAR_UNITS_PER  8u                  /* units per participant: dynamic claiming evens out the rest */
#define PAR_EFF        768u                /* the efficiency floor, of 1024: a participant joins only while the */
                                           /* model keeps every participant >= 75% busy */
#define PAR_UNIT_NS    200u                /* the model's fixed cost of a unit (a claim, two clock reads, a call) */
#define PAR_SCR_MIN    (1u << 20)          /* a participant's first scratch holds texts of this many bytes */
#define PAR_C0         2500u               /* ps per byte before the first measure (400 MB/s) */
#define PAR_EPS0       102u                /* the proportional cost of going wide, of 1024, before the first measure */
#define PAR_WAKE0      30000u              /* ns: a sleeping worker's join delay where create could not measure it */
#define PAR_JOIN0      10000u              /* ns: a spinning worker's cost, and the least create seeds it with: */
                                           /* its join delay is ~1 us, but a short call pays 3-30 us a worker */
                                           /* (measured: cold caches, the split); a pool starts careful and */
                                           /* earns a lower cost from calls that went wide */
#define PAR_SPIN_LO    10000u              /* an idle worker spins as long as a wake costs, within these (ns) */
#define PAR_SPIN_HI    200000u
#define PAR_LINE       128u                /* apple's line; two of x86's (adjacent-line prefetch) */
#define PAR_OVERFLOW   (-1000)             /* internal: a part's ids did not fit its slab */
#define PAR_PART       UINT64_MAX          /* unit.b of a unit that is one part */

typedef struct { _Alignas(PAR_LINE) _Atomic uint32_t v; } line32;
typedef struct { _Alignas(PAR_LINE) _Atomic uint64_t v; } line64;

#if defined(_WIN32)
typedef HANDLE thr_t;
#else
typedef pthread_t thr_t;
#endif

typedef struct slot {                      /* one participant; slot 0 is the calling thread */
    _Alignas(PAR_LINE) _Atomic uint32_t word;   /* the job a worker is asked into: its own wait address */
    _Atomic uint32_t asleep;               /* 1 while it waits in the kernel */
    _Alignas(PAR_LINE) uint8_t *scr;       /* toks_plat_arena'd, bound to the context */
    uint64_t scr_bytes;
    uint64_t scr_len;                      /* the longest text it holds */
    uint64_t t_join;                       /* when it joined its last job (ns) */
    uint64_t busy_ns, busy_bytes;          /* its timed encodes in that job */
    uint32_t grew;                         /* its scratch grew in that job (the job is no sample of the model) */
    _Atomic uint32_t joined;               /* that job's number (written after t_join) */
    uint32_t was_asleep;                   /* the caller's note: asleep when the job asked it in */
    struct toks_par *par;
    uint32_t id;
} slot;

typedef struct unit { uint64_t a, b; } unit;    /* whole items [a, b), or part a when b == PAR_PART */

typedef struct part {                      /* a split item's part */
    uint64_t item;
    uint64_t a, len;                       /* the item's bytes [a, a + len) */
    uint64_t slab;                         /* after the item's first part: ids at stage + slab, room len + 4 */
    uint64_t nxt;                          /* its encoder's list of parts waiting to be placed */
    int64_t  cnt;                          /* its toks_encode result */
    uint32_t first;                        /* 1: the item's first part */
    _Atomic uint32_t done;                 /* 1 once cnt and the ids are final */
} part;

struct toks_par {
    line32  busy;                          /* one call at a time */
    line32  open;                          /* 1 while the job takes participants */
    line32  active;                        /* workers inside a job */
    line32  seq;                           /* the job's number: a worker's word takes it when it is asked in */
    line32  quit;
    line64  next;                          /* the claim cursor over units */
    line64  fin;                           /* units finished (a part counts once placed) */
    _Alignas(PAR_LINE) const toks_ctx *ctx;
    uint32_t n;                            /* participants at most */
    uint32_t n_fast;                       /* the fast cpus the process may run on */
    uint32_t eager;                        /* TOKS_PAR_EAGER: k = n whenever the units allow */
    uint32_t last_k;                       /* participants of the last call */
    uint32_t flags;                        /* the job's encode flags */
    uint32_t scr_flags;                    /* every participant's toks_scratch_init flags */
    _Atomic uint64_t spin_ns;              /* an idle worker's spin before it sleeps: o_wake, bounded */
    uint64_t c_ps;                         /* the model: encode ps per byte of the units (any participants) */
    uint64_t c_ser;                        /* and of the caller alone (timed serial calls) */
    uint64_t o_wake, o_join;               /* the model: what a worker costs a call, woken / spinning (ns) */
    uint64_t lat_wake;                     /* a sleeping worker's join delay alone (ns): the spin's length */
    uint64_t wake0, join0;                 /* create's measures: what a cost forgets back to on serial calls */
    uint64_t call_bytes;                   /* the job's bytes */
    uint64_t eps;                          /* the model: going wide's proportional cost (the copies, the caches a */
                                           /* split shares less), of 1024 of the encode time */
    uint64_t tail_len;                     /* the job's smallest units (bytes) */
    uint64_t t_go;                         /* the job's dispatch time */
    uint64_t t_last;                       /* when the last call that read the clock ended */
    slot    *slots;                        /* [n] */
    thr_t   *thr;                          /* [n - 1] */
    uint8_t *mem;                          /* one block: the pool, slots, threads */
    uint64_t mem_bytes;
    toks_par_item *items;                  /* the job */
    uint64_t n_items;
    const uint32_t *pre, *suf;             /* the post-processor's ids under flags */
    uint32_t n_pre, n_suf;
    unit    *units;
    uint64_t n_units, units_cap;
    part    *parts;
    uint64_t n_parts, parts_cap;
    uint64_t *cuts;
    uint64_t cuts_cap;
    uint32_t *stage;                       /* grow-only staging ids */
    uint64_t stage_ids;
};

/* ---- platform: relax, clock, sleep / wake on an address, threads, cpus ----------------------------------- */

static void relax(void)
{
#if defined(__x86_64__) || defined(_M_X64)
    __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(_M_ARM64)
    __asm__ __volatile__("yield" ::: "memory");
#endif
}

static uint64_t now_ns(void)
{
#if defined(_WIN32)
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    uint64_t q = (uint64_t)c.QuadPart, hz = (uint64_t)f.QuadPart;
    return (q / hz) * 1000000000u + (q % hz) * 1000000000u / hz;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
#endif
}

static void addr_wait(_Atomic uint32_t *a, uint32_t v)   /* returns when *a != v, or spuriously */
{
#if defined(__linux__)
    syscall(SYS_futex, (uint32_t *)a, FUTEX_WAIT_PRIVATE, v, NULL, NULL, 0);
#elif defined(__APPLE__)
    __ulock_wait(TOKS_UL_COMPARE_AND_WAIT | TOKS_ULF_NO_ERRNO, (void *)a, v, 0u);
#elif defined(_WIN32)
    WaitOnAddress((volatile VOID *)a, &v, sizeof v, INFINITE);
#else
    (void)a; (void)v;
    sched_yield();
#endif
}

static void addr_wake(_Atomic uint32_t *a)                /* the one thread waiting on a */
{
#if defined(__linux__)
    syscall(SYS_futex, (uint32_t *)a, FUTEX_WAKE_PRIVATE, 1, NULL, NULL, 0);
#elif defined(__APPLE__)
    __ulock_wake(TOKS_UL_COMPARE_AND_WAIT | TOKS_ULF_NO_ERRNO, (void *)a, 0u);
#elif defined(_WIN32)
    WakeByAddressSingle((PVOID)a);
#else
    (void)a;
#endif
}

/* the cpus the process may run on and the fast ones among them */
typedef struct topo {
    uint32_t n_cpu, n_fast;
#if defined(__linux__)
    cpu_set_t fast;                        /* the fastest class, when the process may run on several */
#endif
} topo;

#if defined(__linux__)
static uint64_t sys_num(uint32_t cpu, const char *leaf)   /* /sys/devices/system/cpu/cpu<cpu>/<leaf>, 0 if absent */
{
    static const char pfx[] = "/sys/devices/system/cpu/cpu";
    char path[96], b[32];
    uint32_t n = 0u, d = 1u;
    for (const char *q = pfx; *q != 0; q++) { path[n++] = *q; }                 /* bound: the prefix */
    while (d * 10u <= cpu) { d *= 10u; }                                        /* bound: the digits */
    for (; d != 0u; d /= 10u) { path[n++] = (char)('0' + (cpu / d) % 10u); }
    path[n++] = '/';
    for (const char *q = leaf; *q != 0 && n + 1u < sizeof path; q++) { path[n++] = *q; }
    path[n] = 0;
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) { return 0u; }
    ssize_t r = read(fd, b, sizeof b - 1u);
    close(fd);
    uint64_t v = 0u;
    for (ssize_t i = 0; i < r && b[i] >= '0' && b[i] <= '9'; i++) { v = v * 10u + (uint64_t)(b[i] - '0'); }
    return v;
}
#endif

static void topo_read(topo *t)
{
    long n = 1;
#if defined(_WIN32)
    n = (long)GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    t->n_fast = (uint32_t)(n < 1 ? 1 : n);
#elif defined(__linux__)
    cpu_set_t all;
    CPU_ZERO(&all);
    CPU_ZERO(&t->fast);
    if (sched_getaffinity(0, sizeof all, &all) != 0) {
        n = sysconf(_SC_NPROCESSORS_ONLN);
        t->n_fast = (uint32_t)(n < 1 ? 1 : n);
    } else {
        static const char *const LEAF[2] = { "cpu_capacity", "cpufreq/cpuinfo_max_freq" };
        n = CPU_COUNT(&all);
        t->n_fast = (uint32_t)n;
        for (uint32_t l = 0u; l < 2u; l++) {   /* bound: 2 sources; the first that tells classes apart decides */
            uint64_t hi = 0u, lo = UINT64_MAX;
            for (uint32_t c = 0u; c < CPU_SETSIZE; c++) {   /* bound: CPU_SETSIZE */
                if (!CPU_ISSET(c, &all)) { continue; }
                uint64_t v = sys_num(c, LEAF[l]);
                if (v > hi) { hi = v; }
                if (v < lo) { lo = v; }
            }
            if (hi == 0u || lo * 10u >= hi * 9u) { continue; }   /* absent, or one class */
            uint32_t f = 0u;
            for (uint32_t c = 0u; c < CPU_SETSIZE; c++) {   /* bound: CPU_SETSIZE */
                if (CPU_ISSET(c, &all) && sys_num(c, LEAF[l]) * 10u >= hi * 9u) { CPU_SET(c, &t->fast); f++; }
            }
            t->n_fast = f;
            break;
        }
    }
#else
    n = sysconf(_SC_NPROCESSORS_ONLN);
    t->n_fast = (uint32_t)(n < 1 ? 1 : n);
#  if defined(__APPLE__)
    int32_t pc = 0;
    size_t z = sizeof pc;
    if (sysctlbyname("hw.perflevel0.logicalcpu", &pc, &z, NULL, 0) == 0 && pc > 0 && pc < n) {
        t->n_fast = (uint32_t)pc;          /* the performance cores */
    }
#  endif
#endif
    if (n < 1) { n = 1; }
    if (n > (long)PAR_MAX_N) { n = (long)PAR_MAX_N; }
    t->n_cpu = (uint32_t)n;
    if (t->n_fast < 1u || t->n_fast > t->n_cpu) { t->n_fast = t->n_cpu; }
}

static void worker_loop(slot *s);

#if defined(_WIN32)
static DWORD WINAPI thr_main(LPVOID arg) { worker_loop((slot *)arg); return 0; }
static int thr_start(thr_t *t, slot *s, const topo *tp, int fast)
{
    (void)tp; (void)fast;
    *t = CreateThread(NULL, 0, thr_main, s, 0, NULL);
    return *t != NULL ? 0 : -1;
}
static void thr_join(thr_t t) { WaitForSingleObject(t, INFINITE); CloseHandle(t); }
#else
static void *thr_main(void *arg) { worker_loop((slot *)arg); return NULL; }
static int thr_start(thr_t *t, slot *s, const topo *tp, int fast)
{
    pthread_attr_t at;
    if (pthread_attr_init(&at) != 0) { return -1; }
#if defined(__linux__)
    if (fast && tp->n_fast < tp->n_cpu) { pthread_attr_setaffinity_np(&at, sizeof tp->fast, &tp->fast); }
#elif defined(__APPLE__)
    (void)tp; (void)fast;
    pthread_attr_set_qos_class_np(&at, qos_class_self(), 0);   /* the caller's class: p-cores for p-work */
#else
    (void)tp; (void)fast;
#endif
    int r = pthread_create(t, &at, thr_main, s);
    pthread_attr_destroy(&at);
    return r;
}
static void thr_join(thr_t t) { pthread_join(t, NULL); }
#endif

/* ---- scratch, grow-only arrays ---------------------------------------------------------------------------- */

/* 0 when s's scratch holds texts of len bytes (grown geometrically when it did not), else TOKS_E_NOMEM; *grew
 * is set when it grew (that unit's time is then not a sample of the encode cost) */
static int64_t scratch_fit(const struct toks_par *p, slot *s, uint64_t len, int *grew)
{
    if (s->scr != NULL && len <= s->scr_len) { return 0; }
    uint64_t want = s->scr_len * 2u;
    if (want < len) { want = len; }
    if (want < PAR_SCR_MIN) { want = PAR_SCR_MIN; }
    if (want > TOKS_MAX_TEXT) { want = TOKS_MAX_TEXT; }
    uint64_t bytes = toks_scratch_bytes(p->ctx, want, p->scr_flags);
    uint8_t *m = toks_plat_arena(bytes);
    if (m == NULL) { return TOKS_E_NOMEM; }
    if (toks_scratch_init(p->ctx, m, bytes, p->scr_flags) != 0) { toks_plat_arena_free(m, bytes); return TOKS_E_NOMEM; }
    toks_plat_arena_free(s->scr, s->scr_bytes);
    s->scr = m;
    s->scr_bytes = bytes;
    s->scr_len = want;
    *grew = 1;
    return 0;
}

/* *a holds at least want elements of z bytes (grown by half again, contents kept); 0 or TOKS_E_NOMEM */
static int64_t grow(void **a, uint64_t *cap, uint64_t want, uint64_t z)
{
    if (want <= *cap) { return 0; }
    uint64_t c = want + want / 2u + 64u;
    void *m = toks_plat_alloc(c * z);
    if (m == NULL) { return TOKS_E_NOMEM; }
    if (*a != NULL) {
        memcpy(m, *a, (size_t)(*cap * z));
        toks_plat_free(*a, *cap * z);
    }
    *a = m;
    *cap = c;
    return 0;
}

/* ---- the job: units, encodes, placement -------------------------------------------------------------------- */

static int item_ok(const toks_par_item *it) { return it->len <= TOKS_MAX_TEXT && (it->text != NULL || it->len == 0u); }

static uint64_t item_bytes(const toks_par_item *it) { return item_ok(it) ? it->len : 0u; }

/* whole items [a, b) on s: each one toks_encode straight into its own out */
static void run_group(struct toks_par *p, slot *s, uint64_t a, uint64_t b)
{
    int grew = 0;
    uint64_t t0 = now_ns(), bytes = 0u;
    for (uint64_t i = a; i < b; i++) {     /* bound: the unit's items */
        toks_par_item *it = &p->items[i];
        int64_t r = item_ok(it) ? scratch_fit(p, s, it->len, &grew) : 0;   /* else toks_encode reports it */
        it->n = (r < 0) ? r : toks_encode(p->ctx, it->text, it->len, p->flags, it->out, it->cap, s->scr);
        bytes += item_bytes(it);
    }
    if (!grew) {
        s->busy_ns += now_ns() - t0;
        s->busy_bytes += bytes;
    }
    s->grew |= (uint32_t)grew;
}

static void run_part(struct toks_par *p, slot *s, uint64_t i)
{
    part *q = &p->parts[i];
    toks_par_item *it = &p->items[q->item];
    uint32_t fl = p->flags | TOKS_NO_POSTPROCESS | (q->first ? 0u : TOKS_CONTINUATION);
    int grew = 0;
    int64_t r = scratch_fit(p, s, q->len, &grew);
    uint64_t t0 = now_ns();
    if (r == 0 && q->first) {              /* its final place is known: straight into out */
        uint64_t c0 = (it->cap > p->n_pre) ? it->cap - p->n_pre : 0u;
        r = toks_encode(p->ctx, it->text, q->len, fl, c0 != 0u ? it->out + p->n_pre : NULL, c0, s->scr);
    } else if (r == 0) {
        r = toks_encode(p->ctx, (const uint8_t *)it->text + q->a, q->len, fl, p->stage + q->slab, q->len + 4u, s->scr);
        if (r > (int64_t)(q->len + 4u)) { r = PAR_OVERFLOW; }
    }
    if (!grew) {
        s->busy_ns += now_ns() - t0;
        s->busy_bytes += q->len;
    }
    s->grew |= (uint32_t)grew;
    q->cnt = r;
    atomic_store_explicit(&q->done, 1u, memory_order_release);
}

/* part i's ids from its slab to its item's out + at (clipped to cap); a first part is already there */
static void place(struct toks_par *p, uint64_t i, uint64_t at, int bad)
{
    part *q = &p->parts[i];
    toks_par_item *it = &p->items[q->item];
    if (!q->first && !bad && q->cnt > 0 && at < it->cap) {
        uint64_t m = (uint64_t)q->cnt;
        if (m > it->cap - at) { m = it->cap - at; }
        toks_cpy(it->out + at, p->stage + q->slab, m * 4u);
    }
    atomic_fetch_add_explicit(&p->fin.v, 1u, memory_order_release);
}

/* a participant's view of the finished prefix of the parts: [0, k) done, sum = the ids of k's item's parts
 * before k, bad = one of them failed; head .. tail = its own parts not placed yet (a list through nxt, written
 * and read by this participant only) */
typedef struct cursor { uint64_t k, sum, head, tail; int bad; } cursor;

static void advance(struct toks_par *p, cursor *c, uint64_t upto, int wait)
{
    while (c->k < upto) {                  /* bound: n_parts; a wait lasts until another participant's part is done */
        part *q = &p->parts[c->k];
        if (atomic_load_explicit(&q->done, memory_order_acquire) == 0u) {
            if (!wait) { return; }
            relax();
            continue;
        }
        if (q->first) { c->sum = 0u; c->bad = 0; }
        if (q->cnt < 0) { c->bad = 1; }
        if (c->k == c->head) {
            place(p, c->k, p->n_pre + c->sum, c->bad);
            c->head = q->nxt;
        }
        c->sum += (q->cnt > 0) ? (uint64_t)q->cnt : 0u;
        c->k++;
    }
}

static void run_job(struct toks_par *p, slot *s)
{
    cursor c = { 0u, 0u, UINT64_MAX, UINT64_MAX, 0 };
    for (;;) {                             /* bound: n_units claims */
        uint64_t u = atomic_fetch_add_explicit(&p->next.v, 1u, memory_order_relaxed);
        if (u >= p->n_units) { break; }
        const unit *x = &p->units[u];
        if (x->b != PAR_PART) {
            run_group(p, s, x->a, x->b);
            atomic_fetch_add_explicit(&p->fin.v, 1u, memory_order_release);
            continue;
        }
        uint64_t i = x->a;
        run_part(p, s, i);
        p->parts[i].nxt = UINT64_MAX;
        if (c.head == UINT64_MAX) { c.head = i; } else { p->parts[c.tail].nxt = i; }
        c.tail = i;
        advance(p, &c, i + 1u, 0);         /* place it now if every earlier part is done */
    }
    if (c.head != UINT64_MAX) { advance(p, &c, c.tail + 1u, 1); }
}

/* ---- planning ------------------------------------------------------------------------------------------- */

static uint64_t unit_len(uint64_t bytes, uint32_t k);   /* the cost model's (below) */
static uint64_t tail_len(uint64_t t);
static void set_spin(struct toks_par *p);

static int64_t unit_add(struct toks_par *p, uint64_t a, uint64_t b)
{
    if (grow((void **)&p->units, &p->units_cap, p->n_units + 1u, sizeof(unit)) != 0) { return TOKS_E_NOMEM; }
    p->units[p->n_units].a = a;
    p->units[p->n_units].b = b;
    p->n_units++;
    return 0;
}

static int64_t part_add(struct toks_par *p, uint64_t i, uint64_t a, uint64_t b, uint64_t *ids)
{
    if (grow((void **)&p->parts, &p->parts_cap, p->n_parts + 1u, sizeof(part)) != 0 ||
        grow((void **)&p->units, &p->units_cap, p->n_units + 1u, sizeof(unit)) != 0) {
        return TOKS_E_NOMEM;
    }
    part *q = &p->parts[p->n_parts];
    q->item = i;
    q->a = a;
    q->len = b - a;
    q->first = (a == 0u);
    q->slab = *ids;
    q->cnt = 0;
    atomic_store_explicit(&q->done, 0u, memory_order_relaxed);
    if (a != 0u) { *ids += q->len + 4u; }   /* a slab holds b_j + 4 ids */
    p->units[p->n_units].a = p->n_parts;
    p->units[p->n_units].b = PAR_PART;
    p->n_units++;
    p->n_parts++;
    return 0;
}

/* item i (len bytes, the call's bytes [at, at + len)) as parts at its certified cuts: about t bytes up to the
 * call's byte `head`, about `tail` after it (the last quarter of the work in small units: whoever runs out of
 * work last waits for a small unit, not a big one); one whole unit when it has no cut */
static int64_t split(struct toks_par *p, uint64_t i, uint64_t len, uint64_t t, uint64_t tail, uint64_t at,
                     uint64_t head, uint64_t *ids)
{
    const toks_par_item *it = &p->items[i];
    uint64_t want = (len + tail - 1u) / tail;
    if (want > 0xFFFFFFFFu) { want = 0xFFFFFFFFu; }
    if (grow((void **)&p->cuts, &p->cuts_cap, want + 1u, 8u) != 0) { return TOKS_E_NOMEM; }
    int64_t nc = toks_split_points(p->ctx, it->text, len, p->flags, (uint32_t)want, p->cuts, want - 1u, NULL);
    if (nc <= 0) { return unit_add(p, i, i + 1u); }
    p->cuts[nc] = len;
    uint64_t a = 0u;
    for (uint64_t j = 0u; j <= (uint64_t)nc; j++) {   /* bound: nc + 1 cuts: a part ends at some of them */
        uint64_t b = p->cuts[j];
        if (j < (uint64_t)nc && at + b <= head && b - a < t) { continue; }   /* a head part grows toward t */
        if (part_add(p, i, a, b, ids) != 0) { return TOKS_E_NOMEM; }
        a = b;
    }
    return 0;
}

/* the job's units for k participants (bytes: the items' total): whole items grouped, big ones split, about t
 * bytes each over the first three quarters of the bytes and about t / 4 over the rest */
static int64_t plan(struct toks_par *p, uint64_t bytes, uint32_t k)
{
    uint64_t t = unit_len(bytes, k), tail = tail_len(t), head = bytes - bytes / 4u, at = 0u, g = 0u, acc = 0u, ids = 0u;
    p->tail_len = tail;
    p->n_units = 0u;
    p->n_parts = 0u;
    for (uint64_t i = 0u; i < p->n_items; i++) {   /* bound: n_items */
        uint64_t len = item_bytes(&p->items[i]);
        if (len >= 2u * t) {
            if ((g < i && unit_add(p, g, i) != 0) || split(p, i, len, t, tail, at, head, &ids) != 0) {
                return TOKS_E_NOMEM;
            }
            at += len;
            g = i + 1u;
            acc = 0u;
            continue;
        }
        acc += len;
        at += len;
        if (acc >= (at <= head ? t : tail)) {
            if (unit_add(p, g, i + 1u) != 0) { return TOKS_E_NOMEM; }
            g = i + 1u;
            acc = 0u;
        }
    }
    if (g < p->n_items && unit_add(p, g, p->n_items) != 0) { return TOKS_E_NOMEM; }
    if (ids > p->stage_ids) {              /* grow-only, released with the pool */
        uint64_t z = ids + ids / 2u;
        uint32_t *st = (uint32_t *)(void *)toks_plat_arena(z * 4u);
        if (st == NULL) { return TOKS_E_NOMEM; }
        toks_plat_arena_free((uint8_t *)p->stage, p->stage_ids * 4u);
        p->stage = st;
        p->stage_ids = z;
    }
    return 0;
}

/* after the job: each split item's count and post-processor ids, or its serial encode when a part failed */
static void finish_parts(struct toks_par *p)
{
    for (uint64_t j = 0u; j < p->n_parts;) {   /* bound: n_parts, one item's run of parts at a time */
        toks_par_item *it = &p->items[p->parts[j].item];
        uint64_t n = p->n_pre;
        int bad = 0;
        do {                               /* bound: the item's parts */
            int64_t r = p->parts[j].cnt;
            bad |= r < 0;
            n += (r > 0) ? (uint64_t)r : 0u;
            j++;
        } while (j < p->n_parts && !p->parts[j].first);
        if (!bad) {
            for (uint64_t i = 0u; i < p->n_pre && i < it->cap; i++) { toks_st32(it->out + i, p->pre[i]); }
            for (uint64_t i = 0u; i < p->n_suf && n + i < it->cap; i++) { toks_st32(it->out + n + i, p->suf[i]); }
            it->n = (int64_t)(n + p->n_suf);
        } else {
            slot *s0 = &p->slots[0];
            int grew = 0;
            int64_t r = scratch_fit(p, s0, it->len, &grew);
            it->n = (r < 0) ? r : toks_encode(p->ctx, it->text, it->len, p->flags, it->out, it->cap, s0->scr);
        }
    }
}

/* ---- the cost model --------------------------------------------------------------------------------------- */

static void ewma(uint64_t *x, uint64_t v)
{
    uint64_t hi = *x * 4u + 20000u;        /* one outlier (a descheduled worker) moves the estimate boundedly */
    if (v > hi) { v = hi; }
    *x = v >= *x ? *x + (v - *x) / 4u : *x - (*x - v) / 4u;
}

static void ewma_cost(uint64_t *x, uint64_t v)   /* going wide's cost: up fast, down slowly (stay on one core */
{                                                /* until going wide has shown it pays) */
    uint64_t hi = *x * 4u + 20000u;
    if (v > hi) { v = hi; }
    *x = v >= *x ? *x + (v - *x) / 2u : *x - (*x - v) / 8u;
}

/* the units of a call of `bytes` on k participants: about t bytes, t / 4 over the last quarter (plan) */
static uint64_t unit_len(uint64_t bytes, uint32_t k)
{
    uint64_t t = bytes / ((uint64_t)k * PAR_UNITS_PER);
    return t < PAR_MIN_UNIT ? PAR_MIN_UNIT : t > PAR_MAX_UNIT ? PAR_MAX_UNIT : t;
}
static uint64_t tail_len(uint64_t t) { return t / 4u < PAR_MIN_UNIT ? PAR_MIN_UNIT : t / 4u; }

/* participant time (ns) beyond est (ns) when k participants share a call of `bytes` at c ps a byte, the join
 * delays aside: eps of est, the idle tail (all but the last to finish wait half a last unit, on average),
 * PAR_UNIT_NS a unit */
static uint64_t extra(const struct toks_par *p, uint64_t bytes, uint32_t k, uint64_t est, uint64_t c)
{
    uint64_t t = unit_len(bytes, k), tl = tail_len(t);
    uint64_t units = (bytes - bytes / 4u) / t + bytes / 4u / tl + 1u;
    return est / 1024u * p->eps + (uint64_t)(k - 1u) * (tl / 2u * c / 1000u) + units * PAR_UNIT_NS;
}

/* the cost a byte the decision assumes: the lower of the units' and the caller's alone. Units of a split call
 * run slower than one core on the same text (each participant's caches see only its units), so the units'
 * cost alone would make the serial estimate, and the participants, grow with going wide; the lower one keeps
 * a medium call narrow. A call too big for the difference to matter goes wide either way. */
static uint64_t c_dec(const struct toks_par *p) { return p->c_ser < p->c_ps ? p->c_ser : p->c_ps; }

/* participants for a call of `bytes` at time `now`: the most that keep the model's efficiency est / (k T(k)) >=
 * PAR_EFF / 1024, k T(k) = est + extra + the join delays of workers 1 .. k - 1, each by its state: a sleeping
 * worker costs a wake, unless calls come back to back (the last ended within a spin): then the one wake buys
 * workers that stay spinning for the calls after it */
static uint32_t choose_k(const struct toks_par *p, uint64_t bytes, uint64_t now)
{
    if (p->n == 1u || bytes < PAR_MIN_BYTES) { return 1u; }
    uint64_t kmax = bytes / PAR_MIN_UNIT;
    if (kmax > p->n) { kmax = p->n; }
    if (p->eager) { return (uint32_t)kmax; }
    uint64_t c = c_dec(p), est = bytes / 1000u * c, delays = 0u;
    int hot = p->t_last != 0u && now - p->t_last < atomic_load_explicit(&p->spin_ns, memory_order_relaxed);
    uint32_t k = 1u;
    while (k < kmax) {                     /* bound: n */
        int asleep = atomic_load_explicit(&p->slots[k].asleep, memory_order_relaxed) != 0u;
        delays += asleep && !hot ? p->o_wake : p->o_join;
        uint64_t all = est + extra(p, bytes, k + 1u, est, c) + delays;
        if (est / 1024u * 1024u < all / 1024u * PAR_EFF) { break; }
        k++;
    }
    return k;
}

static void set_spin(struct toks_par *p)
{
    uint64_t v = p->lat_wake < PAR_SPIN_LO ? PAR_SPIN_LO : p->lat_wake > PAR_SPIN_HI ? PAR_SPIN_HI : p->lat_wake;
    atomic_store_explicit(&p->spin_ns, v, memory_order_relaxed);
}

/* after job q of k participants, planned from t0, its units finished at t_end: the encode cost; a sleeping
 * worker's join delay (the spin); and what the model did not foresee. A call of >= 2 ms of encodes files it
 * under eps (as a share of the encodes: the copies, caches a split shares less); a shorter call, per worker,
 * under the state worker 1 was in (o_wake / o_join: the join delay and what a woken core costs beyond it,
 * caches and clocks that start cold) */
static void learn(struct toks_par *p, uint32_t k, uint32_t q, uint64_t t0, uint64_t t_end)
{
    uint64_t ns = 0u, bytes = 0u, delays = 0u;
    uint32_t grew = 0u;
    for (uint32_t j = 0u; j < k; j++) {    /* bound: k */
        slot *s = &p->slots[j];
        if (j != 0u) {
            int in = atomic_load_explicit(&s->joined, memory_order_acquire) == q && s->t_join >= p->t_go;
            uint64_t d = (in ? s->t_join : t_end) - p->t_go;
            if (s->was_asleep) { ewma(&p->lat_wake, d); }
            delays += d;
            if (!in) { continue; }
        }
        ns += s->busy_ns;
        bytes += s->busy_bytes;
        grew |= s->grew;
    }
    if (bytes >= PAR_MIN_BYTES) { ewma(&p->c_ps, ns * 1000u / bytes); }
    set_spin(p);
    if (grew) { return; }                  /* a scratch grew: no sample of the costs */
    uint64_t est = p->call_bytes / 1000u * p->c_ps, all = (uint64_t)k * (t_end - t0);
    uint64_t idle = (uint64_t)(k - 1u) * (p->tail_len / 2u * p->c_ps / 1000u) + p->n_units * PAR_UNIT_NS;
    if (ns >= 2000000u) {                  /* long: the proportional rest */
        uint64_t fixed = ns + delays + idle;
        uint64_t e = all > fixed ? (all - fixed) * 1024u / ns : 0u;
        ewma_cost(&p->eps, e > 512u ? 512u : e);
    } else {                               /* short: the rest per worker */
        uint64_t fixed = est + est / 1024u * p->eps + idle;
        uint64_t o = all > fixed ? (all - fixed) / (k - 1u) : 0u;
        ewma_cost(p->slots[1].was_asleep ? &p->o_wake : &p->o_join, o < 100u ? 100u : o);
        if (p->o_join > p->o_wake) { p->o_join = p->o_wake; }   /* a spinning worker never costs more */
    }
}

/* ---- dispatch ------------------------------------------------------------------------------------------- */

static void worker_loop(slot *s)
{
    struct toks_par *p = s->par;
    uint32_t seen = 0u;
    for (;;) {                             /* bound: one pass per job until quit */
        uint32_t w = atomic_load_explicit(&s->word, memory_order_acquire);
        if (w == seen) {
            uint64_t t0 = now_ns();
            for (uint32_t i = 1u; (w = atomic_load_explicit(&s->word, memory_order_acquire)) == seen; i++) {
                relax();                   /* spin: a busy caller's next job starts without a wake */
                if ((i & 63u) != 0u || now_ns() - t0 < atomic_load_explicit(&p->spin_ns, memory_order_relaxed)) {
                    continue;              /* (re-read: create's calibration ends a spin early) */
                }
                atomic_store(&s->asleep, 1u);
                while ((w = atomic_load(&s->word)) == seen) { addr_wait(&s->word, seen); }   /* bound: until asked */
                atomic_store(&s->asleep, 0u);
                break;
            }
        }
        seen = w;
        if (atomic_load(&p->quit.v) != 0u) { return; }
        atomic_fetch_add(&p->active.v, 1u);
        if (atomic_load(&p->open.v) != 0u && atomic_load(&p->seq.v) == w) {
            s->t_join = now_ns();
            s->busy_ns = 0u;
            s->busy_bytes = 0u;
            s->grew = 0u;
            atomic_store_explicit(&s->joined, w, memory_order_release);
            run_job(p, s);
        }
        atomic_fetch_sub(&p->active.v, 1u);
    }
}

static void lock(struct toks_par *p)
{
    uint32_t z = 0u;
    while (!atomic_compare_exchange_weak(&p->busy.v, &z, 1u)) {   /* bound: until the other call returns */
        z = 0u;
#if defined(_WIN32)
        SwitchToThread();
#else
        sched_yield();
#endif
    }
}

static void unlock(struct toks_par *p) { atomic_store(&p->busy.v, 0u); }

/* opens job q to workers 1 .. k - 1: exactly the workers it takes, each woken on its own word if asleep */
static uint32_t open_job(struct toks_par *p, uint32_t k)
{
    uint32_t q = atomic_load(&p->seq.v) + 1u;
    if (q == 0u) { q = 1u; }               /* words start at 0: a job's number is never 0 */
    atomic_store_explicit(&p->next.v, 0u, memory_order_relaxed);
    atomic_store_explicit(&p->fin.v, 0u, memory_order_relaxed);
    atomic_store(&p->seq.v, q);
    atomic_store(&p->open.v, 1u);
    p->t_go = now_ns();
    for (uint32_t j = 1u; j < k; j++) {    /* bound: k */
        slot *s = &p->slots[j];
        atomic_store(&s->word, q);
        s->was_asleep = atomic_load(&s->asleep);
        if (s->was_asleep != 0u) { addr_wake(&s->word); }
    }
    return q;
}

static void close_job(struct toks_par *p)
{
    atomic_store(&p->open.v, 0u);
    while (atomic_load(&p->active.v) != 0u) { relax(); }    /* stragglers leave before the job is rewritten */
}

/* the planned job (planning began at t0) on participants 0 .. k - 1, this thread being 0, until every unit is
 * finished; then learn */
static void dispatch(struct toks_par *p, uint32_t k, uint64_t t0)
{
    slot *s0 = &p->slots[0];
    s0->busy_ns = 0u;
    s0->busy_bytes = 0u;
    s0->grew = 0u;
    uint32_t q = open_job(p, k);
    run_job(p, s0);
    while (atomic_load_explicit(&p->fin.v, memory_order_acquire) < p->n_units) { relax(); }   /* bound: the job */
    uint64_t t_end = now_ns();
    close_job(p);
    learn(p, k, q, t0, t_end);
}

/* create's measure of this host: an empty job, open until every worker has joined (or 20 ms): four times with
 * the workers held spinning (the first lets new threads settle on their cpus; the fastest of the other three
 * seeds o_join, never under PAR_JOIN0: a delay only ever grows by noise), then twice after they all fell asleep
 * and sat 2 ms in the kernel (the slower seeds o_wake and the spin: a sporadic call finds its workers in a deep
 * idle state). About 5 ms, once per pool. */
static void calibrate(struct toks_par *p)
{
    uint64_t wake = 0u, join = UINT64_MAX;
    for (uint32_t r = 0u; r < 6u; r++) {   /* bound: 6 pings */
        atomic_store_explicit(&p->spin_ns, r < 4u ? 20000000u : 0u, memory_order_relaxed);
        if (r >= 4u) {
            uint64_t t = now_ns();
            for (uint32_t j = 1u; j < p->n && now_ns() - t < 20000000u; ) {   /* bound: 20 ms */
                if (atomic_load(&p->slots[j].asleep) != 0u) { j++; } else { relax(); }
            }
            for (t = now_ns(); now_ns() - t < 2000000u; ) { relax(); }   /* bound: 2 ms, deep in the kernel */
        }
        p->n_units = 0u;
        uint32_t q = open_job(p, p->n);
        uint64_t t = p->t_go, sum = 0u, cnt = 0u;
        for (uint32_t j = 1u; j < p->n; j++) {   /* bound: n - 1 workers, 20 ms */
            slot *s = &p->slots[j];
            while (atomic_load_explicit(&s->joined, memory_order_acquire) != q && now_ns() - t < 20000000u) { relax(); }
            if (atomic_load_explicit(&s->joined, memory_order_acquire) == q && s->t_join >= t) {
                sum += s->t_join - t;
                cnt++;
            }
        }
        close_job(p);
        if (cnt != 0u && r >= 1u && r < 4u && sum / cnt < join) { join = sum / cnt; }
        if (cnt != 0u && r >= 4u && sum / cnt > wake) { wake = sum / cnt; }
    }
    if (wake != 0u) { p->o_wake = p->lat_wake = wake < 1000u ? 1000u : wake; }
    if (join != UINT64_MAX && join > p->o_join) { p->o_join = join; }   /* never under PAR_JOIN0 */
    if (p->o_wake < p->o_join) { p->o_wake = p->o_join; }   /* a woken worker costs at least a spinning one */
    p->wake0 = p->o_wake;
    p->join0 = p->o_join;
    set_spin(p);
}

/* every item of the job, on as many participants as its bytes justify */
static void run_items(struct toks_par *p, uint64_t bytes)
{
    int timed = p->n > 1u && bytes >= PAR_MIN_BYTES;
    uint64_t t0 = timed ? now_ns() : 0u;
    p->call_bytes = bytes;
    uint32_t k = timed ? choose_k(p, bytes, t0) : 1u;
    if (k > 1u && plan(p, bytes, k) == 0 && p->n_units > 1u) {
        if (k > p->n_units) { k = (uint32_t)p->n_units; }
        p->last_k = k;
        dispatch(p, k, t0);
        finish_parts(p);
        p->t_last = now_ns();
        return;
    }
    slot *s0 = &p->slots[0];               /* the caller alone: timed for the model when the call could split */
    p->last_k = 1u;
    if (timed) {
        s0->busy_ns = 0u;
        s0->busy_bytes = 0u;
        s0->grew = 0u;
        run_group(p, s0, 0u, p->n_items);
        if (s0->busy_bytes >= PAR_MIN_BYTES) {
            ewma(&p->c_ps, s0->busy_ns * 1000u / s0->busy_bytes);
            ewma(&p->c_ser, s0->busy_ns * 1000u / s0->busy_bytes);
        }
        p->o_wake -= p->o_wake > p->wake0 ? (p->o_wake - p->wake0) / 64u : 0u;   /* forget slowly: a cost */
        p->o_join -= p->o_join > p->join0 ? (p->o_join - p->join0) / 64u : 0u;   /* that kept calls serial */
        p->t_last = now_ns();                                                     /* gets measured again */
        return;
    }
    for (uint64_t i = 0u; i < p->n_items; i++) {   /* bound: n_items */
        toks_par_item *it = &p->items[i];
        int grew = 0;
        int64_t r = item_ok(it) ? scratch_fit(p, s0, it->len, &grew) : 0;
        it->n = (r < 0) ? r : toks_encode(p->ctx, it->text, it->len, p->flags, it->out, it->cap, s0->scr);
    }
}

/* ---- public ------------------------------------------------------------------------------------------- */

static uint64_t up(uint64_t v) { return (v + PAR_LINE - 1u) & ~(uint64_t)(PAR_LINE - 1u); }

int64_t toks_par_create(toks_par **out, const toks_ctx *ctx, uint32_t n_threads, uint32_t scratch_flags)
{
    if (out == NULL) { return TOKS_E_ARG; }
    *out = NULL;
    if (ctx == NULL || n_threads > PAR_MAX_N) { return TOKS_E_ARG; }
    if ((scratch_flags & ~(TOKS_SCRATCH_MEMO_MIB(0xFFFu) | TOKS_SCRATCH_CACHE_MIB(0xFFu))) != 0u ||
        !toks_scr_cache_ok(toks_scr_cache_mib(scratch_flags))) {
        return TOKS_E_ARG;                 /* what toks_scratch_init refuses */
    }
    topo tp;
    topo_read(&tp);
    uint32_t n = n_threads;
    if (n == 0u) { n = tp.n_fast < PAR_DEFAULT_N ? tp.n_fast : PAR_DEFAULT_N; }
    if (n > tp.n_cpu) { n = tp.n_cpu; }    /* more participants than cpus never wins */
    uint64_t o_slots = up(sizeof(struct toks_par));
    uint64_t o_thr = o_slots + up((uint64_t)n * sizeof(slot));
    uint64_t bytes = o_thr + up((uint64_t)n * sizeof(thr_t));
    uint8_t *m = toks_plat_arena(bytes);   /* zeroed pages */
    if (m == NULL) { return TOKS_E_NOMEM; }
    struct toks_par *p = (struct toks_par *)(void *)m;
    p->ctx = ctx;
    p->n = n;
    p->n_fast = tp.n_fast;
    p->scr_flags = scratch_flags;
    p->mem = m;
    p->mem_bytes = bytes;
    p->slots = (slot *)(void *)(m + o_slots);
    p->thr = (thr_t *)(void *)(m + o_thr);
    p->c_ps = p->c_ser = PAR_C0;
    p->eps = PAR_EPS0;
    p->o_wake = p->lat_wake = p->wake0 = PAR_WAKE0;
    p->o_join = p->join0 = PAR_JOIN0;
    atomic_store(&p->spin_ns, (uint64_t)PAR_WAKE0);
    char ev[8];
    p->eager = toks_plat_getenv("TOKS_PAR_EAGER", ev, sizeof ev) == 1 && ev[0] == '1';
    for (uint32_t i = 0u; i < n; i++) {    /* bound: n */
        p->slots[i].par = p;
        p->slots[i].id = i;
    }
    for (uint32_t i = 1u; i < n; i++) {    /* bound: n - 1: workers 1 .. n_fast - 1 on the fast cores */
        if (thr_start(&p->thr[i - 1u], &p->slots[i], &tp, i < tp.n_fast) != 0) {
            p->n = i;                      /* threads [1, i) run: stop and join them */
            toks_par_destroy(p);
            return TOKS_E_NOMEM;
        }
    }
    if (n > 1u) { calibrate(p); }
    *out = p;
    return 0;
}

void toks_par_destroy(toks_par *p)
{
    if (p == NULL) { return; }
    atomic_store(&p->quit.v, 1u);
    for (uint32_t i = 1u; i < p->n; i++) {     /* bound: n - 1 */
        atomic_fetch_add(&p->slots[i].word, 1u);
        addr_wake(&p->slots[i].word);
    }
    for (uint32_t i = 1u; i < p->n; i++) { thr_join(p->thr[i - 1u]); }   /* bound: n - 1 */
    for (uint32_t i = 0u; i < p->n; i++) { toks_plat_arena_free(p->slots[i].scr, p->slots[i].scr_bytes); }
    toks_plat_arena_free((uint8_t *)p->stage, p->stage_ids * 4u);
    toks_plat_free(p->units, p->units_cap * sizeof(unit));
    toks_plat_free(p->parts, p->parts_cap * sizeof(part));
    toks_plat_free(p->cuts, p->cuts_cap * 8u);
    toks_plat_arena_free(p->mem, p->mem_bytes);
}

static int flags_ok(uint32_t flags)
{
    return (flags & ~(TOKS_ADDED_MASK | TOKS_NO_POSTPROCESS | TOKS_CONTINUATION)) == 0u &&
           (flags & TOKS_ADDED_MASK) != TOKS_ADDED_MASK;
}

static void job_begin(struct toks_par *p, toks_par_item *items, uint64_t n_items, uint32_t flags)
{
    p->flags = flags;
    p->items = items;
    p->n_items = n_items;
    toks_pp_ids(p->ctx, flags, &p->pre, &p->n_pre, &p->suf, &p->n_suf);
    p->n_units = 0u;
    p->n_parts = 0u;
}

int64_t toks_par_encode_batch(toks_par *p, toks_par_item *items, uint64_t n_items, uint32_t flags)
{
    if (p == NULL || (items == NULL && n_items != 0u) || !flags_ok(flags)) { return TOKS_E_ARG; }
    lock(p);
    uint64_t bytes = 0u;
    for (uint64_t i = 0u; i < n_items; i++) { bytes += item_bytes(&items[i]); }   /* bound: n_items */
    job_begin(p, items, n_items, flags);
    run_items(p, bytes);
    unlock(p);
    return 0;
}

int64_t toks_par_encode(toks_par *p, const void *text, uint64_t len, uint32_t flags, uint32_t *out, uint64_t cap)
{
    if (p == NULL) { return TOKS_E_ARG; }
    if (!flags_ok(flags) || (out == NULL && cap != 0u)) {
        return toks_encode(p->ctx, text, len, flags, out, cap, NULL);   /* its argument error: no scratch read */
    }
    toks_par_item one = { text, len, out, cap, 0 };
    lock(p);
    job_begin(p, &one, 1u, flags);
    run_items(p, item_bytes(&one));
    unlock(p);
    return one.n;
}

int64_t toks_par_get_info(const toks_par *par, toks_par_info *out)
{
    if (par == NULL || out == NULL || out->size < sizeof(toks_par_info)) { return TOKS_E_ARG; }
    struct toks_par *p = (struct toks_par *)(uintptr_t)par;
    toks_par_info in;
    memset(&in, 0, sizeof in);
    in.size = sizeof in;
    lock(p);
    in.threads = p->n;
    in.fast = p->n_fast;
    in.last = p->last_k;
    in.ns_per_mib = c_dec(p) * 1048576u / 1000u;
    in.wake_ns = p->o_wake;
    in.join_ns = p->o_join;
    if (p->n > 1u) {                       /* the smallest call that takes worker 1 now */
        uint64_t lo = PAR_MIN_BYTES, hi = TOKS_MAX_TEXT, now = now_ns();
        if (choose_k(p, lo, now) > 1u) { hi = lo; }
        else if (choose_k(p, hi, now) == 1u) { lo = hi = 0u; }   /* never: going wide does not pay here */
        while (lo < hi) {                  /* bound: 30 halvings */
            uint64_t mid = lo + (hi - lo) / 2u;
            if (choose_k(p, mid, now) > 1u) { hi = mid; } else { lo = mid + 1u; }
        }
        in.min_bytes = hi;
    }
    unlock(p);
    memcpy(out, &in, sizeof in);
    return 0;
}
