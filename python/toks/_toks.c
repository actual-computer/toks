/*
 * _toks.c: toks for CPython, a thin binding over include/toks.h (language bindings are satellites on
 * the c abi; this is the first one). The library is linked in statically (python/setup.py
 * builds it with the repository's Makefile), so the extension module is the whole package's native code
 * and the library's load-time tier dispatch picks the asm tier of the machine it runs on.
 *
 * Speed (maintainer doctrine, speed): the text is read in place: a str's utf-8 buffer (a compact ascii str is
 * its own utf-8; CPython caches the utf-8 of any other str on first use), a bytes object, or any
 * contiguous buffer. Ids go from the core into a staging array that a scratch slot keeps across calls,
 * or straight into a caller's buffer (encode_into, no copy at all). A list of ids takes one reference
 * per id from the tokenizer's table of int objects, made once per id on first sight, so building the
 * list allocates nothing per id after warm-up. Calls on texts of GIL_TEXT bytes or more, and batches,
 * run without the GIL.
 *
 * Threads: a context is read-only after load and shared by every thread (toks.h). Every encode-type
 * call takes a scratch slot (scratch + staging) from the tokenizer's pool and gives it back; the pool is
 * only touched with the GIL held, so two calls never share a scratch, with or without the GIL. The int
 * table and the vocabulary are also only touched with the GIL held. The module is single-phase and
 * declares nothing about free threading, so a free-threaded interpreter keeps the GIL on while it is
 * imported.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdarg.h>
#include <stdint.h>
#include <string.h>

#include "toks.h"

#if PY_VERSION_HEX < 0x030A0000
#  error "toks needs CPython 3.10 or later"
#endif

#define DEFAULT_TEXT  (64u << 10)   /* a slot's first scratch holds texts of up to 64 KiB */
#define KEEP_TEXT     (1u << 20)    /* a slot keeps a scratch for texts of up to 1 MiB between calls */
#define KEEP_IDS      (1u << 20)    /* ... and staging for up to 1 Mi ids (4 MiB) */
#define GIL_TEXT      2048u         /* encode / pieces run without the GIL from this many bytes on */
#define GIL_IDS       8192u         /* decode runs without the GIL from this many ids on */
#define STACK_IDS     512           /* decode / push read up to this many ids into a stack array */

#define KIND_PATH     0             /* Tokenizer.source: what was loaded */
#define KIND_BYTES    1
#define KIND_STR      2

static PyObject *Error;             /* toks.Error */
static PyTypeObject TokenizerType;
static PyTypeObject StreamType;
static PyTypeObject EncodingType;

/* ======================================================================================================
 * errors: toks.Error(message) with .code (the TOKS_E_* value) and .name ("TOKS_E_...")
 * ====================================================================================================== */

static const char *err_name(int64_t r)
{
    switch (r) {
    case TOKS_E_OPEN:            return "TOKS_E_OPEN";
    case TOKS_E_FORMAT:          return "TOKS_E_FORMAT";
    case TOKS_E_UNSUPPORTED:     return "TOKS_E_UNSUPPORTED";
    case TOKS_E_TIER:            return "TOKS_E_TIER";
    case TOKS_E_SCRATCH:         return "TOKS_E_SCRATCH";
    case TOKS_E_ID:              return "TOKS_E_ID";
    case TOKS_E_CAP:             return "TOKS_E_CAP";
    case TOKS_E_LIMIT:           return "TOKS_E_LIMIT";
    case TOKS_E_ARG:             return "TOKS_E_ARG";
    case TOKS_E_NOMEM:           return "TOKS_E_NOMEM";
    default:                     return "TOKS_E_UNKNOWN";
    }
}

/* a toks.Error instance for code r with the message "<name>: <what>" */
static PyObject *error_new(int64_t r, PyObject *what)
{
    PyObject *msg = PyUnicode_FromFormat("%s: %U", err_name(r), what);
    if (msg == NULL) { return NULL; }
    PyObject *exc = PyObject_CallOneArg(Error, msg);
    Py_DECREF(msg);
    if (exc == NULL) { return NULL; }
    PyObject *code = PyLong_FromLongLong(r);
    PyObject *name = PyUnicode_FromString(err_name(r));
    if (code == NULL || name == NULL || PyObject_SetAttrString(exc, "code", code) < 0 ||
        PyObject_SetAttrString(exc, "name", name) < 0) {
        Py_XDECREF(code);
        Py_XDECREF(name);
        Py_DECREF(exc);
        return NULL;
    }
    Py_DECREF(code);
    Py_DECREF(name);
    return exc;
}

/* raises toks.Error (MemoryError for TOKS_E_NOMEM); returns NULL. */
static PyObject *fail(int64_t r, const char *fmt, ...)
{
    if (r == TOKS_E_NOMEM) { return PyErr_NoMemory(); }
    va_list ap;
    va_start(ap, fmt);
    PyObject *what = PyUnicode_FromFormatV(fmt, ap);
    va_end(ap);
    if (what == NULL) { return NULL; }
    PyObject *exc = error_new(r, what);
    Py_DECREF(what);
    if (exc != NULL) {
        PyErr_SetObject((PyObject *)Py_TYPE(exc), exc);
        Py_DECREF(exc);
    }
    return NULL;
}

/* toks._toks._error(code, message) -> a toks.Error instance (for toks._vocab). */
static PyObject *mod_error(PyObject *mod, PyObject *const *args, Py_ssize_t nargs)
{
    (void)mod;
    if (nargs != 2 || !PyLong_Check(args[0]) || !PyUnicode_Check(args[1])) {
        PyErr_SetString(PyExc_TypeError, "_error(code: int, message: str)");
        return NULL;
    }
    long long r = PyLong_AsLongLong(args[0]);
    if (r == -1 && PyErr_Occurred()) { return NULL; }
    return error_new((int64_t)r, args[1]);
}

/* ======================================================================================================
 * argument parsing for METH_FASTCALL | METH_KEYWORDS: names are interned at module init, so the
 * interpreter's kwnames match them by pointer; a non-interned spelling falls back to a compare.
 * ====================================================================================================== */

typedef struct argspec {
    const char *fname;
    int         n;          /* names */
    int         npos;       /* the first npos may be positional */
    int         nreq;       /* the first nreq are required */
    PyObject   *names[6];
} argspec;

static int parse(argspec *sp, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames, PyObject **out)
{
    if (nargs > sp->npos) {
        PyErr_Format(PyExc_TypeError, "%s() takes at most %d positional argument%s (%zd given)", sp->fname,
                     sp->npos, sp->npos == 1 ? "" : "s", nargs);
        return -1;
    }
    for (int i = 0; i < sp->n; i++) { out[i] = i < nargs ? args[i] : NULL; }
    if (kwnames != NULL) {
        Py_ssize_t nk = PyTuple_GET_SIZE(kwnames);
        for (Py_ssize_t k = 0; k < nk; k++) {
            PyObject *key = PyTuple_GET_ITEM(kwnames, k);
            int j = 0;
            while (j < sp->n && key != sp->names[j]) { j++; }
            if (j == sp->n) {
                j = 0;
                while (j < sp->n && PyUnicode_Compare(key, sp->names[j]) != 0) { j++; }
            }
            if (j == sp->n) {
                PyErr_Format(PyExc_TypeError, "%s() got an unexpected keyword argument '%U'", sp->fname, key);
                return -1;
            }
            if (out[j] != NULL) {
                PyErr_Format(PyExc_TypeError, "%s() got multiple values for argument '%U'", sp->fname, key);
                return -1;
            }
            out[j] = args[nargs + k];
        }
    }
    for (int i = 0; i < sp->nreq; i++) {
        if (out[i] == NULL) {
            PyErr_Format(PyExc_TypeError, "%s() missing required argument '%U'", sp->fname, sp->names[i]);
            return -1;
        }
    }
    return 0;
}

static PyObject *s_all, *s_nonspecial, *s_none;     /* added_tokens values */

static argspec A_ENCODE      = { "encode", 6, 1, 1, { NULL } };
static argspec A_ENCODE_EX   = { "encode_ex", 4, 1, 1, { NULL } };
static argspec A_ENCODE_INTO = { "encode_into", 5, 2, 2, { NULL } };
static argspec A_BATCH       = { "encode_batch", 4, 1, 1, { NULL } };
static argspec A_BATCH_EX    = { "encode_batch_ex", 4, 1, 1, { NULL } };
static argspec A_PIECES      = { "pieces", 3, 1, 1, { NULL } };
static argspec A_DECODE      = { "decode", 2, 2, 1, { NULL } };
static argspec A_DECODE_B    = { "decode_batch", 2, 2, 1, { NULL } };
static argspec A_BYTES_B     = { "decode_bytes_batch", 2, 1, 1, { NULL } };
static argspec A_STREAM      = { "decode_stream", 1, 1, 0, { NULL } };
static argspec A_FROM_FILE   = { "from_file", 3, 1, 1, { NULL } };
static argspec A_FROM_STR    = { "from_str", 2, 1, 1, { NULL } };
static argspec A_FROM_BUF    = { "from_buffer", 2, 1, 1, { NULL } };
static argspec A_VOCAB       = { "get_vocab", 1, 1, 0, { NULL } };
static argspec A_VOCAB_SIZE  = { "get_vocab_size", 1, 1, 0, { NULL } };
static argspec A_NSPECIAL    = { "num_special_tokens_to_add", 1, 1, 0, { NULL } };

static int intern_names(argspec *sp, ...)
{
    va_list ap;
    va_start(ap, sp);
    for (int i = 0; i < sp->n; i++) {
        const char *s = va_arg(ap, const char *);
        sp->names[i] = PyUnicode_InternFromString(s);
        if (sp->names[i] == NULL) { va_end(ap); return -1; }
    }
    va_end(ap);
    return 0;
}

/* a truth value with a default for an absent argument */
static int truth(PyObject *o, int dflt)
{
    return o == NULL ? dflt : PyObject_IsTrue(o);
}

/* ======================================================================================================
 * the tokenizer object and its scratch slots
 * ====================================================================================================== */

typedef struct slot {
    struct slot *next;
    uint8_t     *scr;          /* the scratch, bound to the context (scr_bytes) */
    uint64_t     scr_bytes;
    uint64_t     max_len;      /* the longest text the scratch is initialized for */
    uint32_t    *ids;          /* staging for ids / piece ends */
    uint64_t     ids_cap;
    uint8_t     *buf;          /* staging for decoded bytes */
    uint64_t     buf_cap;
} slot;

typedef struct {
    PyObject_HEAD
    toks_ctx   *ctx;
    toks_info   info;            /* toks_get_info at load */
    PyObject  **ints;            /* info.n_ids int objects, each made on first use (NULL until the first list) */
    slot       *pool;            /* idle slots (GIL held) */
    PyObject   *source;          /* the path (str or bytes), the bytes or the str that was loaded */
    PyObject   *vocab;           /* toks._vocab.load(...) on first use */
    PyObject   *tt;              /* the tiktoken view on first use: ({id: special}, {special: id}, ordinary flags) */
    uint8_t    *tt_bits;         /* with tt: a bit per id, set for tt's specials */
    uint32_t   *tmpl;            /* toks_template at load: tmpl_n ids, then their tmpl_n type ids (NULL: none) */
    uint32_t    tmpl_n;          /* hf's num_special_tokens_to_add(False) */
    uint32_t    tmpl_pre;        /* the first tmpl_pre go before the text's ids, the rest after */
    uint32_t    tier_asked;      /* the TOKS_TIER_* asked for at load */
    uint32_t    scr_flags;       /* every slot's toks_scratch_init flags: from_file's cache_mib */
    uint8_t     kind;            /* KIND_* of source */
    uint8_t     encode_special;  /* hf's encode_special_tokens: the default mode is NONSPECIAL */
} Tok;

static void slot_free(slot *s)
{
    if (s == NULL) { return; }
    PyMem_RawFree(s->scr);
    PyMem_RawFree(s->ids);
    PyMem_RawFree(s->buf);
    PyMem_RawFree(s);
}

static void slot_give(Tok *t, slot *s)
{
    if (s->max_len > KEEP_TEXT) {               /* a huge text: give the memory back now */
        PyMem_RawFree(s->scr);
        s->scr = NULL;
        s->scr_bytes = 0;
        s->max_len = 0;
    }
    if (s->ids_cap > KEEP_IDS) {
        PyMem_RawFree(s->ids);
        s->ids = NULL;
        s->ids_cap = 0;
    }
    if (s->buf_cap > 4u * KEEP_IDS) {
        PyMem_RawFree(s->buf);
        s->buf = NULL;
        s->buf_cap = 0;
    }
    s->next = t->pool;
    t->pool = s;
}

/* a slot; with scr, its scratch holds texts of len bytes (len <= TOKS_MAX_TEXT). NULL with an exception set. */
static slot *slot_take(Tok *t, uint64_t len, int scr)
{
    slot *s = t->pool;
    if (s != NULL) {
        t->pool = s->next;
        s->next = NULL;
    } else {
        s = (slot *)PyMem_RawCalloc(1, sizeof *s);
        if (s == NULL) { PyErr_NoMemory(); return NULL; }
    }
    if (scr && (s->scr == NULL || s->max_len < len)) {
        uint64_t want = len < DEFAULT_TEXT ? DEFAULT_TEXT : len;
        if (s->scr != NULL && want < 2u * s->max_len) { want = 2u * s->max_len; }
        if (want > TOKS_MAX_TEXT) { want = TOKS_MAX_TEXT; }
        uint64_t bytes = toks_scratch_bytes(t->ctx, want, t->scr_flags);
        uint8_t *p = bytes <= (uint64_t)PY_SSIZE_T_MAX ? (uint8_t *)PyMem_RawMalloc((size_t)bytes) : NULL;
        if (p == NULL) { slot_give(t, s); PyErr_NoMemory(); return NULL; }
        int64_t r = toks_scratch_init(t->ctx, p, bytes, t->scr_flags);
        if (r < 0) {
            PyMem_RawFree(p);
            slot_give(t, s);
            fail(r, "toks_scratch_init(%llu bytes)", (unsigned long long)bytes);
            return NULL;
        }
        PyMem_RawFree(s->scr);
        s->scr = p;
        s->scr_bytes = bytes;
        s->max_len = want;
    }
    return s;
}

/* s->ids holds n ids; 0, or -1 (no memory, nothing raised: the caller may not hold the GIL) */
static int slot_ids(slot *s, uint64_t n)
{
    if (s->ids_cap >= n) { return 0; }
    uint64_t want = n < 4096u ? 4096u : n;
    if (want < 2u * s->ids_cap) { want = 2u * s->ids_cap; }
    if (want > ((uint64_t)PY_SSIZE_T_MAX >> 3)) { return -1; }
    uint32_t *p = (uint32_t *)PyMem_RawMalloc((size_t)want * 4u);
    if (p == NULL) { return -1; }
    PyMem_RawFree(s->ids);
    s->ids = p;
    s->ids_cap = want;
    return 0;
}

/* s->buf holds n bytes; 0 or -1 as slot_ids */
static int slot_buf(slot *s, uint64_t n)
{
    if (s->buf_cap >= n) { return 0; }
    uint64_t want = n < 16384u ? 16384u : n;
    if (want < 2u * s->buf_cap) { want = 2u * s->buf_cap; }
    if (want > ((uint64_t)PY_SSIZE_T_MAX >> 1)) { return -1; }
    uint8_t *p = (uint8_t *)PyMem_RawMalloc((size_t)want);
    if (p == NULL) { return -1; }
    PyMem_RawFree(s->buf);
    s->buf = p;
    s->buf_cap = want;
    return 0;
}

/* ======================================================================================================
 * inputs: texts and id sequences
 * ====================================================================================================== */

typedef struct text_in {
    const char *p;
    Py_ssize_t  n;
    Py_buffer   view;
    int         has_view;
} text_in;

/* the bytes of a str (its utf-8), a bytes object or any contiguous buffer, in place. */
static int text_get(PyObject *o, text_in *t)
{
    t->has_view = 0;
    if (PyUnicode_Check(o)) {
        t->p = PyUnicode_AsUTF8AndSize(o, &t->n);
        return t->p != NULL ? 0 : -1;
    }
    if (PyBytes_Check(o)) {
        t->p = PyBytes_AS_STRING(o);
        t->n = PyBytes_GET_SIZE(o);
        return 0;
    }
    if (PyObject_CheckBuffer(o)) {
        if (PyObject_GetBuffer(o, &t->view, PyBUF_SIMPLE) < 0) { return -1; }
        t->has_view = 1;
        t->p = (const char *)t->view.buf;
        t->n = t->view.len;
        return 0;
    }
    PyErr_Format(PyExc_TypeError, "text must be str or a bytes-like object, not %.200s", Py_TYPE(o)->tp_name);
    return -1;
}

static void text_release(text_in *t)
{
    if (t->has_view) {
        PyBuffer_Release(&t->view);
        t->has_view = 0;
    }
}

/* one id from an int-like object; OverflowError outside [0, 2^32) as hf's u32 conversion. */
static int id_of(PyObject *o, uint32_t *v)
{
#if PY_VERSION_HEX >= 0x030C0000
    if (PyLong_CheckExact(o) && PyUnstable_Long_IsCompact((PyLongObject *)o)) {
        Py_ssize_t x = PyUnstable_Long_CompactValue((PyLongObject *)o);    /* |x| < 2^30 */
        if (x >= 0) { *v = (uint32_t)x; return 0; }
        PyErr_SetString(PyExc_OverflowError, "can't convert negative int to an id");
        return -1;
    }
#endif
    PyObject *i = PyNumber_Index(o);
    if (i == NULL) { return -1; }
    int over = 0;
    long long x = PyLong_AsLongLongAndOverflow(i, &over);
    Py_DECREF(i);
    if (x == -1 && over == 0 && PyErr_Occurred()) { return -1; }
    if (over < 0 || (over == 0 && x < 0)) {
        PyErr_SetString(PyExc_OverflowError, "can't convert negative int to an id");
        return -1;
    }
    if (over > 0 || x > 0xFFFFFFFFll) {
        PyErr_SetString(PyExc_OverflowError, "id does not fit in 32 bits");
        return -1;
    }
    *v = (uint32_t)x;
    return 0;
}

typedef struct ids_in {
    const uint32_t *p;
    Py_ssize_t      n;
    uint32_t       *owned;      /* PyMem_Malloc'd conversion, or NULL */
    Py_buffer       view;
    int             has_view;
} ids_in;

static void ids_release(ids_in *in)
{
    if (in->has_view) { PyBuffer_Release(&in->view); in->has_view = 0; }
    PyMem_Free(in->owned);
    in->owned = NULL;
}

/* the destination for n converted ids: the stack array or a new allocation */
static uint32_t *ids_dst(ids_in *in, Py_ssize_t n, uint32_t *stack, Py_ssize_t stack_n)
{
    if (n <= stack_n) { return stack; }
    if ((size_t)n > PY_SSIZE_T_MAX / 4u) { PyErr_NoMemory(); return NULL; }
    in->owned = (uint32_t *)PyMem_Malloc((size_t)n * 4u);
    if (in->owned == NULL) { PyErr_NoMemory(); }
    return in->owned;
}

/* ids from a tuple (items cannot change under __index__) */
static int ids_from_tuple(PyObject *tup, ids_in *in, uint32_t *stack, Py_ssize_t stack_n)
{
    Py_ssize_t n = PyTuple_GET_SIZE(tup);
    uint32_t *d = ids_dst(in, n, stack, stack_n);
    if (d == NULL) { return -1; }
    for (Py_ssize_t i = 0; i < n; i++) {
        if (id_of(PyTuple_GET_ITEM(tup, i), &d[i]) < 0) { return -1; }
    }
    in->p = d;
    in->n = n;
    return 0;
}

/* a buffer of integers (array.array, numpy, memoryview): 'I' (4-byte unsigned) is used in place; any
 * other integer format is converted with the same range rule as ints. */
static int ids_from_buffer(PyObject *o, ids_in *in, uint32_t *stack, Py_ssize_t stack_n)
{
    if (PyObject_GetBuffer(o, &in->view, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) < 0) { return -1; }
    in->has_view = 1;
    const char *f = in->view.format != NULL ? in->view.format : "B";
    if (*f == '@' || *f == '=' || *f == '<') { f++; }
    Py_ssize_t isz = in->view.itemsize;
    int code = (f[0] != 0 && f[1] == 0) ? f[0] : 0;
    int sgn;
    switch (code) {
    case 'b': case 'h': case 'i': case 'l': case 'q': case 'n': sgn = 1; break;
    case 'B': case 'H': case 'I': case 'L': case 'Q': case 'N': sgn = 0; break;
    default: sgn = -1; break;
    }
    if (sgn < 0 || (isz != 1 && isz != 2 && isz != 4 && isz != 8) || in->view.ndim > 1) {
        PyErr_Format(PyExc_TypeError, "ids must be integers, not a buffer of format '%s'",
                     in->view.format != NULL ? in->view.format : "B");
        return -1;
    }
    Py_ssize_t n = in->view.len / isz;
    if (sgn == 0 && isz == 4) {                                 /* uint32: in place */
        in->p = (const uint32_t *)in->view.buf;
        in->n = n;
        return 0;
    }
    uint32_t *d = ids_dst(in, n, stack, stack_n);
    if (d == NULL) { return -1; }
    const uint8_t *b = (const uint8_t *)in->view.buf;
    for (Py_ssize_t i = 0; i < n; i++) {
        int64_t x;
        int big = 0;
        if (isz == 1) { x = sgn ? (int64_t)((const int8_t *)b)[i] : (int64_t)b[i]; }
        else if (isz == 2) {
            uint16_t u; memcpy(&u, b + 2 * i, 2);
            x = sgn ? (int64_t)(int16_t)u : (int64_t)u;
        } else if (isz == 4) {
            uint32_t u; memcpy(&u, b + 4 * i, 4);
            x = (int64_t)(int32_t)u;                            /* sgn: uint32 took the in-place path */
        } else {
            uint64_t u; memcpy(&u, b + 8 * i, 8);
            big = !sgn && u > 0xFFFFFFFFull;
            x = big ? 0 : (int64_t)u;
        }
        if (big || x > 0xFFFFFFFFll) {
            PyErr_SetString(PyExc_OverflowError, "id does not fit in 32 bits");
            return -1;
        }
        if (x < 0) {
            PyErr_SetString(PyExc_OverflowError, "can't convert negative int to an id");
            return -1;
        }
        d[i] = (uint32_t)x;
    }
    in->p = d;
    in->n = n;
    return 0;
}

/* ids from a list / tuple of ints (fast), a buffer of integers, or any iterable of ints. */
static int ids_get(PyObject *o, ids_in *in, uint32_t *stack, Py_ssize_t stack_n)
{
    memset(in, 0, sizeof *in);
    if (PyUnicode_Check(o) || PyBytes_Check(o) || PyByteArray_Check(o)) {
        PyErr_Format(PyExc_TypeError, "ids must be a sequence of ints, not %.200s", Py_TYPE(o)->tp_name);
        return -1;
    }
#if PY_VERSION_HEX >= 0x030C0000
    if (PyList_CheckExact(o) || PyTuple_CheckExact(o)) {
        Py_ssize_t n = PySequence_Fast_GET_SIZE(o);
        PyObject **it = PySequence_Fast_ITEMS(o);
        Py_ssize_t i = 0;
        uint32_t *d = ids_dst(in, n, stack, stack_n);
        if (d == NULL) { return -1; }
        for (; i < n; i++) {                                    /* exact small ints: no python code runs */
            PyObject *x = it[i];
            if (!PyLong_CheckExact(x) || !PyUnstable_Long_IsCompact((PyLongObject *)x)) { break; }
            Py_ssize_t v = PyUnstable_Long_CompactValue((PyLongObject *)x);
            if (v < 0) { break; }
            d[i] = (uint32_t)v;
        }
        if (i == n) { in->p = d; in->n = n; return 0; }
        PyMem_Free(in->owned);
        in->owned = NULL;
    }
#endif
    if (PyObject_CheckBuffer(o)) { return ids_from_buffer(o, in, stack, stack_n); }
    PyObject *tup = PySequence_Tuple(o);                       /* a snapshot: __index__ may run code */
    if (tup == NULL) {
        if (PyErr_ExceptionMatches(PyExc_TypeError)) {
            PyErr_Clear();
            PyErr_Format(PyExc_TypeError, "ids must be a sequence of ints, not %.200s", Py_TYPE(o)->tp_name);
        }
        return -1;
    }
    int r = ids_from_tuple(tup, in, stack, stack_n);
    Py_DECREF(tup);
    return r;
}

/* ======================================================================================================
 * outputs
 * ====================================================================================================== */

/* a list of the n ints in v, sharing the tokenizer's int objects for values < n_ids */
static PyObject *int_list(Tok *t, const uint32_t *v, int64_t n)
{
    PyObject *list = PyList_New((Py_ssize_t)n);
    if (list == NULL || n == 0) { return list; }
    PyObject **ints = t->ints;
    uint32_t nid = t->info.n_ids;
    if (ints == NULL) {
        ints = (PyObject **)PyMem_Calloc(nid != 0u ? nid : 1u, sizeof *ints);
        if (ints == NULL) { Py_DECREF(list); return PyErr_NoMemory(); }
        t->ints = ints;
    }
    for (int64_t i = 0; i < n; i++) {
        uint32_t id = v[i];
        PyObject *o;
        if (id < nid) {
            o = ints[id];
            if (o == NULL) {
                o = PyLong_FromUnsignedLong(id);
                if (o == NULL) { Py_DECREF(list); return NULL; }
                ints[id] = o;
            }
            Py_INCREF(o);
        } else {
            o = PyLong_FromUnsignedLong(id);
            if (o == NULL) { Py_DECREF(list); return NULL; }
        }
        PyList_SET_ITEM(list, (Py_ssize_t)i, o);
    }
    return list;
}

/* int_list of n ids with n_pad of the file's pad id before them (pad_left) or after them */
static PyObject *int_list_pad(Tok *t, const uint32_t *v, int64_t n, uint32_t n_pad)
{
    if (n_pad == 0u) { return int_list(t, v, n); }
    PyObject *list = PyList_New((Py_ssize_t)(n + (int64_t)n_pad));
    if (list == NULL) { return NULL; }
    PyObject *body = int_list(t, v, n);
    PyObject *pad = PyLong_FromUnsignedLong(t->info.pad_id);
    if (body == NULL || pad == NULL) { Py_XDECREF(body); Py_XDECREF(pad); Py_DECREF(list); return NULL; }
    Py_ssize_t at = t->info.pad_left ? (Py_ssize_t)n_pad : 0, p0 = t->info.pad_left ? 0 : (Py_ssize_t)n;
    for (Py_ssize_t i = 0; i < (Py_ssize_t)n; i++) {
        PyObject *o = PyList_GET_ITEM(body, i);
        Py_INCREF(o);
        PyList_SET_ITEM(list, at + i, o);
    }
    for (Py_ssize_t i = 0; i < (Py_ssize_t)n_pad; i++) {
        Py_INCREF(pad);
        PyList_SET_ITEM(list, p0 + i, pad);
    }
    Py_DECREF(pad);
    Py_DECREF(body);
    return list;
}

/* the pad ids hf adds to an encoding of r ids (template included): target = the Fixed length, else `longest` (the
 * batch's longest member, r for one text: BatchLongest), rounded up to pad_to_multiple_of; none where the call
 * applies no padding (TOKS_NO_PAD, a TOKS_CONTINUATION part: toks.h's whole-document steps). docs/algorithms/
 * wordpiece.md 7.4, the rule the core applies to one text; hf's pad_encodings over a batch. */
static uint32_t pad_count(const Tok *t, uint32_t flags, uint64_t r, uint64_t longest)
{
    const toks_info *i = &t->info;
    if (!i->pad_on || (flags & (TOKS_NO_PAD | TOKS_CONTINUATION)) != 0u) { return 0u; }
    uint64_t target = i->pad_fixed ? i->pad_len : longest;
    if (i->pad_multiple != 0u && target % i->pad_multiple != 0u) { target += i->pad_multiple - target % i->pad_multiple; }
    return target > r ? (uint32_t)(target - r) : 0u;
}

/* ======================================================================================================
 * Encoding: hf's Encoding of one text (encode_ex, encode_batch_ex). The ids are a list; the masks and type ids are
 * made on access from the layout [template prefix][text][template suffix] with the pads on the file's side, and
 * tokens on first access (toks._vocab.tokens: hf's strings, read from the file).
 * ====================================================================================================== */

typedef struct {
    PyObject_HEAD
    Tok      *tok;
    PyObject *ids;          /* list of the ids, pads included */
    PyObject *text;         /* the text as str or bytes (another buffer is copied): tokens read it */
    PyObject *tokens;       /* list of str on first access, else NULL */
    uint32_t  flags;        /* the encode's flags (toks.h) */
    uint32_t  n_pre, n_suf; /* the template's ids before / after the text's */
    uint32_t  n_pad;        /* pad ids, before the rest under pad_left, else after */
    uint8_t   pad_left;
} Enc;

/* an Encoding of r ids in v (the template's included, not padded) and n_pad pads; text is the object encoded
 * (tx: its bytes) */
static PyObject *enc_new(Tok *t, const uint32_t *v, int64_t r, uint32_t n_pad, uint32_t flags, PyObject *text,
                         const text_in *tx)
{
    PyObject *keep = (PyUnicode_Check(text) || PyBytes_Check(text)) ? Py_NewRef(text)
                                                                     : PyBytes_FromStringAndSize(tx->p, tx->n);
    if (keep == NULL) { return NULL; }
    PyObject *ids = int_list_pad(t, v, r, n_pad);
    if (ids == NULL) { Py_DECREF(keep); return NULL; }
    Enc *e = PyObject_New(Enc, &EncodingType);
    if (e == NULL) { Py_DECREF(keep); Py_DECREF(ids); return NULL; }
    e->tok = (Tok *)Py_NewRef((PyObject *)t);
    e->ids = ids;
    e->text = keep;
    e->tokens = NULL;
    e->flags = flags;
    int post = (flags & TOKS_NO_POSTPROCESS) == 0u && (uint64_t)r >= t->tmpl_n;   /* the core writes it whole */
    e->n_pre = post ? t->tmpl_pre : 0u;
    e->n_suf = post ? t->tmpl_n - t->tmpl_pre : 0u;
    e->n_pad = n_pad;
    e->pad_left = (uint8_t)(t->info.pad_left != 0u);
    return (PyObject *)e;
}

enum { E_TYPES, E_ATTENTION, E_SPECIAL };

/* type_ids / attention_mask / special_tokens_mask: the template's type ids around seq_type_id, pads pad_type_id;
 * attention 0 on pads only; special 1 on the template and the pads */
static PyObject *enc_list(Enc *e, int what)
{
    const Tok *t = e->tok;
    Py_ssize_t n = PyList_GET_SIZE(e->ids);
    PyObject *list = PyList_New(n);
    if (list == NULL) { return NULL; }
    Py_ssize_t b0 = e->pad_left ? (Py_ssize_t)e->n_pad : 0, b1 = e->pad_left ? n : n - (Py_ssize_t)e->n_pad;
    Py_ssize_t s0 = b0 + (Py_ssize_t)e->n_pre, s1 = b1 - (Py_ssize_t)e->n_suf;   /* the text's ids: [s0, s1) */
    const uint32_t *types = t->tmpl != NULL ? t->tmpl + t->tmpl_n : NULL;
    for (Py_ssize_t i = 0; i < n; i++) {
        unsigned long v;
        if (i < b0 || i >= b1) {
            v = what == E_TYPES ? t->info.pad_type_id : what == E_ATTENTION ? 0u : 1u;
        } else if (i >= s0 && i < s1) {
            v = what == E_TYPES ? t->info.seq_type_id : what == E_ATTENTION ? 1u : 0u;
        } else {
            Py_ssize_t k = i < s0 ? i - b0 : (Py_ssize_t)t->tmpl_pre + (i - s1);   /* the template's position */
            v = what == E_TYPES ? (types != NULL ? types[k] : 0u) : 1u;
        }
        PyObject *o = PyLong_FromUnsignedLong(v);
        if (o == NULL) { Py_DECREF(list); return NULL; }
        PyList_SET_ITEM(list, i, o);
    }
    return list;
}

static PyObject *Enc_ids(Enc *e, void *c) { (void)c; return PyList_GetSlice(e->ids, 0, PY_SSIZE_T_MAX); }
static PyObject *Enc_type_ids(Enc *e, void *c) { (void)c; return enc_list(e, E_TYPES); }
static PyObject *Enc_attention(Enc *e, void *c) { (void)c; return enc_list(e, E_ATTENTION); }
static PyObject *Enc_special(Enc *e, void *c) { (void)c; return enc_list(e, E_SPECIAL); }
static PyObject *Enc_overflowing(Enc *e, void *c) { (void)e; (void)c; return PyList_New(0); }
static PyObject *Enc_n_sequences(Enc *e, void *c) { (void)e; (void)c; return PyLong_FromLong(1); }
static PyObject *Enc_text(Enc *e, void *c) { (void)c; return Py_NewRef(e->text); }

static PyObject *Enc_layout(Enc *e, void *c)
{
    (void)c;
    return Py_BuildValue("(IIIiI)", (unsigned)e->n_pre, (unsigned)e->n_suf, (unsigned)e->n_pad, (int)e->pad_left,
                         (unsigned)e->flags);
}

static PyObject *Enc_tokens(Enc *e, void *c)
{
    (void)c;
    if (e->tokens == NULL) {
        PyObject *mod = PyImport_ImportModule("toks._vocab");
        if (mod == NULL) { return NULL; }
        PyObject *v = PyObject_CallMethod(mod, "tokens", "OO", (PyObject *)e->tok, (PyObject *)e);
        Py_DECREF(mod);
        if (v == NULL) { return NULL; }
        if (!PyList_Check(v) || PyList_GET_SIZE(v) != PyList_GET_SIZE(e->ids)) {
            Py_DECREF(v);
            PyErr_SetString(PyExc_RuntimeError, "toks._vocab.tokens returned something else than a list per id");
            return NULL;
        }
        if (e->tokens == NULL) { e->tokens = v; } else { Py_DECREF(v); }
    }
    return PyList_GetSlice(e->tokens, 0, PY_SSIZE_T_MAX);
}

static Py_ssize_t Enc_len(Enc *e) { return PyList_GET_SIZE(e->ids); }

static PyObject *Enc_repr(Enc *e)
{
    return PyUnicode_FromFormat("Encoding(num_tokens=%zd, attributes=[ids, type_ids, tokens, attention_mask, "
                                "special_tokens_mask, overflowing])", PyList_GET_SIZE(e->ids));
}

static void Enc_dealloc(Enc *e)
{
    Py_XDECREF(e->tok);
    Py_XDECREF(e->ids);
    Py_XDECREF(e->text);
    Py_XDECREF(e->tokens);
    PyObject_Free(e);
}

/* ======================================================================================================
 * encode / pieces
 * ====================================================================================================== */

/* flags from add_special_tokens / added_tokens / continuation (absent arguments: NULL) */
static int encode_flags(Tok *t, PyObject *ast, PyObject *mode, PyObject *cont, uint32_t *flags)
{
    uint32_t f = t->encode_special ? TOKS_ADDED_NONSPECIAL : TOKS_ADDED_ALL;
    if (mode != NULL && mode != Py_None) {
        if (mode == s_all || (PyUnicode_Check(mode) && PyUnicode_Compare(mode, s_all) == 0)) {
            f = TOKS_ADDED_ALL;
        } else if (mode == s_nonspecial || (PyUnicode_Check(mode) && PyUnicode_Compare(mode, s_nonspecial) == 0)) {
            f = TOKS_ADDED_NONSPECIAL;
        } else if (mode == s_none || (PyUnicode_Check(mode) && PyUnicode_Compare(mode, s_none) == 0)) {
            f = TOKS_ADDED_NONE;
        } else {
            PyErr_Format(PyExc_ValueError, "added_tokens must be 'all', 'nonspecial' or 'none', not %R", mode);
            return -1;
        }
    }
    int a = truth(ast, 1), c = truth(cont, 0);
    if (a < 0 || c < 0) { return -1; }
    if (!a) { f |= TOKS_NO_POSTPROCESS; }
    if (c) { f |= TOKS_CONTINUATION; }
    *flags = f;
    return 0;
}

/* toks_encode / toks_pieces, without the GIL for long texts */
static int64_t core_run(Tok *t, const text_in *tx, uint32_t flags, uint32_t *out, uint64_t cap, void *scr,
                        int pieces)
{
    int64_t r;
    uint64_t n = (uint64_t)tx->n;
    if (n >= GIL_TEXT) {
        Py_BEGIN_ALLOW_THREADS
        r = pieces ? toks_pieces(t->ctx, tx->p, n, flags, out, cap, scr)
                   : toks_encode(t->ctx, tx->p, n, flags, out, cap, scr);
        Py_END_ALLOW_THREADS
    } else {
        r = pieces ? toks_pieces(t->ctx, tx->p, n, flags, out, cap, scr)
                   : toks_encode(t->ctx, tx->p, n, flags, out, cap, scr);
    }
    return r;
}

enum { OUT_IDS, OUT_PIECES, OUT_ENC };

/* encode / pieces of one text: a list of ids (OUT_IDS), of piece ends (OUT_PIECES), or an Encoding (OUT_ENC: the core
 * leaves the padding to it, so it knows which ids are pads) */
static PyObject *one_out(Tok *t, PyObject *text, uint32_t flags, int kind)
{
    int pieces = kind == OUT_PIECES;
    const char *what = pieces ? "pieces" : kind == OUT_ENC ? "encode_ex" : "encode";
    uint32_t cf = kind == OUT_ENC ? flags | TOKS_NO_PAD : flags;
    text_in tx;
    if (text_get(text, &tx) < 0) { return NULL; }
    PyObject *res = NULL;
    uint64_t len = (uint64_t)tx.n;
    if (len > TOKS_MAX_TEXT) {                  /* the core's TOKS_E_LIMIT (or TOKS_E_ARG), no allocation */
        int64_t r = pieces ? toks_pieces(t->ctx, tx.p, len, cf, NULL, 0, NULL)
                           : toks_encode(t->ctx, tx.p, len, cf, NULL, 0, NULL);
        fail(r < 0 ? r : TOKS_E_LIMIT, "%s: a text of %llu bytes (the limit is %llu)", what,
             (unsigned long long)len, (unsigned long long)TOKS_MAX_TEXT);
        text_release(&tx);
        return NULL;
    }
    slot *s = slot_take(t, len, 1);
    if (s == NULL) { text_release(&tx); return NULL; }
    if (slot_ids(s, len + 64u) < 0) { PyErr_NoMemory(); goto give; }
    int64_t r = core_run(t, &tx, cf, s->ids, s->ids_cap, s->scr, pieces);
    if (r > (int64_t)s->ids_cap) {              /* more ids than bytes + 64 (an expanding normalizer) */
        if (slot_ids(s, (uint64_t)r) < 0) { PyErr_NoMemory(); goto give; }
        r = core_run(t, &tx, cf, s->ids, s->ids_cap, s->scr, pieces);
    }
    if (r < 0) {
        fail(r, "%s(flags %u) on a text of %llu bytes", what, (unsigned)cf, (unsigned long long)len);
        goto give;
    }
    res = kind == OUT_ENC ? enc_new(t, s->ids, r, pad_count(t, flags, (uint64_t)r, (uint64_t)r), flags, text, &tx)
                          : int_list(t, s->ids, r);
give:
    slot_give(t, s);
    text_release(&tx);
    return res;
}

/* ======================================================================================================
 * the tiktoken view (tiktoken 0.14.0's Encoding): its special tokens are the file's own, toks._vocab.tiktoken_view
 * ====================================================================================================== */

#define TT_RAW (TOKS_NO_POSTPROCESS | TOKS_NO_TRUNCATE | TOKS_NO_PAD)   /* tiktoken has no template, cut or pad */

/* t->tt and t->tt_bits on first use (GIL held): 0, or -1 with an exception */
static int tt_get(Tok *t)
{
    if (t->tt != NULL) { return 0; }
    PyObject *entries = PyList_New(0), *mod = NULL, *v = NULL;
    uint8_t *bits = NULL;
    if (entries == NULL) { return -1; }
    for (uint32_t i = 0;; i++) {                /* bound: toks_added answers TOKS_E_ARG past its last entry */
        const void *c = NULL;
        uint64_t len = 0;
        uint32_t id = 0;
        int64_t f = toks_added(t->ctx, i, &c, &len, &id);
        if (f == TOKS_E_ARG) { break; }
        if (f < 0) { fail(f, "toks_added(%u)", (unsigned)i); goto bad; }
        int64_t idf = toks_id_flags(t->ctx, id);
        PyObject *e = Py_BuildValue("(Is#LL)", (unsigned)id, (const char *)c, (Py_ssize_t)len, (long long)f,
                                    (long long)idf);
        if (e == NULL || PyList_Append(entries, e) < 0) { Py_XDECREF(e); goto bad; }
        Py_DECREF(e);
    }
    mod = PyImport_ImportModule("toks._vocab");
    if (mod == NULL) { goto bad; }
    v = PyObject_CallMethod(mod, "tiktoken_view", "OiOO", (PyObject *)t, (int)t->kind, t->source, entries);
    if (v == NULL) { goto bad; }
    if (!PyTuple_Check(v) || PyTuple_GET_SIZE(v) != 3 || !PyDict_Check(PyTuple_GET_ITEM(v, 0))) {
        PyErr_SetString(PyExc_RuntimeError, "toks._vocab.tiktoken_view returned something else than its 3-tuple");
        goto bad;
    }
    bits = (uint8_t *)PyMem_Calloc(((size_t)t->info.n_ids >> 3) + 1u, 1u);
    if (bits == NULL) { PyErr_NoMemory(); goto bad; }
    Py_ssize_t pos = 0;
    PyObject *k, *val;
    while (PyDict_Next(PyTuple_GET_ITEM(v, 0), &pos, &k, &val)) {
        unsigned long id = PyLong_AsUnsignedLong(k);
        if (PyErr_Occurred()) { goto bad; }
        if (id < t->info.n_ids) { bits[id >> 3] = (uint8_t)(bits[id >> 3] | (1u << (id & 7u))); }
    }
    Py_DECREF(entries);
    Py_DECREF(mod);
    if (t->tt == NULL) {                        /* another thread may have got there first (the call can switch) */
        t->tt = v;
        t->tt_bits = bits;
    } else {
        Py_DECREF(v);
        PyMem_Free(bits);
    }
    return 0;
bad:
    Py_DECREF(entries);
    Py_XDECREF(mod);
    Py_XDECREF(v);
    PyMem_Free(bits);
    return -1;
}

/* the text tiktoken encodes: a str with lone surrogates (no utf-8) goes through tiktoken's fix-up,
 * text.encode("utf-16", "surrogatepass").decode("utf-16", "replace"); a new reference */
static PyObject *tt_text(PyObject *text)
{
    if (!PyUnicode_Check(text) || PyUnicode_AsUTF8AndSize(text, NULL) != NULL) { return Py_NewRef(text); }
    if (!PyErr_ExceptionMatches(PyExc_UnicodeEncodeError)) { return NULL; }
    PyErr_Clear();
    PyObject *b = PyUnicode_AsEncodedString(text, "utf-16", "surrogatepass");
    if (b == NULL) { return NULL; }
    PyObject *s = PyUnicode_Decode(PyBytes_AS_STRING(b), PyBytes_GET_SIZE(b), "utf-16", "replace");
    Py_DECREF(b);
    return s;
}

static int is_all(PyObject *o)
{
    return o != NULL && PyUnicode_Check(o) && PyUnicode_CompareWithASCIIString(o, "all") == 0;
}

/* where the text holds the string c: its first index (characters of a str, bytes of anything else), -1 nowhere,
 * -2 with an exception */
static Py_ssize_t tt_find(PyObject *text, PyObject *c)
{
    if (!PyUnicode_Check(c)) { return -1; }
    if (PyUnicode_Check(text)) { return PyUnicode_Find(text, c, 0, PY_SSIZE_T_MAX, 1); }
    PyObject *hay = PyBytes_Check(text) ? Py_NewRef(text) : PyBytes_FromObject(text);
    PyObject *needle = hay != NULL ? PyUnicode_AsUTF8String(c) : NULL;
    PyObject *r = needle != NULL ? PyObject_CallMethod(hay, "find", "O", needle) : NULL;
    Py_ssize_t at = r != NULL ? PyLong_AsSsize_t(r) : -2;
    Py_XDECREF(r);
    Py_XDECREF(needle);
    Py_XDECREF(hay);
    return (at == -1 && PyErr_Occurred()) ? -2 : at;
}

/* encode(text, allowed_special=..., disallowed_special=...): tiktoken's. disallowed "all" = the specials not allowed,
 * and the text holding a disallowed string is tiktoken's ValueError (the leftmost one named). The encoding with
 * every special recognized (ALL) finds the specials: an id of tt's set in it is the text's special when the text
 * holds its string (else the model wrote it: WordPiece's [UNK], Unigram's <unk>), and that encoding is the answer
 * when each special the text holds is allowed; a disallowed string that is no special is looked for in the text, as
 * tiktoken's regex does; an allowed set that leaves a present special as text goes through
 * toks._vocab.encode_subset (tiktoken's own split) */
static PyObject *tt_encode(Tok *t, PyObject *text, PyObject *allowed, PyObject *disallowed)
{
    if (tt_get(t) < 0) { return NULL; }
    PyObject *byid = PyTuple_GET_ITEM(t->tt, 0), *bystr = PyTuple_GET_ITEM(t->tt, 1);
    unsigned long ord = PyLong_AsUnsignedLong(PyTuple_GET_ITEM(t->tt, 2));
    if (PyErr_Occurred()) { return NULL; }
    int a_all = is_all(allowed), d_all = disallowed == NULL || is_all(disallowed);
    PyObject *aset = NULL, *dset = NULL, *ids = NULL, *res = NULL, *fixed = NULL, *seen = NULL, *bad = NULL;
    if (!a_all && (aset = PySet_New(allowed)) == NULL) { return NULL; }      /* allowed NULL: set() */
    if (!d_all && (dset = PySet_New(disallowed)) == NULL) { goto done; }
    fixed = tt_text(text);
    if (fixed == NULL) { goto done; }
    if (!a_all && PySet_GET_SIZE(aset) == 0 && dset != NULL && PySet_GET_SIZE(dset) == 0) {
        res = one_out(t, fixed, (uint32_t)ord | TT_RAW, OUT_IDS);              /* nothing allowed, nothing checked */
        goto done;
    }
    ids = one_out(t, fixed, TOKS_ADDED_ALL | TT_RAW, OUT_IDS);
    if (ids == NULL || (a_all && dset == NULL)) { res = ids; ids = NULL; goto done; }
    if ((seen = PyDict_New()) == NULL) { goto done; }      /* special -> where the text holds it */
    int mixed = 0;
    Py_ssize_t best = -1;                                  /* the leftmost disallowed string the text holds: bad */
    for (Py_ssize_t i = 0, n = PyList_GET_SIZE(ids); i < n; i++) {
        PyObject *key = PyList_GET_ITEM(ids, i);
        unsigned long id = PyLong_AsUnsignedLong(key);
        if (id >= t->info.n_ids || !(t->tt_bits[id >> 3] & (1u << (id & 7u)))) { continue; }
        PyObject *c = PyDict_GetItemWithError(byid, key), *w;
        if (c == NULL) { if (!PyErr_Occurred()) { continue; } goto done; }
        Py_ssize_t at;
        if ((w = PyDict_GetItemWithError(seen, c)) != NULL) {
            at = PyLong_AsSsize_t(w);
        } else {
            if (PyErr_Occurred() || (at = tt_find(fixed, c)) == -2) { goto done; }
            PyObject *v = PyLong_FromSsize_t(at);
            if (v == NULL || PyDict_SetItem(seen, c, v) < 0) { Py_XDECREF(v); goto done; }
            Py_DECREF(v);
        }
        if (at < 0) { continue; }                          /* the model's unknown piece, not the text's special */
        int in_a = a_all ? 1 : PySet_Contains(aset, c);
        int in_d = dset != NULL ? PySet_Contains(dset, c) : !in_a;
        if (in_a < 0 || in_d < 0) { goto done; }
        if (in_d && (best < 0 || at < best)) { best = at; Py_XSETREF(bad, Py_NewRef(c)); }
        if (!in_d && !in_a) { mixed = 1; }
    }
    if (dset != NULL) {                         /* a disallowed string that is not a special: tiktoken's search */
        PyObject *it = PyObject_GetIter(dset), *s;
        if (it == NULL) { goto done; }
        while ((s = PyIter_Next(it)) != NULL) {
            int sp = PyDict_Contains(bystr, s);
            Py_ssize_t at = sp == 0 ? tt_find(fixed, s) : -1;
            if (sp < 0 || at == -2) { Py_DECREF(s); Py_DECREF(it); goto done; }
            if (at >= 0 && (best < 0 || at < best)) { best = at; Py_XSETREF(bad, Py_NewRef(s)); }
            Py_DECREF(s);
        }
        Py_DECREF(it);
        if (PyErr_Occurred()) { goto done; }
    }
    if (bad != NULL) {
        PyObject *mod = PyImport_ImportModule("toks._vocab");
        PyObject *exc = mod != NULL ? PyObject_CallMethod(mod, "disallowed", "O", bad) : NULL;
        Py_XDECREF(mod);
        if (exc != NULL) { PyErr_SetObject((PyObject *)Py_TYPE(exc), exc); Py_DECREF(exc); }
        goto done;
    }
    if (!mixed) { res = ids; ids = NULL; goto done; }
    PyObject *keep = PyDict_New();              /* the allowed specials: {content: id} */
    if (keep == NULL) { goto done; }
    Py_ssize_t pos = 0;
    PyObject *k, *val;
    while (PyDict_Next(bystr, &pos, &k, &val)) {
        int in = PySet_Contains(aset, k);
        if (in < 0 || (in && PyDict_SetItem(keep, k, val) < 0)) { Py_DECREF(keep); goto done; }
    }
    PyObject *mod = PyImport_ImportModule("toks._vocab");
    res = mod != NULL ? PyObject_CallMethod(mod, "encode_subset", "OOO", (PyObject *)t, fixed, keep) : NULL;
    Py_XDECREF(mod);
    Py_DECREF(keep);
done:
    Py_XDECREF(aset);
    Py_XDECREF(dset);
    Py_XDECREF(ids);
    Py_XDECREF(fixed);
    Py_XDECREF(seen);
    Py_XDECREF(bad);
    return res;
}

static PyObject *Tok_encode(Tok *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[6];
    uint32_t flags;
    if (parse(&A_ENCODE, args, nargs, kwnames, a) < 0) { return NULL; }
    if (a[4] != NULL || a[5] != NULL) {         /* tiktoken's keywords: its semantics, nothing of hf's */
        if (a[1] != NULL || a[2] != NULL || a[3] != NULL) {
            PyErr_SetString(PyExc_TypeError, "encode: allowed_special / disallowed_special (tiktoken's) do not combine "
                            "with add_special_tokens, added_tokens or continuation");
            return NULL;
        }
        return tt_encode(self, a[0], a[4], a[5]);
    }
    if (encode_flags(self, a[1], a[2], a[3], &flags) < 0) { return NULL; }
    return one_out(self, a[0], flags, OUT_IDS);
}

/* encode_ordinary(text): tiktoken's, no special recognized (the file's ordinary mode), no template, cut or pad */
static PyObject *Tok_encode_ordinary(Tok *self, PyObject *text)
{
    if (tt_get(self) < 0) { return NULL; }
    unsigned long ord = PyLong_AsUnsignedLong(PyTuple_GET_ITEM(self->tt, 2));
    if (PyErr_Occurred()) { return NULL; }
    PyObject *fixed = tt_text(text);
    if (fixed == NULL) { return NULL; }
    PyObject *res = one_out(self, fixed, (uint32_t)ord | TT_RAW, OUT_IDS);
    Py_DECREF(fixed);
    return res;
}

static PyObject *Tok_encode_ex(Tok *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[4];
    uint32_t flags;
    if (parse(&A_ENCODE_EX, args, nargs, kwnames, a) < 0 || encode_flags(self, a[1], a[2], a[3], &flags) < 0) {
        return NULL;
    }
    return one_out(self, a[0], flags, OUT_ENC);
}

static PyObject *Tok_pieces(Tok *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[3];
    uint32_t flags;
    if (parse(&A_PIECES, args, nargs, kwnames, a) < 0 || encode_flags(self, NULL, a[1], a[2], &flags) < 0) {
        return NULL;
    }
    return one_out(self, a[0], flags, OUT_PIECES);
}

/* encode_into(text, out, *, ...) -> n: the ids go straight into out (a writable buffer of 4-byte ints);
 * out[:min(n, len(out))] holds the exact prefix (toks_encode's capacity rule). */
static PyObject *Tok_encode_into(Tok *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[5];
    uint32_t flags;
    if (parse(&A_ENCODE_INTO, args, nargs, kwnames, a) < 0 || encode_flags(self, a[2], a[3], a[4], &flags) < 0) {
        return NULL;
    }
    Py_buffer ob;
    if (PyObject_GetBuffer(a[1], &ob, PyBUF_WRITABLE | PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) < 0) { return NULL; }
    const char *f = ob.format != NULL ? ob.format : "B";
    if (*f == '@' || *f == '=' || *f == '<') { f++; }
    if (ob.itemsize != 4 || f[0] == 0 || f[1] != 0 || strchr("IiLl", f[0]) == NULL) {
        PyErr_Format(PyExc_TypeError, "out must be a writable buffer of 4-byte integers (uint32 / int32), "
                     "not format '%s' with %zd-byte items", ob.format != NULL ? ob.format : "B", ob.itemsize);
        PyBuffer_Release(&ob);
        return NULL;
    }
    text_in tx;
    if (text_get(a[0], &tx) < 0) { PyBuffer_Release(&ob); return NULL; }
    PyObject *res = NULL;
    uint64_t len = (uint64_t)tx.n, cap = (uint64_t)ob.len / 4u;
    const char *o0 = (const char *)ob.buf, *o1 = o0 + cap * 4u;
    if (cap != 0u && len != 0u && o0 < tx.p + tx.n && tx.p < o1) {
        PyErr_SetString(PyExc_ValueError, "out overlaps text");
        goto done;
    }
    if (len > TOKS_MAX_TEXT) {
        int64_t r = toks_encode(self->ctx, tx.p, len, flags, NULL, 0, NULL);
        fail(r < 0 ? r : TOKS_E_LIMIT, "encode_into: a text of %llu bytes (the limit is %llu)",
             (unsigned long long)len, (unsigned long long)TOKS_MAX_TEXT);
        goto done;
    }
    slot *s = slot_take(self, len, 1);
    if (s == NULL) { goto done; }
    int64_t r = core_run(self, &tx, flags, cap != 0u ? (uint32_t *)ob.buf : NULL, cap, s->scr, 0);
    slot_give(self, s);
    if (r < 0) {
        fail(r, "encode_into(flags %u) on a text of %llu bytes", (unsigned)flags, (unsigned long long)len);
        goto done;
    }
    res = PyLong_FromLongLong(r);
done:
    text_release(&tx);
    PyBuffer_Release(&ob);
    return res;
}

/* a batch's texts and their ids: every text read in place, the core over the whole batch without the GIL, into one
 * growing array (cnt[i] ids each) */
typedef struct batch {
    PyObject   *seq;            /* the texts: a tuple (strong references for the GIL-free part) */
    Py_ssize_t  n, got;         /* texts; text_get done for the first got */
    text_in    *tx;
    int64_t    *cnt;
    uint32_t   *out;
    uint64_t    longest;        /* the most ids of one text */
} batch;

static void batch_free(batch *b)
{
    PyMem_RawFree(b->out);
    if (b->tx != NULL) {
        for (Py_ssize_t i = 0; i < b->got && i < b->n; i++) { text_release(&b->tx[i]); }
    }
    PyMem_Free(b->tx);
    PyMem_Free(b->cnt);
    Py_XDECREF(b->seq);
}

/* encodes every text of texts under flags into b; 0, or -1 with an exception (b is freed either way by batch_free) */
static int batch_run(Tok *self, const char *what, PyObject *texts, uint32_t flags, batch *b)
{
    memset(b, 0, sizeof *b);
    if (PyUnicode_Check(texts) || PyBytes_Check(texts)) {
        PyErr_Format(PyExc_TypeError, "%s takes a sequence of texts, not one text", what);
        return -1;
    }
    b->seq = PySequence_Tuple(texts);
    if (b->seq == NULL) { return -1; }
    Py_ssize_t n = b->n = PyTuple_GET_SIZE(b->seq);
    b->tx = (text_in *)PyMem_Calloc(n != 0 ? (size_t)n : 1u, sizeof *b->tx);
    b->cnt = (int64_t *)PyMem_Calloc(n != 0 ? (size_t)n : 1u, sizeof *b->cnt);
    if (b->tx == NULL || b->cnt == NULL) { PyErr_NoMemory(); return -1; }
    uint64_t total = 0, mx = 0;
    for (; b->got < n; b->got++) {
        PyObject *it = PyTuple_GET_ITEM(b->seq, b->got);
        if (!PyUnicode_Check(it) && !PyBytes_Check(it) && !PyObject_CheckBuffer(it)) {
            PyErr_Format(PyExc_TypeError, "%s: text %zd must be str or a bytes-like object, not %.200s", what,
                         b->got, Py_TYPE(it)->tp_name);
            return -1;
        }
        if (text_get(it, &b->tx[b->got]) < 0) { return -1; }
        uint64_t l = (uint64_t)b->tx[b->got].n;
        total += l;
        if (l > mx) { mx = l; }
        if (l > TOKS_MAX_TEXT) {
            int64_t r = toks_encode(self->ctx, b->tx[b->got].p, l, flags, NULL, 0, NULL);
            fail(r < 0 ? r : TOKS_E_LIMIT, "%s: text %zd has %llu bytes (the limit is %llu)", what, b->got,
                 (unsigned long long)l, (unsigned long long)TOKS_MAX_TEXT);
            b->got++;
            return -1;
        }
    }
    slot *s = slot_take(self, mx, 1);
    if (s == NULL) { return -1; }
    uint64_t cap = total + 64u * (uint64_t)n + 64u;
    if (cap > ((uint64_t)PY_SSIZE_T_MAX >> 3) ||
        (b->out = (uint32_t *)PyMem_RawMalloc((size_t)cap * 4u)) == NULL) {
        slot_give(self, s);
        PyErr_NoMemory();
        return -1;
    }
    int64_t err = 0;
    Py_ssize_t bad = -1;
    uint64_t used = 0;
    PyThreadState *ts = total >= GIL_TEXT ? PyEval_SaveThread() : NULL;
    for (Py_ssize_t i = 0; i < n; i++) {
        uint64_t l = (uint64_t)b->tx[i].n;
        int64_t r = toks_encode(self->ctx, b->tx[i].p, l, flags, b->out + used, cap - used, s->scr);
        if (r >= 0 && used + (uint64_t)r > cap) {             /* rare: grow, then the same text again */
            uint64_t want = 2u * cap + (uint64_t)r;
            uint32_t *p = want <= ((uint64_t)PY_SSIZE_T_MAX >> 3)
                              ? (uint32_t *)PyMem_RawRealloc(b->out, (size_t)want * 4u) : NULL;
            if (p == NULL) { err = TOKS_E_NOMEM; bad = i; break; }
            b->out = p;
            cap = want;
            r = toks_encode(self->ctx, b->tx[i].p, l, flags, b->out + used, cap - used, s->scr);
        }
        if (r < 0) { err = r; bad = i; break; }
        b->cnt[i] = r;
        if ((uint64_t)r > b->longest) { b->longest = (uint64_t)r; }
        used += (uint64_t)r;
    }
    if (ts != NULL) { PyEval_RestoreThread(ts); }
    slot_give(self, s);
    if (err != 0) {
        fail(err, "%s: text %zd (flags %u, %llu bytes)", what, bad, (unsigned)flags, (unsigned long long)b->tx[bad].n);
        return -1;
    }
    return 0;
}

/* encode_batch(texts, *, ...) -> list of lists, padded as hf's encode_batch pads (the file's padding over the
 * batch: Fixed, or BatchLongest to the longest member, then pad_to_multiple_of) */
static PyObject *Tok_encode_batch(Tok *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[4];
    uint32_t flags;
    batch b;
    if (parse(&A_BATCH, args, nargs, kwnames, a) < 0 || encode_flags(self, a[1], a[2], a[3], &flags) < 0) {
        return NULL;
    }
    PyObject *res = NULL;
    if (batch_run(self, "encode_batch", a[0], flags | TOKS_NO_PAD, &b) == 0 && (res = PyList_New(b.n)) != NULL) {
        uint64_t used = 0;
        for (Py_ssize_t i = 0; i < b.n; i++) {
            PyObject *l = int_list_pad(self, b.out + used, b.cnt[i],
                                       pad_count(self, flags, (uint64_t)b.cnt[i], b.longest));
            if (l == NULL) { Py_CLEAR(res); break; }
            PyList_SET_ITEM(res, i, l);
            used += (uint64_t)b.cnt[i];
        }
    }
    batch_free(&b);
    return res;
}

/* encode_batch_ex(texts, *, ...) -> list of Encoding, hf's encode_batch */
static PyObject *Tok_encode_batch_ex(Tok *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[4];
    uint32_t flags;
    batch b;
    if (parse(&A_BATCH_EX, args, nargs, kwnames, a) < 0 || encode_flags(self, a[1], a[2], a[3], &flags) < 0) {
        return NULL;
    }
    PyObject *res = NULL;
    if (batch_run(self, "encode_batch_ex", a[0], flags | TOKS_NO_PAD, &b) == 0 && (res = PyList_New(b.n)) != NULL) {
        uint64_t used = 0;
        for (Py_ssize_t i = 0; i < b.n; i++) {
            PyObject *e = enc_new(self, b.out + used, b.cnt[i], pad_count(self, flags, (uint64_t)b.cnt[i], b.longest),
                                  flags, PyTuple_GET_ITEM(b.seq, i), &b.tx[i]);
            if (e == NULL) { Py_CLEAR(res); break; }
            PyList_SET_ITEM(res, i, e);
            used += (uint64_t)b.cnt[i];
        }
    }
    batch_free(&b);
    return res;
}

/* ======================================================================================================
 * decode
 * ====================================================================================================== */

/* the TOKS_E_ID message: the first id at or beyond n_ids */
static PyObject *fail_ids(Tok *t, int64_t r, const char *what, const uint32_t *ids, Py_ssize_t n)
{
    if (r == TOKS_E_ID) {
        for (Py_ssize_t i = 0; i < n; i++) {
            if (ids[i] >= t->info.n_ids) {
                return fail(r, "%s: id %u at index %zd is beyond the table (n_ids %u)", what, (unsigned)ids[i],
                            i, (unsigned)t->info.n_ids);
            }
        }
    }
    return fail(r, "%s of %zd ids", what, n);
}

/* the str of ids (decode's core), using slot s's byte staging; with TOKS_DECODE_RAW the bytes (tiktoken's
 * decode_bytes: an id beyond the table is its KeyError) */
static PyObject *decode_ids(Tok *t, slot *s, const uint32_t *ids, Py_ssize_t n, uint32_t flags)
{
    uint64_t guess = 8u * (uint64_t)n + 16u;
    if (slot_buf(s, s->buf_cap > guess ? s->buf_cap : guess) < 0) { return PyErr_NoMemory(); }
    int64_t r;
    int nogil = n >= (Py_ssize_t)GIL_IDS, raw = (flags & TOKS_DECODE_RAW) != 0u;
    for (int pass = 0; pass < 2; pass++) {
        if (nogil) {
            Py_BEGIN_ALLOW_THREADS
            r = toks_decode(t->ctx, ids, (uint64_t)n, flags, s->buf, s->buf_cap);
            Py_END_ALLOW_THREADS
        } else {
            r = toks_decode(t->ctx, ids, (uint64_t)n, flags, s->buf, s->buf_cap);
        }
        if (r == TOKS_E_ID && raw) {
            for (Py_ssize_t i = 0; i < n; i++) {
                if (ids[i] >= t->info.n_ids) {
                    return PyErr_Format(PyExc_KeyError, "Invalid token for decoding: %u", (unsigned)ids[i]);
                }
            }
        }
        if (r < 0) { return fail_ids(t, r, raw ? "decode_bytes" : "decode", ids, n); }
        if ((uint64_t)r <= s->buf_cap) {
            return raw ? PyBytes_FromStringAndSize((const char *)s->buf, (Py_ssize_t)r)
                       : PyUnicode_DecodeUTF8((const char *)s->buf, (Py_ssize_t)r, "strict");
        }
        if (slot_buf(s, (uint64_t)r) < 0) { return PyErr_NoMemory(); }
    }
    return fail(TOKS_E_CAP, "decode: the output grew between two passes");
}

/* decode / decode_bytes of one id sequence */
static PyObject *decode_one(Tok *self, PyObject *ids, uint32_t flags)
{
    uint32_t stack[STACK_IDS];
    ids_in in;
    if (ids_get(ids, &in, stack, STACK_IDS) < 0) { ids_release(&in); return NULL; }
    PyObject *res = NULL;
    slot *s = slot_take(self, 0, 0);
    if (s != NULL) {
        res = decode_ids(self, s, in.p, in.n, flags);
        slot_give(self, s);
    }
    ids_release(&in);
    return res;
}

/* decode_batch / decode_bytes_batch */
static PyObject *decode_many(Tok *self, PyObject *seqs, uint32_t flags)
{
    PyObject *seq = PySequence_Tuple(seqs);
    if (seq == NULL) { return NULL; }
    Py_ssize_t n = PyTuple_GET_SIZE(seq);
    PyObject *res = PyList_New(n);
    slot *s = res != NULL ? slot_take(self, 0, 0) : NULL;
    if (s == NULL) { Py_XDECREF(res); Py_DECREF(seq); return NULL; }
    uint32_t stack[STACK_IDS];
    for (Py_ssize_t i = 0; i < n; i++) {
        ids_in in;
        PyObject *str = NULL;
        if (ids_get(PyTuple_GET_ITEM(seq, i), &in, stack, STACK_IDS) == 0) {
            str = decode_ids(self, s, in.p, in.n, flags);
        }
        ids_release(&in);
        if (str == NULL) { Py_CLEAR(res); break; }
        PyList_SET_ITEM(res, i, str);
    }
    slot_give(self, s);
    Py_DECREF(seq);
    return res;
}

static PyObject *Tok_decode(Tok *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[2];
    if (parse(&A_DECODE, args, nargs, kwnames, a) < 0) { return NULL; }
    int skip = truth(a[1], 1);                  /* hf: skip_special_tokens=True */
    if (skip < 0) { return NULL; }
    return decode_one(self, a[0], skip ? TOKS_SKIP_SPECIAL : 0u);
}

static PyObject *Tok_decode_batch(Tok *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[2];
    if (parse(&A_DECODE_B, args, nargs, kwnames, a) < 0) { return NULL; }
    int skip = truth(a[1], 1);
    if (skip < 0) { return NULL; }
    return decode_many(self, a[0], skip ? TOKS_SKIP_SPECIAL : 0u);
}

/* decode_bytes(ids) -> bytes: tiktoken's, the bytes the ids spell (toks.h TOKS_DECODE_RAW), every id kept */
static PyObject *Tok_decode_bytes(Tok *self, PyObject *ids)
{
    return decode_one(self, ids, TOKS_DECODE_RAW);
}

/* decode_bytes_batch(batch, *, num_threads=8) -> list[bytes]: tiktoken's (num_threads is accepted and unused) */
static PyObject *Tok_decode_bytes_batch(Tok *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[2];
    if (parse(&A_BYTES_B, args, nargs, kwnames, a) < 0) { return NULL; }
    return decode_many(self, a[0], TOKS_DECODE_RAW);
}

/* token_bytes(id) -> bytes | None: the bytes the id decodes to alone (toks_token), None if it has none */
static PyObject *Tok_token_bytes(Tok *self, PyObject *arg)
{
    uint32_t id;
    if (id_of(arg, &id) < 0) { return NULL; }
    uint64_t len = 0;
    const uint8_t *p = toks_token(self->ctx, id, &len);
    if (p == NULL) { Py_RETURN_NONE; }
    return PyBytes_FromStringAndSize((const char *)p, (Py_ssize_t)len);
}

/* token_byte_values() -> list[bytes]: tiktoken's, the bytes of every id outside the added tokens (its mergeable
 * ranks), sorted */
static PyObject *Tok_token_byte_values(Tok *self, PyObject *noargs)
{
    (void)noargs;
    PyObject *list = PyList_New(0);
    if (list == NULL) { return NULL; }
    for (uint32_t id = 0; id < self->info.n_ids; id++) {
        int64_t f = toks_id_flags(self->ctx, id);
        if (f < 0) { Py_DECREF(list); return fail(f, "token_byte_values: id %u", (unsigned)id); }
        uint64_t len = 0;
        const uint8_t *p = (f & TOKS_ID_ADDED) ? NULL : toks_token(self->ctx, id, &len);
        if (p == NULL) { continue; }
        PyObject *b = PyBytes_FromStringAndSize((const char *)p, (Py_ssize_t)len);
        if (b == NULL || PyList_Append(list, b) < 0) { Py_XDECREF(b); Py_DECREF(list); return NULL; }
        Py_DECREF(b);
    }
    if (PyList_Sort(list) < 0) { Py_DECREF(list); return NULL; }
    return list;
}

/* num_special_tokens_to_add(is_pair=False) -> int: toks_template's count, hf's; a pair is not something toks encodes */
static PyObject *Tok_num_special_tokens_to_add(Tok *self, PyObject *const *args, Py_ssize_t nargs,
                                               PyObject *kwnames)
{
    PyObject *a[1];
    if (parse(&A_NSPECIAL, args, nargs, kwnames, a) < 0) { return NULL; }
    int pair = truth(a[0], 0);
    if (pair < 0) { return NULL; }
    if (pair) { return fail(TOKS_E_UNSUPPORTED, "num_special_tokens_to_add(is_pair=True): toks encodes one sequence"); }
    return PyLong_FromUnsignedLong(self->tmpl_n);
}

/* get_added_tokens_decoder() -> {id: AddedToken}: toks_added, hf's (the content listed last for an id, its options) */
static PyObject *Tok_get_added_tokens_decoder(Tok *self, PyObject *noargs)
{
    (void)noargs;
    PyObject *entries = PyList_New(0);
    if (entries == NULL) { return NULL; }
    for (uint32_t i = 0;; i++) {                /* bound: toks_added answers TOKS_E_ARG past its last entry */
        const void *c = NULL;
        uint64_t len = 0;
        uint32_t id = 0;
        int64_t f = toks_added(self->ctx, i, &c, &len, &id);
        if (f == TOKS_E_ARG) { break; }
        PyObject *e = f < 0 ? fail(f, "toks_added(%u)", (unsigned)i)
                            : Py_BuildValue("(Is#L)", (unsigned)id, (const char *)c, (Py_ssize_t)len, (long long)f);
        if (e == NULL || PyList_Append(entries, e) < 0) { Py_XDECREF(e); Py_DECREF(entries); return NULL; }
        Py_DECREF(e);
    }
    PyObject *mod = PyImport_ImportModule("toks._vocab");
    PyObject *res = mod != NULL ? PyObject_CallMethod(mod, "added_tokens_decoder", "O", entries) : NULL;
    Py_XDECREF(mod);
    Py_DECREF(entries);
    return res;
}

/* ======================================================================================================
 * vocabulary (token_to_id / id_to_token / get_vocab / get_vocab_size): hf's strings come from the
 * tokenizer.json (toks._vocab), the core keeps only the bytes of each id
 * ====================================================================================================== */

/* the tuple (tok2id, id2tok, model_vocab, n_model, n_total, extra); borrowed */
static PyObject *vocab_get(Tok *t)
{
    if (t->vocab != NULL) { return t->vocab; }
    PyObject *mod = PyImport_ImportModule("toks._vocab");
    if (mod == NULL) { return NULL; }
    PyObject *v = PyObject_CallMethod(mod, "load", "iOy#", (int)t->kind, t->source,
                                      (const char *)t->info.source_sha256, (Py_ssize_t)32);
    Py_DECREF(mod);
    if (v == NULL) { return NULL; }
    if (!PyTuple_Check(v) || PyTuple_GET_SIZE(v) != 6) {
        Py_DECREF(v);
        PyErr_SetString(PyExc_RuntimeError, "toks._vocab.load returned something else than its 6-tuple");
        return NULL;
    }
    if (t->vocab == NULL) { t->vocab = v; } else { Py_DECREF(v); }    /* another thread got there first */
    return t->vocab;
}

/* _vocab() -> vocab_get's tuple, for toks._vocab.tokens */
static PyObject *Tok_vocab(Tok *self, PyObject *noargs)
{
    (void)noargs;
    PyObject *v = vocab_get(self);
    return v != NULL ? Py_NewRef(v) : NULL;
}

/* token_to_id(token): a str is hf's written form (toks._vocab, from the file: "Ġhello", "<s>"); bytes-like is the
 * core's toks_token_to_id (the bytes an id decodes to: b" hello", b"<s>"); None when no id has it either way */
static PyObject *Tok_token_to_id(Tok *self, PyObject *token)
{
    if (!PyUnicode_Check(token) && PyObject_CheckBuffer(token)) {
        Py_buffer b;
        if (PyObject_GetBuffer(token, &b, PyBUF_SIMPLE) < 0) { return NULL; }
        int64_t r = toks_token_to_id(self->ctx, b.buf, (uint64_t)b.len);
        PyBuffer_Release(&b);
        if (r == TOKS_E_ID) { Py_RETURN_NONE; }
        if (r < 0) { return fail(r, "token_to_id"); }
        return PyLong_FromLongLong((long long)r);
    }
    if (!PyUnicode_Check(token)) {
        PyErr_Format(PyExc_TypeError, "token must be str or bytes, not %.200s", Py_TYPE(token)->tp_name);
        return NULL;
    }
    PyObject *v = vocab_get(self);
    if (v == NULL) { return NULL; }
    PyObject *id = PyDict_GetItemWithError(PyTuple_GET_ITEM(v, 0), token);
    if (id == NULL) {
        if (PyErr_Occurred()) { return NULL; }
        Py_RETURN_NONE;
    }
    return Py_NewRef(id);
}

/* id_flags(id) -> int: toks_id_flags, the ID_* bits (hf's added_tokens_decoder: added, its special flag; a byte
 * token); toks.Error ID for an id beyond the table, as decode */
static PyObject *Tok_id_flags(Tok *self, PyObject *arg)
{
    uint32_t id;
    if (id_of(arg, &id) < 0) { return NULL; }
    int64_t r = toks_id_flags(self->ctx, id);
    if (r < 0) { return fail(r, "id_flags(%lu)", (unsigned long)id); }
    return PyLong_FromLongLong((long long)r);
}

static PyObject *Tok_encode_bound(Tok *self, PyObject *arg)
{
    PyObject *i = PyNumber_Index(arg);
    if (i == NULL) { return NULL; }
    unsigned long long n = PyLong_AsUnsignedLongLong(i);   /* OverflowError: negative, or 2^64 and past */
    Py_DECREF(i);
    if (n == (unsigned long long)-1 && PyErr_Occurred()) { return NULL; }
    return PyLong_FromUnsignedLongLong((unsigned long long)toks_encode_bound(self->ctx, (uint64_t)n));
}

static PyObject *Tok_id_to_token(Tok *self, PyObject *arg)
{
    uint32_t id;
    if (id_of(arg, &id) < 0) { return NULL; }
    PyObject *v = vocab_get(self);
    if (v == NULL) { return NULL; }
    PyObject *key = PyLong_FromUnsignedLong(id);
    if (key == NULL) { return NULL; }
    PyObject *tok = PyDict_GetItemWithError(PyTuple_GET_ITEM(v, 1), key);
    Py_DECREF(key);
    if (tok == NULL) {
        if (PyErr_Occurred()) { return NULL; }
        Py_RETURN_NONE;
    }
    if (PyExceptionInstance_Check(tok)) {       /* a string toks._vocab cannot compute exactly: say so */
        PyErr_SetObject((PyObject *)Py_TYPE(tok), tok);
        return NULL;
    }
    return Py_NewRef(tok);
}

static PyObject *Tok_get_vocab(Tok *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[1];
    if (parse(&A_VOCAB, args, nargs, kwnames, a) < 0) { return NULL; }
    int added = truth(a[0], 1);
    if (added < 0) { return NULL; }
    PyObject *v = vocab_get(self);
    if (v == NULL) { return NULL; }
    return PyDict_Copy(PyTuple_GET_ITEM(v, added ? 0 : 2));
}

static PyObject *Tok_get_vocab_size(Tok *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[1];
    if (parse(&A_VOCAB_SIZE, args, nargs, kwnames, a) < 0) { return NULL; }
    int added = truth(a[0], 1);
    if (added < 0) { return NULL; }
    PyObject *v = vocab_get(self);
    if (v == NULL) { return NULL; }
    return Py_NewRef(PyTuple_GET_ITEM(v, added ? 4 : 3));
}

/* ======================================================================================================
 * info, repr, pickling, properties
 * ====================================================================================================== */

static const char *tier_name(uint32_t t)
{
    switch (t) {
    case TOKS_TIER_AUTO:   return "auto";
    case TOKS_TIER_SCALAR: return "scalar";
    case TOKS_TIER_NEON:   return "neon";
    case TOKS_TIER_AVX2:   return "avx2";
    case TOKS_TIER_AVX512: return "avx512";
    default:               return "unknown";
    }
}

static const char *algo_name(uint32_t a)
{
    switch (a) {
    case TOKS_ALGO_BPE_BYTELEVEL: return "bpe_bytelevel";
    case TOKS_ALGO_BPE_SPM:       return "bpe_spm";
    case TOKS_ALGO_UNIGRAM:       return "unigram";
    case TOKS_ALGO_WORDPIECE:     return "wordpiece";
    default:                      return "unknown";
    }
}

static PyObject *hex32(const uint8_t *h)
{
    int any = 0;
    for (int i = 0; i < 32; i++) { any |= h[i]; }
    if (!any) { Py_RETURN_NONE; }
    char s[65];
    static const char X[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        s[2 * i] = X[h[i] >> 4];
        s[2 * i + 1] = X[h[i] & 15];
    }
    return PyUnicode_FromStringAndSize(s, 64);
}

static PyObject *Tok_info(Tok *self, PyObject *noargs)
{
    (void)noargs;
    const toks_info *i = &self->info;
    PyObject *src = hex32(i->source_sha256), *img = hex32(i->image_sha256);
    PyObject *name = PyUnicode_DecodeUTF8(i->name, (Py_ssize_t)strnlen(i->name, sizeof i->name), "replace");
    /* abi 0.4's tail: the file's truncation and padding as the core applies them, and the template's shape */
    PyObject *trunc = i->trunc_on ? Py_BuildValue("{s:I,s:I}", "max_length", (unsigned)i->trunc_max, "stride",
                                                  (unsigned)i->trunc_stride)
                                  : Py_NewRef(Py_None);
    PyObject *pad = i->pad_on ? Py_BuildValue("{s:s,s:I,s:I,s:I,s:I,s:s}",
                                              "strategy", i->pad_fixed ? "fixed" : "batch_longest",
                                              "length", (unsigned)i->pad_len, "pad_to_multiple_of",
                                              (unsigned)i->pad_multiple, "pad_id", (unsigned)i->pad_id,
                                              "pad_type_id", (unsigned)i->pad_type_id,
                                              "direction", i->pad_left ? "left" : "right")
                              : Py_NewRef(Py_None);
    PyObject *tmpl = Py_BuildValue("{s:I,s:I,s:I}", "prefix", (unsigned)i->n_template_prefix, "suffix",
                                   (unsigned)i->n_template_suffix, "seq_type_id", (unsigned)i->seq_type_id);
    if (src == NULL || img == NULL || name == NULL || trunc == NULL || pad == NULL || tmpl == NULL) {
        Py_XDECREF(src); Py_XDECREF(img); Py_XDECREF(name); Py_XDECREF(trunc); Py_XDECREF(pad); Py_XDECREF(tmpl);
        return NULL;
    }
    return Py_BuildValue("{s:(II),s:s,s:s,s:I,s:I,s:{s:O,s:O},s:K,s:K,s:O,s:N,s:N,s:N,s:N,s:N,s:N}",
                         "abi", (unsigned)i->abi_major, (unsigned)i->abi_minor,
                         "algorithm", algo_name(i->algorithm),
                         "tier", tier_name(i->tier),
                         "n_ids", (unsigned)i->n_ids,
                         "n_added", (unsigned)i->n_added,
                         "paths", "scan", (i->paths & TOKS_PATH_SCAN) ? Py_True : Py_False,
                                  "normalize", (i->paths & TOKS_PATH_NORMALIZE) ? Py_True : Py_False,
                         "cpu_features", (unsigned long long)i->cpu_features,
                         "max_text", (unsigned long long)i->max_text,
                         "control_isolation", i->control_isolation ? Py_True : Py_False,
                         "source_sha256", src,
                         "image_sha256", img,
                         "name", name,
                         "truncation", trunc,
                         "padding", pad,
                         "template", tmpl);
}

/* ---- the tiktoken view's properties ---- */

static PyObject *Tok_get_n_vocab(Tok *self, void *closure)
{
    (void)closure;
    return PyLong_FromUnsignedLong(self->info.n_ids);        /* every id is < n_ids: max_token_value + 1 */
}

static PyObject *Tok_get_max_token_value(Tok *self, void *closure)
{
    (void)closure;
    return PyLong_FromLong((long)self->info.n_ids - 1);
}

static PyObject *Tok_get_eot_token(Tok *self, void *closure)
{
    (void)closure;
    if (tt_get(self) < 0) { return NULL; }
    PyObject *id = PyDict_GetItemString(PyTuple_GET_ITEM(self->tt, 1), "<|endoftext|>");
    if (id == NULL) {                           /* tiktoken: self._special_tokens["<|endoftext|>"] */
        PyObject *k = PyUnicode_FromString("<|endoftext|>");
        if (k != NULL) { PyErr_SetObject(PyExc_KeyError, k); Py_DECREF(k); }
        return NULL;
    }
    return Py_NewRef(id);
}

static PyObject *Tok_get_special_tokens_set(Tok *self, void *closure)
{
    (void)closure;
    if (tt_get(self) < 0) { return NULL; }
    return PySet_New(PyTuple_GET_ITEM(self->tt, 1));
}

static PyObject *Tok_get_name(Tok *self, void *closure)
{
    (void)closure;
    return PyUnicode_DecodeUTF8(self->info.name, (Py_ssize_t)strnlen(self->info.name, sizeof self->info.name),
                                "replace");
}

static PyObject *Tok_repr(Tok *self)
{
    return PyUnicode_FromFormat("<toks.Tokenizer %s: %s, %u ids, tier %s>", self->info.name,
                                algo_name(self->info.algorithm), (unsigned)self->info.n_ids,
                                tier_name(self->info.tier));
}

static PyObject *mod_load_obj = NULL;           /* toks._toks._load, for __reduce__ */

/* pickling: (toks._toks._load, (kind, source, tier, source_sha256, encode_special_tokens)) */
static PyObject *Tok_reduce(Tok *self, PyObject *noargs)
{
    (void)noargs;
    return Py_BuildValue("O(iOIy#O)", mod_load_obj, (int)self->kind, self->source, (unsigned)self->tier_asked,
                         (const char *)self->info.source_sha256, (Py_ssize_t)32,
                         self->encode_special ? Py_True : Py_False);
}

static PyObject *Tok_get_encode_special(Tok *self, void *closure)
{
    (void)closure;
    return PyBool_FromLong(self->encode_special);
}

static int Tok_set_encode_special(Tok *self, PyObject *v, void *closure)
{
    (void)closure;
    if (v == NULL) {
        PyErr_SetString(PyExc_AttributeError, "cannot delete encode_special_tokens");
        return -1;
    }
    int b = PyObject_IsTrue(v);
    if (b < 0) { return -1; }
    self->encode_special = (uint8_t)b;
    return 0;
}

static PyObject *Tok_get_source(Tok *self, void *closure)
{
    (void)closure;
    return Py_NewRef(self->source);
}

/* ======================================================================================================
 * loading
 * ====================================================================================================== */

static int tier_of(PyObject *o, uint32_t *tier)
{
    *tier = TOKS_TIER_AUTO;
    if (o == NULL || o == Py_None) { return 0; }
    static const char *const names[] = { "auto", "scalar", "neon", "avx2", "avx512" };
    if (PyUnicode_Check(o)) {
        for (uint32_t i = 0; i < 5u; i++) {
            if (PyUnicode_CompareWithASCIIString(o, names[i]) == 0) { *tier = i; return 0; }
        }
    }
    PyErr_Format(PyExc_ValueError, "tier must be None, 'auto', 'scalar', 'neon', 'avx2' or 'avx512', not %R", o);
    return -1;
}

static PyObject *tok_new(PyTypeObject *cls, toks_ctx *ctx, int kind, PyObject *source, uint32_t tier)
{
    Tok *t = (Tok *)cls->tp_alloc(cls, 0);
    if (t == NULL) { toks_unload(ctx); return NULL; }
    t->ctx = ctx;
    t->info.size = (uint32_t)sizeof(toks_info);
    int64_t r = toks_get_info(ctx, &t->info);
    t->source = Py_NewRef(source);
    t->kind = (uint8_t)kind;
    t->tier_asked = tier;
    if (r < 0) {
        Py_DECREF(t);
        return fail(r, "toks_get_info");
    }
    uint32_t pre = 0;                           /* the template: sized, then written (ids, then their type ids) */
    int64_t n = toks_template(ctx, NULL, NULL, 0, &pre);
    if (n > 0) {
        t->tmpl = (uint32_t *)PyMem_Calloc(2u * (size_t)n, sizeof *t->tmpl);
        if (t->tmpl == NULL) { Py_DECREF(t); return PyErr_NoMemory(); }
        n = toks_template(ctx, t->tmpl, t->tmpl + n, (uint64_t)n, &pre);
    }
    if (n < 0) {
        Py_DECREF(t);
        return fail(n, "toks_template");
    }
    t->tmpl_n = (uint32_t)n;
    t->tmpl_pre = pre;
    return (PyObject *)t;
}

static PyObject *load_path(PyTypeObject *cls, PyObject *path, uint32_t tier)
{
    PyObject *fs = NULL;
    if (!PyUnicode_FSConverter(path, &fs)) { return NULL; }
    PyObject *src = PyOS_FSPath(path);
    if (src == NULL) { Py_DECREF(fs); return NULL; }
    toks_ctx *ctx = NULL;
    toks_diag d;
    memset(&d, 0, sizeof d);
    toks_load_opts o;
    memset(&o, 0, sizeof o);
    o.size = (uint32_t)sizeof o;
    o.tier = tier;
    o.diag = &d;
    int64_t r;
    const char *p = PyBytes_AS_STRING(fs);
    Py_BEGIN_ALLOW_THREADS
    r = toks_load(&ctx, p, &o);
    Py_END_ALLOW_THREADS
    PyObject *res = NULL;
    if (r < 0) {
        d.what[sizeof d.what - 1] = 0;
        fail(r, "%s (loading %R)", d.what[0] != 0 ? d.what : "load failed", src);
    } else {
        res = tok_new(cls, ctx, KIND_PATH, src, tier);
    }
    Py_DECREF(fs);
    Py_DECREF(src);
    return res;
}

static PyObject *load_mem(PyTypeObject *cls, const char *p, Py_ssize_t n, int kind, PyObject *keep, uint32_t tier)
{
    toks_ctx *ctx = NULL;
    toks_diag d;
    memset(&d, 0, sizeof d);
    toks_load_opts o;
    memset(&o, 0, sizeof o);
    o.size = (uint32_t)sizeof o;
    o.tier = tier;
    o.diag = &d;
    int64_t r;
    Py_BEGIN_ALLOW_THREADS
    r = toks_load_mem_copy(&ctx, p, (uint64_t)n, &o);
    Py_END_ALLOW_THREADS
    if (r < 0) {
        d.what[sizeof d.what - 1] = 0;
        return fail(r, "%s (loading %zd bytes)", d.what[0] != 0 ? d.what : "load failed", n);
    }
    return tok_new(cls, ctx, kind, keep, tier);
}

static PyObject *Tok_from_file(PyTypeObject *cls, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[3];
    uint32_t tier;
    if (parse(&A_FROM_FILE, args, nargs, kwnames, a) < 0 || tier_of(a[1], &tier) < 0) { return NULL; }
    long mib = a[2] != NULL && a[2] != Py_None ? PyLong_AsLong(a[2]) : 0;   /* TOKS_SCRATCH_CACHE_MIB */
    if (mib == -1 && PyErr_Occurred()) { return NULL; }
    if (mib != 0 && (mib < 4 || mib > 128 || (mib & (mib - 1)) != 0)) {
        PyErr_Format(PyExc_ValueError, "cache_mib must be 0 or a power of two 4..128, not %ld", mib);
        return NULL;
    }
    PyObject *t = load_path(cls, a[0], tier);
    if (t != NULL) { ((Tok *)t)->scr_flags = TOKS_SCRATCH_CACHE_MIB(mib); }
    return t;
}

static PyObject *Tok_from_str(PyTypeObject *cls, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[2];
    uint32_t tier;
    if (parse(&A_FROM_STR, args, nargs, kwnames, a) < 0 || tier_of(a[1], &tier) < 0) { return NULL; }
    if (!PyUnicode_Check(a[0])) {
        PyErr_Format(PyExc_TypeError, "from_str takes a str, not %.200s", Py_TYPE(a[0])->tp_name);
        return NULL;
    }
    Py_ssize_t n;
    const char *p = PyUnicode_AsUTF8AndSize(a[0], &n);
    if (p == NULL) { return NULL; }
    return load_mem(cls, p, n, KIND_STR, a[0], tier);
}

static PyObject *Tok_from_buffer(PyTypeObject *cls, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[2];
    uint32_t tier;
    if (parse(&A_FROM_BUF, args, nargs, kwnames, a) < 0 || tier_of(a[1], &tier) < 0) { return NULL; }
    PyObject *b = PyBytes_Check(a[0]) ? Py_NewRef(a[0]) : PyBytes_FromObject(a[0]);   /* immutable copy */
    if (b == NULL) { return NULL; }
    PyObject *res = load_mem(cls, PyBytes_AS_STRING(b), PyBytes_GET_SIZE(b), KIND_BYTES, b, tier);
    Py_DECREF(b);
    return res;
}

/* toks._toks._load(kind, source, tier, sha256, encode_special): unpickling */
static PyObject *mod_load(PyObject *mod, PyObject *const *args, Py_ssize_t nargs)
{
    (void)mod;
    if (nargs != 5 || !PyLong_Check(args[0]) || !PyLong_Check(args[2]) || !PyBytes_Check(args[3])) {
        PyErr_SetString(PyExc_TypeError, "_load(kind, source, tier, sha256, encode_special_tokens)");
        return NULL;
    }
    long kind = PyLong_AsLong(args[0]);
    unsigned long tier = PyLong_AsUnsignedLong(args[2]);
    if (PyErr_Occurred()) { return NULL; }
    if (tier > TOKS_TIER_AVX512) { tier = TOKS_TIER_AUTO; }
    PyObject *t;
    if (kind == KIND_PATH) {
        t = load_path(&TokenizerType, args[1], (uint32_t)tier);
    } else if (kind == KIND_STR && PyUnicode_Check(args[1])) {
        Py_ssize_t n;
        const char *p = PyUnicode_AsUTF8AndSize(args[1], &n);
        t = p != NULL ? load_mem(&TokenizerType, p, n, KIND_STR, args[1], (uint32_t)tier) : NULL;
    } else if (kind == KIND_BYTES && PyBytes_Check(args[1])) {
        t = load_mem(&TokenizerType, PyBytes_AS_STRING(args[1]), PyBytes_GET_SIZE(args[1]), KIND_BYTES, args[1],
                     (uint32_t)tier);
    } else {
        PyErr_SetString(PyExc_ValueError, "_load: unknown source kind");
        return NULL;
    }
    if (t == NULL) { return NULL; }
    Tok *k = (Tok *)t;
    if (PyBytes_GET_SIZE(args[3]) != 32 || memcmp(PyBytes_AS_STRING(args[3]), k->info.source_sha256, 32) != 0) {
        Py_DECREF(t);
        return fail(TOKS_E_FORMAT, "%R changed since the tokenizer was pickled (source sha256 differs)", args[1]);
    }
    int b = PyObject_IsTrue(args[4]);
    if (b < 0) { Py_DECREF(t); return NULL; }
    k->encode_special = (uint8_t)b;
    return t;
}

static void Tok_dealloc(Tok *self)
{
    if (self->ints != NULL) {
        for (uint32_t i = 0; i < self->info.n_ids; i++) { Py_XDECREF(self->ints[i]); }
        PyMem_Free(self->ints);
    }
    slot *s = self->pool;
    while (s != NULL) {
        slot *nx = s->next;
        slot_free(s);
        s = nx;
    }
    toks_unload(self->ctx);
    Py_XDECREF(self->source);
    Py_XDECREF(self->vocab);
    Py_XDECREF(self->tt);
    PyMem_Free(self->tt_bits);
    PyMem_Free(self->tmpl);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/* ======================================================================================================
 * DecodeStream
 * ====================================================================================================== */

typedef struct {
    PyObject_HEAD
    Tok         *tok;
    toks_stream  st;
    uint32_t     flags;
    uint8_t     *hold;      /* the stream's hold (toks_stream_hold), grown on TOKS_E_LIMIT; NULL: st's own 44 bytes */
    uint64_t     hold_cap;
} Stream;

static PyObject *Tok_decode_stream(Tok *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[1];
    if (parse(&A_STREAM, args, nargs, kwnames, a) < 0) { return NULL; }
    int skip = truth(a[0], 0);                  /* hf DecodeStream: skip_special_tokens=False */
    if (skip < 0) { return NULL; }
    Stream *s = PyObject_New(Stream, &StreamType);
    if (s == NULL) { return NULL; }
    s->tok = (Tok *)Py_NewRef(self);
    s->flags = skip ? TOKS_SKIP_SPECIAL : 0u;
    s->hold = NULL;
    s->hold_cap = 0u;
    toks_stream_init(self->ctx, &s->st, s->flags);
    return (PyObject *)s;
}

/* the str of a push / flush: out holds bound bytes; the stream is unchanged on any error. A byte-fallback run
 * longer than the hold (TOKS_E_LIMIT: nothing emitted, nothing changed) grows the hold to at least its size plus
 * n, twice it at least, and the same ids go again (toks.h toks_stream_hold): a stream never refuses a run */
static PyObject *stream_call(Stream *s, const uint32_t *ids, Py_ssize_t n, int flush)
{
    toks_ctx *ctx = s->tok->ctx;
    uint8_t stack[4096];
    uint8_t *out = stack;
    int64_t r;
    int grown = 0;
    for (;;) {                                  /* bound: 2 passes (one growth takes the push) */
        uint64_t bound = toks_stream_bound(ctx, flush ? 0u : (uint64_t)n) + 3u * s->hold_cap;
        if (bound < 64u) { bound = 64u; }
        if (bound > sizeof stack) {
            out = bound <= (uint64_t)PY_SSIZE_T_MAX ? (uint8_t *)PyMem_Malloc((size_t)bound) : NULL;
            if (out == NULL) { return PyErr_NoMemory(); }
        }
        r = flush ? toks_stream_flush(ctx, &s->st, out, bound)
                  : toks_stream_push(ctx, &s->st, ids, (uint64_t)n, out, bound);
        /* a too-long run only: n > TOKS_MAX_TEXT is TOKS_E_LIMIT too, and no hold takes that */
        if (r != TOKS_E_LIMIT || flush || grown || (uint64_t)n > TOKS_MAX_TEXT) { break; }
        grown = 1;
        if (out != stack) { PyMem_Free(out); out = stack; }
        uint64_t cur = s->hold_cap != 0u ? s->hold_cap : 44u, cap = 2u * cur + (uint64_t)n;
        uint8_t *h = cap <= (uint64_t)PY_SSIZE_T_MAX ? (uint8_t *)PyMem_Malloc((size_t)cap) : NULL;
        if (h == NULL) { return PyErr_NoMemory(); }
        int64_t m = toks_stream_hold(ctx, &s->st, h, cap);   /* copies from the old hold, then it is free */
        if (m < 0) { PyMem_Free(h); return fail(m, "DecodeStream.push"); }
        PyMem_Free(s->hold);
        s->hold = h;
        s->hold_cap = cap;
    }
    PyObject *res;
    if (r < 0) {
        res = flush ? fail(r, "DecodeStream.flush") : fail_ids(s->tok, r, "DecodeStream.push", ids, n);
    } else {
        res = PyUnicode_DecodeUTF8((const char *)out, (Py_ssize_t)r, "strict");
    }
    if (out != stack) { PyMem_Free(out); }
    return res;
}

/* push(ids) -> str: ids is an int or a sequence of ints */
static PyObject *Stream_push(Stream *self, PyObject *arg)
{
    uint32_t one;
    if (PyLong_Check(arg)) {
        if (id_of(arg, &one) < 0) { return NULL; }
        return stream_call(self, &one, 1, 0);
    }
    uint32_t stack[STACK_IDS];
    ids_in in;
    if (ids_get(arg, &in, stack, STACK_IDS) < 0) { ids_release(&in); return NULL; }
    PyObject *res = stream_call(self, in.p, in.n, 0);
    ids_release(&in);
    return res;
}

static PyObject *Stream_flush(Stream *self, PyObject *noargs)
{
    (void)noargs;
    return stream_call(self, NULL, 0, 1);
}

static PyObject *Stream_repr(Stream *self)
{
    return PyUnicode_FromFormat("<toks.DecodeStream of %s%s>", self->tok->info.name,
                                self->flags ? ", skip_special_tokens" : "");
}

static void Stream_dealloc(Stream *self)
{
    Py_XDECREF(self->tok);
    PyMem_Free(self->hold);
    PyObject_Free(self);
}

/* ======================================================================================================
 * types and module
 * ====================================================================================================== */

#define FASTKW(f) (PyCFunction)(void (*)(void))(f)

static PyMethodDef Tok_methods[] = {
    {"from_file", FASTKW(Tok_from_file), METH_FASTCALL | METH_KEYWORDS | METH_CLASS,
     "from_file(path, *, tier=None, cache_mib=0) -> Tokenizer\n\nLoad a tokenizer.json, a model directory holding one, or any "
     "other source toks_load accepts. tier forces a kernel tier ('scalar', 'neon', 'avx2', 'avx512'); None picks "
     "the fastest this machine runs (the TOKS_TIER environment variable can force one). cache_mib sizes the BPE "
     "piece caches of each scratch: 0 (2 MiB) or a power of two 4..128, faster on text seen before, same ids."},
    {"from_str", FASTKW(Tok_from_str), METH_FASTCALL | METH_KEYWORDS | METH_CLASS,
     "from_str(json, *, tier=None) -> Tokenizer\n\nLoad the text of a tokenizer.json."},
    {"from_buffer", FASTKW(Tok_from_buffer), METH_FASTCALL | METH_KEYWORDS | METH_CLASS,
     "from_buffer(data, *, tier=None) -> Tokenizer\n\nLoad the bytes of a tokenizer.json (precompiled .toks images are "
     "reserved: TOKS_E_UNSUPPORTED in 0.3)."},
    {"encode", FASTKW(Tok_encode), METH_FASTCALL | METH_KEYWORDS,
     "encode(text, *, add_special_tokens=True, added_tokens=None, continuation=False) -> list[int]\n"
     "encode(text, *, allowed_special=set(), disallowed_special='all') -> list[int]\n\n"
     "The ids of text (str, or bytes of any content, invalid UTF-8 included). With the defaults this is exactly hf's "
     "Tokenizer.encode(text).ids. added_tokens: 'all' (hf's default), 'nonspecial' (hf's "
     "encode_special_tokens=True) or 'none'; None follows this tokenizer's encode_special_tokens. "
     "add_special_tokens=False skips the post-processor (bos / eos / template).\n\n"
     "With allowed_special or disallowed_special it is tiktoken's Encoding.encode: no template, truncation or "
     "padding; allowed specials are recognized, a disallowed one in the text raises tiktoken's ValueError "
     "(special_tokens_set says which tokens are special)."},
    {"encode_ordinary", (PyCFunction)Tok_encode_ordinary, METH_O,
     "encode_ordinary(text) -> list[int]\n\ntiktoken's: no special token recognized (special_tokens_set), no template, "
     "truncation or padding."},
    {"encode_ex", FASTKW(Tok_encode_ex), METH_FASTCALL | METH_KEYWORDS,
     "encode_ex(text, *, add_special_tokens=True, added_tokens=None, continuation=False) -> Encoding\n\n"
     "hf's Tokenizer.encode(text): ids (== encode), type_ids, tokens, attention_mask, special_tokens_mask."},
    {"encode_into", FASTKW(Tok_encode_into), METH_FASTCALL | METH_KEYWORDS,
     "encode_into(text, out, *, add_special_tokens=True, added_tokens=None, continuation=False) -> int\n\n"
     "encode, with the ids written straight into out (a writable buffer of 4-byte integers: numpy uint32 / int32, "
     "array('I'), ...). Returns the total count n; out[:min(n, len(out))] holds the exact prefix."},
    {"encode_batch", FASTKW(Tok_encode_batch), METH_FASTCALL | METH_KEYWORDS,
     "encode_batch(texts, *, add_special_tokens=True, added_tokens=None, continuation=False) -> list[list[int]]\n\n"
     "encode of every text, run without the GIL, padded as hf's encode_batch pads (a BatchLongest file: to the "
     "longest member)."},
    {"encode_batch_ex", FASTKW(Tok_encode_batch_ex), METH_FASTCALL | METH_KEYWORDS,
     "encode_batch_ex(texts, *, add_special_tokens=True, added_tokens=None, continuation=False) -> list[Encoding]\n\n"
     "hf's Tokenizer.encode_batch(texts), run without the GIL."},
    {"pieces", FASTKW(Tok_pieces), METH_FASTCALL | METH_KEYWORDS,
     "pieces(text, *, added_tokens=None, continuation=False) -> list[int]\n\n"
     "The pieces the model sees, as end offsets into the normalized utf-8 bytes (an added token is one piece)."},
    {"decode", FASTKW(Tok_decode), METH_FASTCALL | METH_KEYWORDS,
     "decode(ids, skip_special_tokens=True) -> str\n\nhf's Tokenizer.decode. ids: a sequence of ints or a buffer of "
     "integers. An id beyond the table raises toks.Error (TOKS_E_ID)."},
    {"decode_batch", FASTKW(Tok_decode_batch), METH_FASTCALL | METH_KEYWORDS,
     "decode_batch(sequences, skip_special_tokens=True) -> list[str]"},
    {"decode_bytes", (PyCFunction)Tok_decode_bytes, METH_O,
     "decode_bytes(ids) -> bytes\n\ntiktoken's: the bytes the ids spell, special tokens included, never repaired to "
     "U+FFFD. An id beyond the table raises KeyError, as tiktoken's."},
    {"decode_bytes_batch", FASTKW(Tok_decode_bytes_batch), METH_FASTCALL | METH_KEYWORDS,
     "decode_bytes_batch(batch, *, num_threads=8) -> list[bytes]"},
    {"decode_stream", FASTKW(Tok_decode_stream), METH_FASTCALL | METH_KEYWORDS,
     "decode_stream(skip_special_tokens=False) -> DecodeStream"},
    {"token_bytes", (PyCFunction)Tok_token_bytes, METH_O,
     "token_bytes(id) -> bytes | None\n\nThe bytes id decodes to on its own (None: the id has no string)."},
    {"token_byte_values", (PyCFunction)Tok_token_byte_values, METH_NOARGS,
     "token_byte_values() -> list[bytes]\n\ntiktoken's: the bytes of every token outside the added / special ones, "
     "sorted."},
    {"token_to_id", (PyCFunction)Tok_token_to_id, METH_O,
     "token_to_id(token) -> int | None\n\nA str: hf's Tokenizer.token_to_id, written forms ('\xc4\xa0hello'; read from the "
     "tokenizer.json on first use). Bytes: the id whose decoded bytes they are (b' hello'), from the library's index; an "
     "added token's content is found either way. None when absent."},
    {"encode_bound", (PyCFunction)Tok_encode_bound, METH_O,
     "encode_bound(n) -> int\n\nThe most ids encode can return for any text of n bytes (bytes of any content, or a str's "
     "UTF-8), under any arguments: an encode_into buffer of this many ids is never short (toks_encode_bound). Every n >= 0 "
     "has one, saturating at 2**64 - 1; encode refuses texts over MAX_TEXT bytes."},
    {"id_flags", (PyCFunction)Tok_id_flags, METH_O,
     "id_flags(id) -> int\n\nID_ADDED (hf's added_tokens_decoder holds the id), ID_SPECIAL (its AddedToken is special), "
     "ID_BYTE (a <0xHH> byte-fallback token)."},
    {"id_to_token", (PyCFunction)Tok_id_to_token, METH_O,
     "id_to_token(id) -> str | None\n\nhf's Tokenizer.id_to_token (read from the tokenizer.json on first use)."},
    {"num_special_tokens_to_add", FASTKW(Tok_num_special_tokens_to_add), METH_FASTCALL | METH_KEYWORDS,
     "num_special_tokens_to_add(is_pair=False) -> int\n\nhf's: the ids the post-processor adds around one text "
     "(toks_template); a pair raises toks.Error (TOKS_E_UNSUPPORTED)."},
    {"get_added_tokens_decoder", (PyCFunction)Tok_get_added_tokens_decoder, METH_NOARGS,
     "get_added_tokens_decoder() -> dict[int, AddedToken]\n\nhf's, in id order (toks_added)."},
    {"get_vocab", FASTKW(Tok_get_vocab), METH_FASTCALL | METH_KEYWORDS,
     "get_vocab(with_added_tokens=True) -> dict[str, int]"},
    {"get_vocab_size", FASTKW(Tok_get_vocab_size), METH_FASTCALL | METH_KEYWORDS,
     "get_vocab_size(with_added_tokens=True) -> int"},
    {"info", (PyCFunction)Tok_info, METH_NOARGS, "info() -> dict: toks_get_info"},
    {"_vocab", (PyCFunction)Tok_vocab, METH_NOARGS, NULL},
    {"__reduce__", (PyCFunction)Tok_reduce, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL},
};

static PyGetSetDef Tok_getset[] = {
    {"encode_special_tokens", (getter)Tok_get_encode_special, (setter)Tok_set_encode_special,
     "hf's encode_special_tokens: when True, encode's default mode is added_tokens='nonspecial'", NULL},
    {"source", (getter)Tok_get_source, NULL, "what was loaded: the path, the bytes or the str", NULL},
    {"n_vocab", (getter)Tok_get_n_vocab, NULL, "tiktoken's: max_token_value + 1", NULL},
    {"max_token_value", (getter)Tok_get_max_token_value, NULL, "tiktoken's: the highest id", NULL},
    {"eot_token", (getter)Tok_get_eot_token, NULL,
     "tiktoken's: the id of the special token <|endoftext|> (KeyError when the file has none)", NULL},
    {"special_tokens_set", (getter)Tok_get_special_tokens_set, NULL,
     "tiktoken's: the special tokens, the file's own (a tokenizer.json's added tokens marked special, a tiktoken "
     "model's every special token)", NULL},
    {"name", (getter)Tok_get_name, NULL, "the loaded file's name (info()['name'])", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyTypeObject TokenizerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "toks.Tokenizer",
    .tp_basicsize = sizeof(Tok),
    .tp_dealloc = (destructor)Tok_dealloc,
    .tp_repr = (reprfunc)Tok_repr,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "A loaded tokenizer (toks_ctx): read-only, shared by any number of threads.\n\n"
              "Make one with Tokenizer.from_file / from_str / from_buffer.",
    .tp_methods = Tok_methods,
    .tp_getset = Tok_getset,
};

static PyMethodDef Stream_methods[] = {
    {"push", (PyCFunction)Stream_push, METH_O,
     "push(ids) -> str\n\nThe text these ids complete: every byte of the batch decode that no later id can "
     "change. ids: an int or a sequence of ints."},
    {"flush", (PyCFunction)Stream_flush, METH_NOARGS,
     "flush() -> str\n\nThe rest (U+FFFD for a held incomplete sequence); the stream starts over."},
    {NULL, NULL, 0, NULL},
};

static PyTypeObject StreamType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "toks.DecodeStream",
    .tp_basicsize = sizeof(Stream),
    .tp_dealloc = (destructor)Stream_dealloc,
    .tp_repr = (reprfunc)Stream_repr,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Incremental decode of one id stream (toks_stream_*): Tokenizer.decode_stream(). One thread at a "
              "time.",
    .tp_methods = Stream_methods,
};

static PyGetSetDef Enc_getset[] = {
    {"ids", (getter)Enc_ids, NULL, "the ids (== Tokenizer.encode with the same arguments)", NULL},
    {"type_ids", (getter)Enc_type_ids, NULL, "hf's type ids: the template's, the text's (its $A type), pad_type_id", NULL},
    {"tokens", (getter)Enc_tokens, NULL, "hf's token strings (read from the file on first access)", NULL},
    {"attention_mask", (getter)Enc_attention, NULL, "1, but 0 on padding", NULL},
    {"special_tokens_mask", (getter)Enc_special, NULL, "1 on the template's ids and padding, else 0", NULL},
    {"overflowing", (getter)Enc_overflowing, NULL, "[] (the overflow of a truncated text is not computed)", NULL},
    {"n_sequences", (getter)Enc_n_sequences, NULL, "1", NULL},
    {"_layout", (getter)Enc_layout, NULL, NULL, NULL},
    {"_text", (getter)Enc_text, NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PySequenceMethods Enc_as_sequence = { .sq_length = (lenfunc)Enc_len };

static PyTypeObject EncodingType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "toks.Encoding",
    .tp_basicsize = sizeof(Enc),
    .tp_dealloc = (destructor)Enc_dealloc,
    .tp_repr = (reprfunc)Enc_repr,
    .tp_as_sequence = &Enc_as_sequence,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "hf's Encoding of one text: Tokenizer.encode_ex / encode_batch_ex. Every attribute is a new list.",
    .tp_getset = Enc_getset,
};

static PyMethodDef mod_methods[] = {
    {"_load", (PyCFunction)(void (*)(void))mod_load, METH_FASTCALL, NULL},
    {"_error", (PyCFunction)(void (*)(void))mod_error, METH_FASTCALL, NULL},
    {NULL, NULL, 0, NULL},
};

/* PyModule_Add (3.13+) for every version: steals o */
static int add_obj(PyObject *m, const char *name, PyObject *o)
{
    if (o == NULL) { return -1; }
    int r = PyModule_AddObjectRef(m, name, o);
    Py_DECREF(o);
    return r;
}

static struct PyModuleDef toks_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "toks._toks",
    .m_doc = "toks: the CPython binding of include/toks.h",
    .m_size = -1,
    .m_methods = mod_methods,
};

PyMODINIT_FUNC PyInit__toks(void)
{
    if (intern_names(&A_ENCODE, "text", "add_special_tokens", "added_tokens", "continuation", "allowed_special",
                     "disallowed_special") < 0 ||
        intern_names(&A_ENCODE_EX, "text", "add_special_tokens", "added_tokens", "continuation") < 0 ||
        intern_names(&A_ENCODE_INTO, "text", "out", "add_special_tokens", "added_tokens", "continuation") < 0 ||
        intern_names(&A_BATCH, "texts", "add_special_tokens", "added_tokens", "continuation") < 0 ||
        intern_names(&A_BATCH_EX, "texts", "add_special_tokens", "added_tokens", "continuation") < 0 ||
        intern_names(&A_PIECES, "text", "added_tokens", "continuation") < 0 ||
        intern_names(&A_DECODE, "ids", "skip_special_tokens") < 0 ||
        intern_names(&A_DECODE_B, "sequences", "skip_special_tokens") < 0 ||
        intern_names(&A_BYTES_B, "batch", "num_threads") < 0 ||
        intern_names(&A_STREAM, "skip_special_tokens") < 0 ||
        intern_names(&A_FROM_FILE, "path", "tier", "cache_mib") < 0 ||
        intern_names(&A_FROM_STR, "json", "tier") < 0 ||
        intern_names(&A_FROM_BUF, "data", "tier") < 0 ||
        intern_names(&A_VOCAB, "with_added_tokens") < 0 ||
        intern_names(&A_VOCAB_SIZE, "with_added_tokens") < 0 ||
        intern_names(&A_NSPECIAL, "is_pair") < 0) {
        return NULL;
    }
    s_all = PyUnicode_InternFromString("all");
    s_nonspecial = PyUnicode_InternFromString("nonspecial");
    s_none = PyUnicode_InternFromString("none");
    if (s_all == NULL || s_nonspecial == NULL || s_none == NULL) { return NULL; }
    if (PyType_Ready(&TokenizerType) < 0 || PyType_Ready(&StreamType) < 0 || PyType_Ready(&EncodingType) < 0) {
        return NULL;
    }
    PyObject *m = PyModule_Create(&toks_module);
    if (m == NULL) { return NULL; }
    Error = PyErr_NewExceptionWithDoc("toks.Error",
                                      "A toks error: .code is the TOKS_E_* value (include/toks.h), .name its name.",
                                      NULL, NULL);
    mod_load_obj = PyObject_GetAttrString(m, "_load");
    if (Error == NULL || mod_load_obj == NULL ||
        PyModule_AddObjectRef(m, "Error", Error) < 0 ||
        PyModule_AddObjectRef(m, "Tokenizer", (PyObject *)&TokenizerType) < 0 ||
        PyModule_AddObjectRef(m, "DecodeStream", (PyObject *)&StreamType) < 0 ||
        PyModule_AddObjectRef(m, "Encoding", (PyObject *)&EncodingType) < 0 ||
        add_obj(m, "ABI", Py_BuildValue("(II)", (unsigned)TOKS_ABI_MAJOR, (unsigned)TOKS_ABI_MINOR)) < 0 ||
        add_obj(m, "__version__", PyUnicode_FromString(toks_version())) < 0 ||
        PyModule_AddIntConstant(m, "MAX_TEXT", (long)TOKS_MAX_TEXT) < 0 ||
        PyModule_AddIntConstant(m, "ID_ADDED", (long)TOKS_ID_ADDED) < 0 ||
        PyModule_AddIntConstant(m, "ID_SPECIAL", (long)TOKS_ID_SPECIAL) < 0 ||
        PyModule_AddIntConstant(m, "ID_BYTE", (long)TOKS_ID_BYTE) < 0) {
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
