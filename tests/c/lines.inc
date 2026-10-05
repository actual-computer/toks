/* tests/c/lines.inc: the guard of the tests' line-oriented data readers. Every such file under tests/ is LF-only text
 * that ends in a newline (.gitattributes). A checkout that rewrote it (Git for Windows' core.autocrlf: CRLF), a
 * control byte, a line longer than the reader's buffer or a number that is not there fails at once, naming the file,
 * the line and the byte, and the reader stops: it never loops on what it cannot consume (test_tiktoken once spun 37
 * minutes on a CR that its strtoul loop could not pass). Readers open their files in binary mode ("rb"): the msvc
 * ucrt's text mode would turn CRLF into LF and stop at a 0x1A, hiding exactly what this guard is for. Each function
 * prints its FAIL line and returns 0; the test counts the failure in its own counter. */
#ifndef TOKS_TEST_LINES_INC
#define TOKS_TEST_LINES_INC

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const char *path; unsigned long line; } td_at;   /* a file and its 1-based line */

static inline int td_bad(const td_at *at, const char *p, const char *what)
{
    unsigned c = (unsigned char)*p;
    printf("FAIL %s:%lu: %s, at byte 0x%02x%s\n", at->path, at->line, what, c,
           c == '\r' ? " (a CR: CRLF line ends, e.g. a core.autocrlf checkout; test data is LF only)" : "");
    return 0;
}

/* b[0, n): no control byte but tab and newline, and a newline at the end */
static inline int td_text(const char *path, const uint8_t *b, uint64_t n)
{
    td_at at = { path, 1 };
    for (uint64_t i = 0; i < n; i++) {                               /* bound: n */
        if (b[i] == '\n') { at.line++; continue; }
        if (b[i] < 0x20u && b[i] != '\t') { return td_bad(&at, (const char *)b + i, "a control byte"); }
    }
    if (n != 0 && b[n - 1] != '\n') { at.line--; return td_bad(&at, (const char *)b + n - 1, "no newline at the end"); }
    return 1;
}

/* one line as fgets left it in a buffer of cap bytes: the same rule, and the whole line fit */
static inline int td_line(const td_at *at, const char *s, size_t cap)
{
    size_t n = strlen(s);
    for (size_t i = 0; i < n; i++) {                                 /* bound: the line */
        if (s[i] != '\n' && (unsigned char)s[i] < 0x20u && s[i] != '\t') { return td_bad(at, s + i, "a control byte"); }
    }
    if (n == 0 || s[n - 1] != '\n') {
        return td_bad(at, n != 0 ? s + n - 1 : s, n + 1 >= cap ? "a line longer than the reader's buffer" : "no newline at the end");
    }
    return 1;
}

/* the unsigned decimal number at *q into *v and *q past it; none there is a failure, never a stall */
static inline int td_num(const td_at *at, char **q, unsigned long *v)
{
    char *e;
    *v = strtoul(*q, &e, 10);
    if (e == *q) { return td_bad(at, *q, "a number expected"); }
    *q = e;
    return 1;
}

#endif
