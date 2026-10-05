/* file.c: the os surface of loading (posix + win32): whole-file reads (docs/notes/c-core.md §file.c.1) */
#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L
#endif
#include "core.h"

#include <stdlib.h>

/* <dir>/tokenizer.json into buf (a separator is added unless dir ends with one). */
static int64_t dir_join(const char *dir, char *buf, uint64_t cap, char sep)
{
    static const char leaf[] = "tokenizer.json";
    uint64_t dl = 0u;
    while (dir[dl] != 0 && dl < cap) { dl++; }           /* bound: cap */
    if (dl == 0u) { return TOKS_E_OPEN; }
    uint64_t add = (dir[dl - 1u] == '/' || dir[dl - 1u] == '\\') ? 0u : 1u;
    if (dl + add + sizeof leaf > cap) { return TOKS_E_OPEN; }
    memcpy(buf, dir, (size_t)dl);
    if (add != 0u) { buf[dl] = sep; }
    memcpy(buf + dl + add, leaf, sizeof leaf);
    return 0;
}

#if defined(_WIN32)

#include <windows.h>

int64_t toks_plat_getenv(const char *name, char *buf, uint64_t cap)
{
    if (cap == 0u || cap > 0x7FFFFFFFu) { return -2; }
    DWORD n = GetEnvironmentVariableA(name, buf, (DWORD)cap);   /* the process block, not a crt copy */
    if (n == 0u) { buf[0] = 0; return -1; }                     /* unset or empty */
    if ((uint64_t)n >= cap) { buf[0] = 0; return -2; }          /* n = the size it needs */
    return (int64_t)n;
}

int64_t toks_plat_read_file(const char *path, uint8_t **out, uint64_t *len, int *is_dir)
{
    *out = NULL; *len = 0; *is_dir = 0;
    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) {
        DWORD attr = GetFileAttributesA(path);
        if (attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY)) { *is_dir = 1; }
        return TOKS_E_OPEN;
    }
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(h, &sz) || sz.QuadPart <= 0) { CloseHandle(h); return TOKS_E_OPEN; }
    if ((uint64_t)sz.QuadPart > TOKS_MAX_SOURCE_BYTES) { CloseHandle(h); return TOKS_E_LIMIT; }
    uint64_t n = (uint64_t)sz.QuadPart;
    uint8_t *buf = (uint8_t *)toks_plat_alloc(n);
    if (buf == NULL) { CloseHandle(h); return TOKS_E_NOMEM; }
    uint64_t off = 0;
    while (off < n) {                                    /* bound: n (off advances per read) */
        DWORD got = 0;
        DWORD want = (n - off > 0x40000000u) ? 0x40000000u : (DWORD)(n - off);
        if (!ReadFile(h, buf + off, want, &got, NULL) || got == 0) {
            CloseHandle(h); toks_plat_free(buf, n); return TOKS_E_OPEN;
        }
        off += got;
    }
    CloseHandle(h);
    *out = buf; *len = n;
    return 0;
}

int64_t toks_plat_dir_lookup(const char *dir, char *buf, uint64_t cap)
{
    if (dir_join(dir, buf, cap, '\\') != 0) { return TOKS_E_OPEN; }
    DWORD attr = GetFileAttributesA(buf);
    if (attr == INVALID_FILE_ATTRIBUTES || (attr & FILE_ATTRIBUTE_DIRECTORY)) { return TOKS_E_OPEN; }
    return 0;
}

#else /* posix */

#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>

/* rationale: docs/notes/c-core.md §file.c.2 */
static int path_is_dir(const char *path)
{
    int fd = open(path, O_RDONLY | O_DIRECTORY);
    if (fd < 0) { return 0; }
    close(fd);
    return 1;
}

/* the byte size of an open regular file (seekable, at least one byte), else 0; leaves the offset at 0 */
static uint64_t file_size(int fd)
{
    off_t end = lseek(fd, 0, SEEK_END);
    if (end <= 0 || lseek(fd, 0, SEEK_SET) != 0) { return 0u; }
    return (uint64_t)end;
}

int64_t toks_plat_getenv(const char *name, char *buf, uint64_t cap)
{
    if (cap == 0u) { return -2; }
    buf[0] = 0;
    const char *v = getenv(name);
    if (v == NULL || v[0] == 0) { return -1; }
    uint64_t n = 0u;
    while (v[n] != 0) {                                  /* bound: cap (longer is -2) */
        if (n + 1u >= cap) { buf[0] = 0; return -2; }
        buf[n] = v[n];
        n++;
    }
    buf[n] = 0;
    return (int64_t)n;
}

int64_t toks_plat_read_file(const char *path, uint8_t **out, uint64_t *len, int *is_dir)
{
    *out = NULL; *len = 0; *is_dir = 0;
    if (path_is_dir(path)) { *is_dir = 1; return TOKS_E_OPEN; }
    int fd = open(path, O_RDONLY | O_NONBLOCK);
    if (fd < 0) { return TOKS_E_OPEN; }
    uint64_t n = file_size(fd);
    if (n == 0u) { close(fd); return TOKS_E_OPEN; }
    if (n > TOKS_MAX_SOURCE_BYTES) { close(fd); return TOKS_E_LIMIT; }
    uint8_t *buf = (uint8_t *)toks_plat_alloc(n);
    if (buf == NULL) { close(fd); return TOKS_E_NOMEM; }
    uint64_t off = 0;
    while (off < n) {                                    /* bound: n (off advances per read) */
        ssize_t r = read(fd, buf + off, (size_t)(n - off));
        if (r <= 0) { close(fd); toks_plat_free(buf, n); return TOKS_E_OPEN; }
        off += (uint64_t)r;
    }
    close(fd);
    *out = buf; *len = n;
    return 0;
}

int64_t toks_plat_dir_lookup(const char *dir, char *buf, uint64_t cap)
{
    if (dir_join(dir, buf, cap, '/') != 0 || path_is_dir(buf)) { return TOKS_E_OPEN; }
    int fd = open(buf, O_RDONLY | O_NONBLOCK);
    if (fd < 0) { return TOKS_E_OPEN; }
    uint64_t n = file_size(fd);
    close(fd);
    return n != 0u ? 0 : TOKS_E_OPEN;
}

#endif
