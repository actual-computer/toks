//! tools/bench/gigatoken/main.rs: gigatoken (the bar to beat, SPEC §0.6) on exactly the chunks
//! tools/bench/e2e.c measures, one thread, through its Rust API (no Python binding in the timer).
//!
//!   gigatoken-bench <tokenizer.json | kimi dir> <chunk_bytes|0> <reps> <ids_out> <file>...
//!
//! source: the pinned commit copied read-only into build/third-party/gigatoken (tools/bench/gigatoken.sh).
//! chunks: e2e.c's rule (the files concatenated; cut after the first '\n' at or past each chunk boundary;
//!   0 = the whole text in one call).
//! load: a tokenizer.json through load_hf_slice (byte-level bpe, or sentencepiece-style with byte_fallback);
//!   a directory holding tiktoken.model (Kimi) through load_tiktoken with gigatoken's Kimi pretokenizer and no
//!   special tokens and none of the wrapper's cuts (the corpora hold no specials; the ids are checked either way:
//!   texts over 400,000 chars or with a 25,000-char same-class run differ from the reference by construction).
//!   A file gigatoken refuses prints `GIGA na=...` and exits 0: the cell's gigatoken column is n/a.
//! states, e2e.c's: cold = a fresh state (fork / new EncodeState) before EVERY call, outside the timer, each
//!   call timed alone, the cold reps first and back to back (GIGA_COLD_REPS of them: default 2, 0 = none; a fresh
//!   vocabulary-seeded state costs 5-60 ms per call, outside the timer); then each rep: pass = with OTHER text
//!   (GIGA_WARM_ON="<file> ...", cut at WARM_CHUNK = 4096 bytes whatever the cell's chunk, as e2e.c) a state
//!   encodes it untimed, then a fresh state the timed pass
//!   (pass_after=other: the cpu caches hold the other text's lines, not this text's), without it a fresh state's
//!   pass (pass_after=same); warm = the pass right after it; lang = a fresh state, one untimed pass over the other
//!   text, then the timed pass (lang-x: e2e.sh's other text is the other corpora); warmo = a fresh state, an
//!   untimed pass over this text, one over the other text, then the timed pass over this text (the serving
//!   replay: e2e.c's); coldo = each cold rep after an untimed pass over the other text (a fresh state
//!   that is then dropped), GIGA_COLD_REPS of them like cold's.
//! cache budget: gigatoken's default (512 MiB per state for byte-level bpe, its documented per-worker budget;
//!   sentencepiece's states take the model's budget plus a fixed front cache), or GIGA_CACHE_MIB=n MiB
//!   (set_max_cache_bytes before any fork / EncodeState::with_budget: gigatoken floors a budget below its
//!   vocabulary seed, so a small n is the seed plus headroom, not n MiB).
//! keys: <state>_s = the best rep, <state>_reps_s = every rep in order; cache_mib = the configured budget (0 =
//!   unbounded); entries_<state> = the cached pretokens / units after that state's pass (last rep).
//! exact: one untimed pass from a fresh state writes the id stream (u32 little-endian) to ids_out;
//!   tools/bench/e2e_ref.py compares it with hf's ids without the post-processor (gigatoken applies none).
use std::{env, fs, hint::black_box, path::Path, process::exit, time::Instant};

use gigatoken_rs::EncodeState;
use gigatoken_rs::load_tokenizer::hf::{HfTokenizer, load_hf_slice};
use gigatoken_rs::load_tokenizer::tiktoken::load_tiktoken;
use gigatoken_rs::pretokenize::PretokenizerType;

const WARM_CHUNK: usize = 4096; // the OTHER text's chunk, whatever the cell's (e2e.c's WARM_CHUNK)
const REV: &str = match option_env!("GIGA_REV") {
    Some(r) => r,
    None => "unknown",
};

fn chunks(buf: &[u8], chunk: usize) -> Vec<&[u8]> {
    let (mut out, mut p, n) = (Vec::new(), 0usize, buf.len());
    while p < n {
        let mut e = n;
        if chunk > 0 && p + chunk < n {
            e = p + chunk;
            while e < n && buf[e - 1] != b'\n' {
                e += 1;
            }
        }
        out.push(&buf[p..e]);
        p = e;
    }
    out
}

fn read_all<'a>(files: impl Iterator<Item = &'a str>) -> Vec<u8> {
    let mut buf = Vec::new();
    for f in files {
        buf.extend_from_slice(&fs::read(f).unwrap_or_else(|e| panic!("{f}: {e}")));
    }
    buf
}

fn na(why: String) -> ! {
    println!("GIGA tool=gigatoken version={REV} na={}", why.replace(char::is_whitespace, "_"));
    exit(0);
}

/// per state (cold, pass, warm, lang): every rep's seconds, and the cache entries after its pass
#[derive(Default)]
struct Tm {
    s: [Vec<f64>; 6],
    entries: [usize; 6],
}

const STATES: [&str; 6] = ["cold", "pass", "warm", "lang", "warmo", "coldo"];

fn main() {
    let a: Vec<String> = env::args().collect();
    if a.len() < 6 {
        eprintln!("usage: gigatoken-bench <tokenizer.json | kimi dir> <chunk_bytes|0> <reps> <ids_out> <file>...");
        exit(2);
    }
    let chunk: usize = a[2].parse().expect("chunk");
    let reps: usize = a[3].parse().expect("reps");
    let cold_reps: usize = env::var("GIGA_COLD_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(2).min(reps);
    let cache_mib: Option<usize> = env::var("GIGA_CACHE_MIB").ok().and_then(|v| v.parse().ok());
    let buf = read_all(a[5..].iter().map(String::as_str));
    let texts = chunks(&buf, chunk);
    let wbuf = read_all(env::var("GIGA_WARM_ON").unwrap_or_default().split_whitespace());
    let wtexts = chunks(&wbuf, WARM_CHUNK);
    let t0 = Instant::now();
    let path = Path::new(&a[1]);
    let mut model = if path.is_dir() {
        match load_tiktoken(path.join("tiktoken.model"), PretokenizerType::Kimi, vec![]) {
            Ok(t) => HfTokenizer::Bpe(t),
            Err(e) => na(format!("load:{e}")),
        }
    } else {
        let data = fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        match load_hf_slice(&data) {
            Ok(m) => m,
            Err(e) => na(format!("load:{e}")),
        }
    };
    let load_ms = t0.elapsed().as_secs_f64() * 1e3;
    let mut ids: Vec<u32> = Vec::with_capacity(buf.len() + 64);
    let mut tm = Tm::default();
    let (kind, total, setup_ms, budget) = match &mut model {
        HfTokenizer::Bpe(base) => {
            let kind = if path.is_dir() { "tiktoken" } else { "bpe" };
            if let Some(m) = cache_mib {
                base.set_max_cache_bytes(Some(m << 20)); // forks inherit it (each its full budget)
            }
            let base = &*base;
            // exactness pass, fresh state
            let mut all: Vec<u32> = Vec::with_capacity(buf.len() + 64);
            let mut t = base.fork();
            for text in &texts {
                t.encode_with_added_tokens_flat(text, &mut all);
            }
            let f0 = Instant::now();
            let fresh = base.fork();
            let setup_ms = f0.elapsed().as_secs_f64() * 1e3;
            drop(fresh);
            write_ids(&a[4], &all);
            let total = all.len();
            for _ in 0..cold_reps {                    // the cold reps first, back to back (e2e.c's order)
                let (mut c, mut n) = (0.0, 0);
                for text in &texts {
                    let mut t = base.fork();
                    ids.clear();
                    let s = Instant::now();
                    t.encode_with_added_tokens_flat(text, &mut ids);
                    n += black_box(&ids).len();
                    c += s.elapsed().as_secs_f64();
                }
                assert_eq!(n, total, "cold pass id count");
                tm.s[0].push(c);
            }
            let other = |t: &mut gigatoken_rs::Tokenizer| {   // an untimed pass over the other text
                let mut w = Vec::new();
                for text in &wtexts {
                    w.clear();
                    t.encode_with_added_tokens_flat(text, &mut w);
                }
            };
            let coldo_reps = if wtexts.is_empty() { 0 } else { cold_reps };
            for _ in 0..coldo_reps {                   // coldo: each cold rep after the other text
                other(&mut base.fork());
                let (mut c, mut n) = (0.0, 0);
                for text in &texts {
                    let mut t = base.fork();
                    ids.clear();
                    let s = Instant::now();
                    t.encode_with_added_tokens_flat(text, &mut ids);
                    n += black_box(&ids).len();
                    c += s.elapsed().as_secs_f64();
                }
                assert_eq!(n, total, "coldo pass id count");
                tm.s[5].push(c);
            }
            for _ in 0..reps {
                let mut timed = |t: &mut gigatoken_rs::Tokenizer, st: usize| {
                    let s = Instant::now();
                    let mut n = 0;
                    for text in &texts {
                        ids.clear();
                        t.encode_with_added_tokens_flat(text, &mut ids);
                        n += black_box(&ids).len();
                    }
                    tm.s[st].push(s.elapsed().as_secs_f64());
                    assert_eq!(n, total, "{} pass id count", STATES[st]);
                    tm.entries[st] = t.cache_entries();
                };
                if !wtexts.is_empty() {                // pass: after the other text, on a fresh state
                    other(&mut base.fork());
                }
                let mut t = base.fork();
                timed(&mut t, 1);
                timed(&mut t, 2);
                if !wtexts.is_empty() {                // lang: the state the other text left
                    let mut t = base.fork();
                    other(&mut t);
                    timed(&mut t, 3);
                    let mut t = base.fork();           // warmo: this text, the other text, this text again
                    let mut w = Vec::new();
                    for text in &texts {
                        w.clear();
                        t.encode_with_added_tokens_flat(text, &mut w);
                    }
                    other(&mut t);
                    timed(&mut t, 4);
                }
            }
            (kind, total, setup_ms, base.max_cache_bytes())
        }
        HfTokenizer::SentencePiece(spm) => {
            let mut strs = Vec::with_capacity(texts.len());
            for text in &texts {
                match std::str::from_utf8(text) {
                    Ok(s) => strs.push(s),
                    Err(_) => na("input_not_utf8".to_string()),
                }
            }
            let wstrs: Vec<&str> = wtexts.iter().map(|t| std::str::from_utf8(t).unwrap_or("")).collect();
            let budget = cache_mib.map(|m| m << 20).or(spm.max_cache_bytes());
            let spm = &*spm;
            let mut tmp = Vec::new();
            let mut all: Vec<u32> = Vec::with_capacity(buf.len() + 64);
            let mut st = EncodeState::with_budget(budget);
            for s in &strs {
                tmp.clear();
                spm.encode_raw_with(&mut st, s, &mut tmp);
                all.extend(tmp.iter().map(|t| t.0));
            }
            write_ids(&a[4], &all);
            let total = all.len();
            let f0 = Instant::now();
            let fresh = EncodeState::with_budget(budget);
            let setup_ms = f0.elapsed().as_secs_f64() * 1e3;
            drop(fresh);
            for _ in 0..cold_reps {                    // the cold reps first, back to back (e2e.c's order)
                let (mut c, mut n) = (0.0, 0);
                for s in &strs {
                    let mut st = EncodeState::with_budget(budget);
                    tmp.clear();
                    let t = Instant::now();
                    spm.encode_raw_with(&mut st, s, &mut tmp);
                    n += black_box(&tmp).len();
                    c += t.elapsed().as_secs_f64();
                }
                assert_eq!(n, total, "cold pass id count");
                tm.s[0].push(c);
            }
            let other = |es: &mut EncodeState| {      // an untimed pass over the other text
                let mut w = Vec::new();
                for s in &wstrs {
                    w.clear();
                    spm.encode_raw_with(es, s, &mut w);
                }
            };
            let coldo_reps = if wstrs.is_empty() { 0 } else { cold_reps };
            for _ in 0..coldo_reps {                   // coldo: each cold rep after the other text
                other(&mut EncodeState::with_budget(budget));
                let (mut c, mut n) = (0.0, 0);
                for s in &strs {
                    let mut st = EncodeState::with_budget(budget);
                    tmp.clear();
                    let t = Instant::now();
                    spm.encode_raw_with(&mut st, s, &mut tmp);
                    n += black_box(&tmp).len();
                    c += t.elapsed().as_secs_f64();
                }
                assert_eq!(n, total, "coldo pass id count");
                tm.s[5].push(c);
            }
            for _ in 0..reps {
                let mut timed = |es: &mut EncodeState, k: usize| {
                    let t = Instant::now();
                    let mut n = 0;
                    for s in &strs {
                        tmp.clear();
                        spm.encode_raw_with(es, s, &mut tmp);
                        n += black_box(&tmp).len();
                    }
                    tm.s[k].push(t.elapsed().as_secs_f64());
                    assert_eq!(n, total, "{} pass id count", STATES[k]);
                    tm.entries[k] = es.cache_size();
                };
                if !wstrs.is_empty() {                 // pass: after the other text, on a fresh state
                    other(&mut EncodeState::with_budget(budget));
                }
                let mut es = EncodeState::with_budget(budget);
                timed(&mut es, 1);
                timed(&mut es, 2);
                if !wstrs.is_empty() {                 // lang: the state the other text left
                    let mut es = EncodeState::with_budget(budget);
                    other(&mut es);
                    timed(&mut es, 3);
                    let mut es = EncodeState::with_budget(budget);   // warmo: this text, the other text, this again
                    let mut w = Vec::new();
                    for s in &strs {
                        w.clear();
                        spm.encode_raw_with(&mut es, s, &mut w);
                    }
                    other(&mut es);
                    timed(&mut es, 4);
                }
            }
            ("spm", total, setup_ms, budget)
        }
    };
    let best = |st: usize| tm.s[st].iter().cloned().fold(f64::INFINITY, f64::min);
    let (cold, pass, warm) = (best(0), best(1), best(2));
    let lang = if tm.s[3].is_empty() { 0.0 } else { best(3) };
    let warmo = if tm.s[4].is_empty() { 0.0 } else { best(4) };
    let coldo = if tm.s[5].is_empty() { 0.0 } else { best(5) };
    let n = buf.len() as f64;
    println!(
        "giga  chunk {:7}  cold {:7.1} MB/s | pass {:7.1} MB/s | warm {:7.1} MB/s | {kind}",
        chunk,
        n / cold / 1e6,
        n / pass / 1e6,
        n / warm / 1e6
    );
    let mut line = format!(
        "GIGA tool=gigatoken version={REV} kind={kind} chunk={chunk} bytes={} calls={} ids={total} cold_s={} \
         pass_s={pass:.6} warm_s={warm:.6} lang_s={lang:.6} warmo_s={warmo:.6} coldo_s={coldo:.6} reps={reps} cold_reps={cold_reps} load_ms={load_ms:.0} \
         fresh_state_ms={setup_ms:.3} cache_entries={} cache_mib={} pass_after={}",
        buf.len(),
        texts.len(),
        if tm.s[0].is_empty() { "na".to_string() } else { format!("{cold:.6}") },
        tm.entries[2],
        budget.map_or(0, |b| b >> 20),
        if wtexts.is_empty() { "same" } else { "other" }
    );
    for st in 0..STATES.len() {
        if tm.s[st].is_empty() {
            continue;
        }
        let v: Vec<String> = tm.s[st].iter().map(|x| format!("{x:.6}")).collect();
        line += &format!(" {}_reps_s={}", STATES[st], v.join(","));
        if st > 0 && st < 5 {                          // coldo's states are fresh forks: no entries
            line += &format!(" entries_{}={}", STATES[st], tm.entries[st]);
        }
    }
    println!("{line}");
}

fn write_ids(path: &str, ids: &[u32]) {
    let mut b = Vec::with_capacity(ids.len() * 4);
    for &x in ids {
        b.extend_from_slice(&x.to_le_bytes());
    }
    fs::write(path, b).unwrap_or_else(|e| panic!("{path}: {e}"));
}
