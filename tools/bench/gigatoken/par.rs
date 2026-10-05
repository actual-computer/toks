//! tools/bench/gigatoken/par.rs: gigatoken's PARALLEL api (encode_docs_ragged: its worker pool on rayon's
//! global pool of RAYON_NUM_THREADS threads, LPT chunks of >= 1 MiB, the overlapped gather into one flat
//! buffer) on exactly the inputs tests/par/bench_par.c measures: the toks_par receipts' comparator (the
//! runner tests/par/run_par.sh sets RAYON_NUM_THREADS and the cpus). Never a dependency of the library.
//!
//!   RAYON_NUM_THREADS=k taskset -c <cpus> gigatoken-par <tokenizer.json> <text> doc <MiB,..> [reps]
//!   RAYON_NUM_THREADS=k taskset -c <cpus> gigatoken-par <tokenizer.json> <text> batch <doc bytes> <MiB,..> [reps]
//!
//! doc: S MiB of the text as one document; batch: cut into documents after the first '\n' at or past every
//! <doc bytes> boundary (bench_par.c's and e2e.c's rule). Windows and states, bench_par.c's: the text holds
//! K = len / S disjoint windows of S bytes; first = a fresh WorkerPool's first call (its lazy forks included) on
//! window 0; pass = reps samples, sample i a fresh pool that encodes window i + 1 mod K untimed, then window
//! i mod K timed (K < 2: warmed on the text's last S bytes, which overlap: pass_seen > 0, HALF-WARM); pass_ms =
//! the median sample; warm = the last sample's documents again (best and median of reps). ids: fnv-1a-64 over
//! the flat id stream (no post-processor: gigatoken applies none), printed as fnv_nopp (window 0) and
//! pass_fnv_nopp (every sample's window) like bench_par.c's toks rows.
use std::{env, fs, hint::black_box, process::exit, time::Instant};

use gigatoken_rs::load_tokenizer::hf::{HfTokenizer, load_hf_slice};
use gigatoken_rs::{WorkerPool, encode_docs_ragged};

fn fnv(ids: &[u32]) -> u64 {
    let mut h = 0xCBF29CE484222325u64;
    for &i in ids {
        h = (h ^ i as u64).wrapping_mul(0x100000001B3);
    }
    h
}

fn cut_docs(t: &[u8], doc: usize) -> Vec<&[u8]> {
    let (mut out, mut p, n) = (Vec::new(), 0usize, t.len());
    while p < n {
        let mut e = n;
        if doc > 0 && p + doc < n {
            e = p + doc;
            while e < n && t[e - 1] != b'\n' {
                e += 1;
            }
        }
        out.push(&t[p..e]);
        p = e;
    }
    out
}

/// the pass state's sample i: its timed window, and its warm-up text (bench_par.c's rule)
fn pass_text(buf: &[u8], len: usize, i: usize) -> &[u8] {
    let k = buf.len() / len;
    if k >= 2 { &buf[i % k * len..(i % k + 1) * len] } else { &buf[..len] }
}

fn pass_warm(buf: &[u8], len: usize, i: usize) -> &[u8] {
    let k = buf.len() / len;
    if k >= 2 { &buf[(i + 1) % k * len..((i + 1) % k + 1) * len] } else { &buf[buf.len() - len..] }
}

fn ms(v: &[f64]) -> String {
    v.iter().map(|x| format!("{:.3}", x * 1e3)).collect::<Vec<_>>().join(",")
}

fn main() {
    let a: Vec<String> = env::args().collect();
    let usage = || -> ! {
        eprintln!("usage: gigatoken-par <tokenizer.json> <text> doc <MiB,..> [reps] | batch <doc bytes> <MiB,..> [reps]");
        exit(2)
    };
    if a.len() < 5 {
        usage();
    }
    let data = fs::read(&a[1]).unwrap_or_else(|e| panic!("{}: {e}", a[1]));
    let base = match load_hf_slice(&data) {
        Ok(HfTokenizer::Bpe(t)) => t,
        Ok(_) => {
            println!("GIGA na=not_byte_level_bpe");
            exit(0)
        }
        Err(e) => {
            println!("GIGA na=load:{}", e.to_string().replace(char::is_whitespace, "_"));
            exit(0)
        }
    };
    let buf = fs::read(&a[2]).unwrap_or_else(|e| panic!("{}: {e}", a[2]));
    let (doc, sizes, reps) = match a[3].as_str() {
        "doc" => (0usize, &a[4], a.get(5)),
        "batch" if a.len() >= 6 => (a[4].parse().expect("doc bytes"), &a[5], a.get(6)),
        _ => usage(),
    };
    let reps: usize = reps.map(|r| r.parse().expect("reps")).unwrap_or(7).clamp(1, 64);
    let threads = env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".to_string());
    let tok = a[1].rsplit('/').next().unwrap_or(&a[1]);
    let mode = if doc == 0 { "doc" } else { "batch" };
    for s in sizes.split(',') {
        let len = (s.parse::<f64>().expect("MiB") * 1048576.0) as usize;
        if len == 0 || len > buf.len() {
            println!("GIGA mode={mode} tok={tok} doc={doc} bytes={len} threads={threads} skip=text_short text_bytes={}", buf.len());
            continue;
        }
        let docs_of = |t| if doc == 0 { vec![t] } else { cut_docs(t, doc) };
        let run = |w: &WorkerPool, d: &[&[u8]]| -> (f64, Vec<u32>) {
            let t0 = Instant::now();
            let (flat, lens) = encode_docs_ragged(w, &base, d);
            let dt = t0.elapsed().as_secs_f64();
            assert_eq!(lens.len(), d.len(), "one row per document");
            (dt, black_box(flat))
        };
        let docs = docs_of(&buf[..len]);
        let w = WorkerPool::new();
        let (first, ids) = run(&w, &docs);
        drop(w);
        let (mut pass, mut pass_fnv, mut last) = (Vec::with_capacity(reps), Vec::with_capacity(reps), None);
        for i in 0..reps {
            let w = WorkerPool::new();
            run(&w, &docs_of(pass_warm(&buf, len, i)));
            let (t, p_ids) = run(&w, &docs_of(pass_text(&buf, len, i)));
            pass.push(t);
            pass_fnv.push(format!("{:016x}", fnv(&p_ids)));
            last = Some(w);
        }
        let w = last.expect("reps >= 1");
        let wdocs = docs_of(pass_text(&buf, len, reps - 1));
        let wids = run(&w, &wdocs).1.len();
        let mut warm: Vec<f64> = (0..reps)
            .map(|_| {
                let (t, i) = run(&w, &wdocs);
                assert!(i.len() == wids, "warm id count");
                t
            })
            .collect();
        let pass_ms = ms(&pass);
        pass.sort_by(|x, y| x.partial_cmp(y).unwrap());
        warm.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let windows = buf.len() / len;
        let seen = if windows >= 2 { 0.0 } else { (2 * len - buf.len()) as f64 / len as f64 };
        let (mb, pm) = (len as f64 / 1e6, pass[reps / 2]);
        println!(
            "GIGA mode={mode} tok={tok} doc={doc} docs={} bytes={len} threads={threads} ids={} fnv_nopp={:016x} first_ms={:.3} \
             pass_ms={:.3} warm_best_ms={:.3} warm_med_ms={:.3} first_mbps={:.1} pass_mbps={:.1} warm_mbps={:.1} \
             pass_seen={seen:.2} windows={windows} pass_reps_ms={pass_ms} pass_fnv_nopp={}",
            docs.len(),
            ids.len(),
            fnv(&ids),
            first * 1e3,
            pm * 1e3,
            warm[0] * 1e3,
            warm[reps / 2] * 1e3,
            mb / first,
            mb / pm,
            mb / warm[0],
            pass_fnv.join(",")
        );
    }
}
