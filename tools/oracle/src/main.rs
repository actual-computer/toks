//! tiktoken-pieces: the piece oracle for tiktoken's pre-tokenizer split.
//!
//! tiktoken's CoreBPE (src/lib.rs of tiktoken 0.14.0) compiles the encoding's pat_str with
//! `fancy_regex::Regex::new(pattern)` (default options, so fancy-regex's default backtrack limit) and, between
//! two special tokens, takes the pieces of `regex.find_iter(&text[start..end])`; a regex runtime error (the
//! backtrack limit) makes encode fail. This program does exactly that for each input and prints the matches,
//! so a scanner can be diffed against tiktoken's engine piece by piece. It is pattern-agnostic: Kimi K3,
//! o200k_base, cl100k_base and r50k_base run through the same binary (tools/oracle/pieces.py holds the
//! pattern strings, read from their pinned sources).
//!
//! usage: tiktoken-pieces <pattern>
//!   stdin   records: u32 LE n, then n bytes of text (one segment: the text between two special tokens)
//!   stdout  per record, in order: u32 LE k, then k pairs (u32 LE start, u32 LE end) of byte offsets, the
//!           matches of find_iter in order; or u32 LE 0xFFFFFFFF, u32 LE m, m bytes of a message (the text is
//!           not utf-8, or the regex failed at run time).
//!   exit    0; 2 when the pattern does not compile (the message on stderr).
use std::io::{self, BufReader, BufWriter, Read, Write};

fn main() -> io::Result<()> {
    let pat = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: tiktoken-pieces <pattern>");
            std::process::exit(2);
        }
    };
    let re = match fancy_regex::Regex::new(&pat) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("tiktoken-pieces: pattern does not compile: {e}");
            std::process::exit(2);
        }
    };
    let mut inp = BufReader::with_capacity(1 << 20, io::stdin().lock());
    let mut out = BufWriter::with_capacity(1 << 20, io::stdout().lock());
    let mut lenb = [0u8; 4];
    let mut buf: Vec<u8> = Vec::new();
    let mut se: Vec<u32> = Vec::new();
    loop {
        match inp.read_exact(&mut lenb) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
        let n = u32::from_le_bytes(lenb) as usize;
        buf.resize(n, 0);
        inp.read_exact(&mut buf)?;
        se.clear();
        let err: Option<String> = match std::str::from_utf8(&buf) {
            Err(e) => Some(format!("invalid utf-8: {e}")),
            Ok(s) => {
                let mut err = None;
                for m in re.find_iter(s) {
                    match m {
                        Ok(m) => {
                            se.push(m.start() as u32);
                            se.push(m.end() as u32);
                        }
                        Err(e) => {
                            err = Some(format!("regex: {e}"));
                            break;
                        }
                    }
                }
                err
            }
        };
        match err {
            None => {
                out.write_all(&((se.len() / 2) as u32).to_le_bytes())?;
                for v in &se {
                    out.write_all(&v.to_le_bytes())?;
                }
            }
            Some(m) => {
                out.write_all(&u32::MAX.to_le_bytes())?;
                out.write_all(&(m.len() as u32).to_le_bytes())?;
                out.write_all(m.as_bytes())?;
            }
        }
    }
    out.flush()
}
