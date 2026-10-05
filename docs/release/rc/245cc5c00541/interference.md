# Other jobs during the 0.3.0 rc's timed legs, and what was done about them

The rc's bench legs ran under each linux host's `~/toks-ci/timing.lock` (gb10c, cpu 8; tr9970x,
cpu 20) and by announcement on m2ultra2. Four `make test` runs (make -j10 / -j14) of other work on the same
machines still landed inside them: on gb10c's cores 10-19 (the other GB10 cluster) 07:08:29-07:09:20Z and
07:30:21-07:31:10Z, on tr9970x's CCD3 (cores 25-31, 57-63) 06:52:48-06:53:36Z and 07:09:21-07:10:09Z. The bench core and its SMT sibling stayed clean
in every cell (the BUSY lines: cpu20 98-100%, cpu52 1-4%; the GB10 has no SMT). What follows is what each did to the
numbers and what the table does with it.

## tr9970x: measured, no effect, the cells stand

The control is gigatoken (fac0114b, the same build and rustc) per cell against two older quiet runs of the same cells,
`docs/bench/raw/tr9970x-avx2-5f71528328b8.log` and `-f5af6566b066.log` (cpu 12). A burst cell is one whose 1-min load
rose by 1.0 or more over the cell, or started at 5 or more: 26 of 88 (both CCD3 runs among them).

```
giga cold now/ctl: median 0.994 p10 0.949 p90 1.035 min 0.808 max 1.072
ctl/ctl cold:      median 0.997 p10 0.960 p90 1.027 min 0.872 max 1.173
median giga cold now/ctl: burst 1.001 (n=26) quiet 0.991 (n=62)
median giga pass now/ctl: burst 0.988 quiet 0.968
```

The burst cells (gigatoken cold and pass, now / control, then control / control):

```
gpt2           en   4096 load  7.15 ->  5.99  giga cold now/ctl 0.965  pass 0.909  ctl/ctl 0.979 BURST
gpt2           en   0    load  5.99 ->  5.59  giga cold now/ctl 0.987  pass 0.936  ctl/ctl 1.000 BURST
gpt2           code 4096 load  5.59 ->  5.26  giga cold now/ctl 0.986  pass 0.938  ctl/ctl 0.982 BURST
gpt2           code 0    load  5.26 ->  4.92  giga cold now/ctl 1.001  pass 0.942  ctl/ctl 1.022 BURST
llama3         en   4096 load  2.97 ->  4.20  giga cold now/ctl 1.056  pass 0.804  ctl/ctl 0.978 BURST
llama3         ml   4096 load  3.29 ->  9.28  giga cold now/ctl 0.965  pass 1.026  ctl/ctl 1.027 BURST
llama3         ml   0    load  9.28 ->  6.93  giga cold now/ctl 0.977  pass 1.004  ctl/ctl 0.976 BURST
llama3         cjk  4096 load  6.93 ->  5.61  giga cold now/ctl 0.965  pass 1.002  ctl/ctl 0.979 BURST
llama3         cjk  0    load  5.61 ->  5.35  giga cold now/ctl 0.995  pass 0.983  ctl/ctl 0.981 BURST
glm53          en   4096 load  5.35 ->  4.38  giga cold now/ctl 1.072  pass 0.839  ctl/ctl 1.023 BURST
glm53          ml   4096 load  3.60 ->  4.87  giga cold now/ctl 0.974  pass 1.016  ctl/ctl 1.033 BURST
qwen38         ml   4096 load  1.71 ->  4.13  giga cold now/ctl 0.978  pass 0.995  ctl/ctl 0.986 BURST
qwen38         ml   0    load  4.13 ->  6.65  giga cold now/ctl 1.011  pass 1.000  ctl/ctl 0.996 BURST
qwen38         cjk  4096 load  6.65 ->  5.02  giga cold now/ctl 0.995  pass 1.016  ctl/ctl 1.001 BURST
qwen38         cjk  0    load  5.02 ->  4.40  giga cold now/ctl 1.004  pass 1.000  ctl/ctl 1.007 BURST
gemma4         en   0    load  1.59 ->  5.34  giga cold now/ctl 0.949  pass 0.871  ctl/ctl 0.994 BURST
gemma4         code 4096 load  5.34 ->  6.27  giga cold now/ctl 1.022  pass 0.893  ctl/ctl 0.987 BURST
gemma4         code 0    load  6.27 ->  5.46  giga cold now/ctl 0.996  pass 0.891  ctl/ctl 1.025 BURST
gemma4         ml   4096 load  5.46 ->  4.85  giga cold now/ctl 1.035  pass 1.014  ctl/ctl 0.948 BURST
minimaxm2      en   4096 load  1.92 ->  3.76  giga cold now/ctl 1.039  pass 0.920  ctl/ctl 0.950 BURST
minimaxm2      code 4096 load  4.62 ->  7.35  giga cold now/ctl 1.001  pass 0.992  ctl/ctl 0.980 BURST
minimaxm2      code 0    load  7.35 ->  6.68  giga cold now/ctl 1.001  pass 0.926  ctl/ctl 1.014 BURST
minimaxm2      ml   4096 load  6.68 ->  2.34  giga cold now/ctl 1.010  pass 0.972  ctl/ctl 0.997 BURST
minimaxm2      cjk  4096 load  1.96 ->  3.35  giga cold now/ctl 1.015  pass 1.003  ctl/ctl 0.964 BURST
minimaxm2      cjk  0    load  3.35 ->  4.62  giga cold now/ctl 1.005  pass 1.005  ctl/ctl 1.007 BURST
dsv4           ml   4096 load  2.58 ->  4.02  giga cold now/ctl 1.002  pass 1.006  ctl/ctl 1.010 BURST
```

Every burst cell is inside the band the two quiet controls span between themselves, so tr9970x's cells are the rc's
as measured.

## gb10c: re-measured, merged

Decided 2026-10-05 07:41Z: every gb10c cell whose timed window overlapped either of its two runs is re-measured on the same
core, under the lock, after the leg. The overlapped cells are where the bench log's per-cell LOAD lines jump (gb10c
reads 1.00 with the bench alone): glm53 cjk 4096 and whole, llama4 code 4096 and whole; the cell after each burst
(qwen38 en 4096, llama4 ml 4096) and llama4 ml whole were re-measured with them. `tools/bench/e2e.sh` (REPS 5, toks,
gigatoken, hf, tiktoken), PINCPU 8, `taskset -c 8`, the rc's own tree on gb10c, 2026-10-05 08:14:52-08:19:56Z, load
0.63-1.09; logs `docs/bench/raw/gb10c-neon-245cc5c00541-rerun-*.log`, all 7 cells exact. Re-run / rc, MB/s:

```
== glm53 cjk chunk 4096  load rc 1.05->3.82  re-run 0.63->0.79  exact rc yes re-run yes
   cold  toks    123.9 ->    123.9 (0.999)   giga     17.3 ->     17.4 (1.007)   toks/giga  7.15 ->  7.10
   pass  toks    114.9 ->    113.8 (0.991)   giga     34.0 ->     34.9 (1.026)   toks/giga  3.38 ->  3.26
   warm  toks    337.6 ->    334.5 (0.991)   giga    261.7 ->    281.8 (1.077)   toks/giga  1.29 ->  1.19
== glm53 cjk chunk 0  load rc 3.82->4.52  re-run 0.79->0.83  exact rc yes re-run yes
   cold  toks    124.0 ->    131.5 (1.060)   giga     33.4 ->     34.9 (1.044)   toks/giga  3.72 ->  3.77
   pass  toks    116.1 ->    121.1 (1.043)   giga     34.1 ->     34.8 (1.020)   toks/giga  3.40 ->  3.48
   warm  toks    143.2 ->    147.7 (1.032)   giga    255.7 ->    276.5 (1.081)   toks/giga  0.56 ->  0.53
== qwen38 en chunk 4096  load rc 4.52->2.20  re-run 0.83->0.98  exact rc yes re-run yes
   cold  toks    281.6 ->    279.1 (0.991)   giga    105.9 ->    107.1 (1.011)   toks/giga  2.66 ->  2.61
   pass  toks    239.2 ->    254.9 (1.066)   giga    246.1 ->    245.7 (0.998)   toks/giga  0.97 ->  1.04
   warm  toks  11134.0 ->  11256.3 (1.011)   giga    598.5 ->    600.1 (1.003)   toks/giga 18.60 -> 18.76
== llama4 code chunk 4096  load rc 1.00->3.23  re-run 0.98->0.98  exact rc yes re-run yes
   cold  toks    460.1 ->    462.5 (1.005)   giga     94.5 ->    100.8 (1.067)   toks/giga  4.87 ->  4.59
   pass  toks    295.2 ->    291.7 (0.988)   giga    207.5 ->    224.8 (1.084)   toks/giga  1.42 ->  1.30
   warm  toks  13660.0 ->  13970.4 (1.023)   giga    603.8 ->    609.8 (1.010)   toks/giga 22.62 -> 22.91
== llama4 code chunk 0  load rc 3.23->4.26  re-run 0.98->0.99  exact rc yes re-run yes
   cold  toks    521.4 ->    524.5 (1.006)   giga    256.7 ->    261.6 (1.019)   toks/giga  2.03 ->  2.01
   pass  toks    277.4 ->    300.7 (1.084)   giga    211.5 ->    227.7 (1.076)   toks/giga  1.31 ->  1.32
   warm  toks  11598.1 ->  13970.4 (1.205)   giga    621.5 ->    632.4 (1.017)   toks/giga 18.66 -> 22.09
== llama4 ml chunk 4096  load rc 4.26->2.91  re-run 0.99->1.09  exact rc yes re-run yes
   cold  toks    105.2 ->    106.8 (1.016)   giga     28.0 ->     28.9 (1.034)   toks/giga  3.76 ->  3.69
   pass  toks    125.6 ->    127.8 (1.018)   giga     90.6 ->     90.5 (0.999)   toks/giga  1.39 ->  1.41
   warm  toks    250.5 ->    250.5 (1.000)   giga    384.2 ->    385.6 (1.004)   toks/giga  0.65 ->  0.65
== llama4 ml chunk 0  load rc 2.91->2.26  re-run 1.09->1.06  exact rc yes re-run yes
   cold  toks    137.2 ->    136.3 (0.993)   giga     91.0 ->     90.4 (0.993)   toks/giga  1.51 ->  1.51
   pass  toks    131.9 ->    132.0 (1.000)   giga     90.9 ->     90.7 (0.998)   toks/giga  1.45 ->  1.46
   warm  toks    147.0 ->    147.6 (1.004)   giga    385.5 ->    382.1 (0.991)   toks/giga  0.38 ->  0.39

largest moves (|re-run / rc - 1|):
    20.5%  toks warm  llama4 code 0
     8.4%  toks pass  llama4 code 0
     8.4%  giga pass  llama4 code 4096
     8.1%  giga warm  glm53 cjk 0
     7.7%  giga warm  glm53 cjk 4096
     7.6%  giga pass  llama4 code 0
     6.7%  giga cold  llama4 code 4096
     6.6%  toks pass  qwen38 en 4096
```

Here the bursts moved cells by 6-8% (one memo replay by 20%): on the GB10 both clusters share the memory system, so a
job on the other cluster is not off the bench core's path the way another CCD is on the 9970X. `docs/bench/e2e.md`
takes both runs under the label gb10c-neon, `tools/bench/e2e_table.py`'s rule for one label given several logs: the
best run per cell and state for toks, the fastest comparator time over the host's logs, and every run's load before
and after in the cell's load column (a re-measured cell shows four values). The command (the arguments
`tools/release/rc.sh` passes, plus the three re-run logs under gb10c-neon):

```
uv run tools/bench/e2e_table.py gb10c-neon=docs/bench/raw/gb10c-neon-245cc5c00541.log \
    gb10c-neon=docs/bench/raw/gb10c-neon-245cc5c00541-rerun-glm53-cjk.log \
    gb10c-neon=docs/bench/raw/gb10c-neon-245cc5c00541-rerun-qwen38-en.log \
    gb10c-neon=docs/bench/raw/gb10c-neon-245cc5c00541-rerun-llama4-code-ml.log \
    gb10c-scalar=docs/bench/raw/gb10c-scalar-245cc5c00541.log m2ultra2-neon=docs/bench/raw/m2ultra2-neon-245cc5c00541.log \
    tr9970x-avx2=docs/bench/raw/tr9970x-avx2-245cc5c00541.log tr9970x-scalar=docs/bench/raw/tr9970x-scalar-245cc5c00541.log \
    tokv1-gb10c=docs/bench/raw/tokv1-gb10c-245cc5c00541.log tokv1-tr9970x=docs/bench/raw/tokv1-tr9970x-245cc5c00541.log \
    > docs/bench/e2e.md
```

What the merge changes in the Gates: gb10c-neon pass vs gigatoken pass 82 / 85 -> 83 / 85 (qwen38 en 4096, 0.97x ->
1.04x); every count the release report prints (tiktoken, gigatoken cold, gigatoken warm, per host and tier) is the
same with and without it.
