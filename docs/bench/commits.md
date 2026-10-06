# before / after throughput: master f0edc39 (speed push start) vs 9b9c0cb

tools/bench/e2e_commits.sh (one pinned core, 4 KiB chunks, best of 3 reps a run, 3 abba rounds = 6 runs a side, the median shown); MB/s before -> after; ids sha equal in every cell. hf / tiktoken: their MB/s on a GB10 host from docs/bench/e2e.md (another run, for scale). Raw: docs/bench/raw/commits-f0edc39-9b9c0cb-*.log. gb10e's run-to-run spread was about +-10% in this run (fuzzers on the A725 cores), tr9970x's about +-2%.
host names are chipset keys (see docs/machines.md); home directories were replaced by $HOME in the receipts.

## tr9970x

```
HOST tr9970x Linux 6.8.0-117-generic x86_64 PIN 'taskset -c 12' CHUNK 4096 REPS 3 ROUNDS 3 A=$HOME/toks-ci/bench-before B=$HOME/toks-ci/bench-commits
UPTIME  16:42:35 up 68 days, 23:35,  5 users,  load average: 12.36, 8.20, 5.43
UPTIME  16:45:01 up 68 days, 23:37,  3 users,  load average: 9.31, 8.64, 6.01

cell                          cold MB/s          pass MB/s          warm MB/s    hf / tiktoken  load
llama3 en                 277 ->     285      391 ->     400      733 ->     737     5.3 /   29.7  12.1-12.4
llama3 code               391 ->     395      448 ->     453      746 ->     745     5.6 /   27.0  12.1-12.1
llama3 cjk                164 ->     175      175 ->     185      197 ->     208     5.7 /   25.7  11.8-12.1
o200k en                  289 ->     296      401 ->     409      724 ->     730     5.1 /   32.7  11.7-11.8
o200k code                422 ->     431      487 ->     495      817 ->     822     5.3 /   29.0  11.5-11.7
o200k cjk                 177 ->     185      189 ->     197      213 ->     222     5.4 /   21.7  11.3-11.5
qwen38 en                 290 ->     294      394 ->     396      666 ->     668     4.6 /   28.3  11.4-11.4
qwen38 code               312 ->     311      381 ->     380      551 ->     549     5.0 /   26.3  11.4-11.4
qwen38 cjk                172 ->     169      182 ->     179      203 ->     199     5.0 /    n/a  11.3-11.4
glm53 en                  289 ->     293      397 ->     401      734 ->     733     5.3 /   27.5  11.3-11.3
glm53 code                385 ->     391      439 ->     443      715 ->     716     5.5 /   26.8  11.3-11.3
glm53 cjk                 145 ->     149      153 ->     159      172 ->     179     5.9 /   24.7  11.2-11.3
kimik3 en                 173 ->     174      211 ->     212      279 ->     277     3.6 /   29.5  11.1-11.2
kimik3 code               229 ->     233      242 ->     248      301 ->     307     3.8 /   26.4  11.1-11.1
kimik3 cjk                128 ->     130      132 ->     135      145 ->     148     5.0 /   22.9  10.8-11.1
dsv3 en                   274 ->     272      392 ->     388      653 ->     649                   10.7-10.8
dsv3 code                 272 ->     272      348 ->     348      525 ->     529                   10.5-10.7
dsv3 cjk                  208 ->     207      222 ->     221      250 ->     248                   10.3-10.5
gemma4 en                 154 ->     173      240 ->     261      326 ->     337     5.4 /    n/a  10.2-10.3
gemma4 code               163 ->     164      259 ->     264      393 ->     394     5.4 /    n/a  10.0-10.2
gemma4 cjk                187 ->     207      231 ->     268      251 ->     295    11.1 /    n/a  9.9-10.0
uni_bgem3 en               39 ->      85       39 ->     125       39 ->     159                   9.8-9.9
uni_bgem3 code             47 ->     108       47 ->     123       47 ->     147                   9.6-9.9
uni_bgem3 cjk              75 ->     103       75 ->     103       75 ->     105                   9.6-9.9
wp-bert-uncased en         90 ->     229       90 ->     241       90 ->     258                   9.6-9.6
wp-bert-uncased code       77 ->     214       77 ->     225       77 ->     231                   9.6-9.6
wp-bert-uncased cjk        64 ->     150       64 ->     177       64 ->     178                   9.5-9.6
wp-minilm-l6 en           593 ->    1469      593 ->    1532      593 ->    1690                   9.5-9.5
wp-minilm-l6 code         565 ->    1541      566 ->    1628      569 ->    1729                   9.3-9.5
wp-minilm-l6 cjk          630 ->    1415      632 ->    1688      632 ->    1763                   9.3-9.3
```

## gb10e

```
HOST gb10e Linux 6.17.0-1021-nvidia aarch64 PIN 'taskset -c 9' CHUNK 4096 REPS 3 ROUNDS 3 A=$HOME/toks-ci/bench-before B=$HOME/toks-ci/bench-commits
UPTIME  12:42:35 up 3 days, 22:54,  2 users,  load average: 1.20, 1.25, 0.93
UPTIME  12:45:27 up 3 days, 22:57,  3 users,  load average: 1.96, 1.58, 1.12

cell                          cold MB/s          pass MB/s          warm MB/s    hf / tiktoken  load
llama3 en                 203 ->     193      275 ->     261      596 ->     584     5.3 /   29.7  1.2-1.3
llama3 code               411 ->     419      473 ->     479      744 ->     740     5.6 /   27.0  1.3-1.3
llama3 cjk                116 ->     119      121 ->     124      142 ->     148     5.7 /   25.7  1.3-1.4
o200k en                  192 ->     181      255 ->     251      555 ->     548     5.1 /   32.7  1.4-1.4
o200k code                447 ->     449      510 ->     512      803 ->     801     5.3 /   29.0  1.4-1.5
o200k cjk                 111 ->     115      118 ->     123      139 ->     147     5.4 /   21.7  1.5-1.6
qwen38 en                 243 ->     215      325 ->     285      600 ->     555     4.6 /   28.3  1.6-1.6
qwen38 code               330 ->     330      406 ->     407      562 ->     562     5.0 /   26.3  1.6-1.6
qwen38 cjk                126 ->     124      133 ->     127      158 ->     150     5.0 /    n/a  1.6-1.6
glm53 en                  196 ->     193      259 ->     254      567 ->     566     5.3 /   27.5  1.7-1.7
glm53 code                409 ->     410      465 ->     467      718 ->     718     5.5 /   26.8  1.7-1.7
glm53 cjk                  96 ->     105      101 ->     110      122 ->     129     5.9 /   24.7  1.7-1.8
kimik3 en                 139 ->     136      171 ->     168      266 ->     263     3.6 /   29.5  1.8-1.8
kimik3 code               249 ->     251      268 ->     271      328 ->     332     3.8 /   26.4  1.8-1.8
kimik3 cjk                 86 ->      88       92 ->      90      106 ->     103     5.0 /   22.9  1.8-1.8
dsv3 en                   214 ->     214      302 ->     312      562 ->     564                   1.8-1.8
dsv3 code                 279 ->     279      355 ->     357      512 ->     514                   1.8-1.8
dsv3 cjk                  152 ->     154      166 ->     164      199 ->     201                   1.8-1.9
gemma4 en                 104 ->     129      157 ->     179      276 ->     287     5.4 /    n/a  1.9-1.9
gemma4 code               156 ->     163      252 ->     266      403 ->     412     5.4 /    n/a  1.9-1.9
gemma4 cjk                156 ->     170      195 ->     219      237 ->     281    11.1 /    n/a  1.9-1.9
uni_bgem3 en               42 ->      79       42 ->     116       42 ->     153                   1.9-1.9
uni_bgem3 code             49 ->     112       49 ->     130       49 ->     155                   1.9-1.9
uni_bgem3 cjk              77 ->     102       77 ->     103       77 ->     106                   1.9-1.9
wp-bert-uncased en         93 ->     241       94 ->     256       94 ->     277                   1.9-1.9
wp-bert-uncased code       79 ->     226       79 ->     240       79 ->     247                   1.9-1.9
wp-bert-uncased cjk        63 ->     149       63 ->     174       63 ->     176                   1.9-1.9
wp-minilm-l6 en           625 ->    1586      626 ->    1676      629 ->    1876                   1.9-1.9
wp-minilm-l6 code         606 ->    1729      610 ->    1891      611 ->    2059                   1.9-2.0
wp-minilm-l6 cjk          626 ->    1441      629 ->    1725      629 ->    1789                   2.0-2.0
```


# the piece dictionary (src/gen/dict.c, kernels.md §6): before / after

tools/bench/e2e_commits.sh, e2e.md's states (cold: a fresh scratch before every call, back to back; pass and lang-x:
after the other corpora; warm: the memo's replay), best of 3 reps a run, abba rounds as each HOST line says, the
median of each side's runs shown; ids sha equal in every cell of every table (e2e_commits.py prints VOID otherwise).
Each table is tools/bench/e2e_commits.py over the raw log named above it. The trees in the HOST lines: dict-A =
4913495 (on m2ultra1 3647c5b, which differs from 4913495 in stream.c and toks.h only), dict-Bpp (m2ultra1: dict-B) =
9be533c, dict-C2 (m2ultra1: dict-C) = 9be533c with toks_dict_n = 0 (the list off, the same code); pub-1cb69dc3 =
1cb69dc, pub-10dee097 = 10dee09 (the list's library; later commits change tests and docs only), pub-10dee097-C =
10dee09 with toks_dict_n = 0.

## gb10e (GB10 X925 cpu 7, under the timing lock; builds on the A725s)

4913495 -> 9be533c, 4 KiB chunks, abba x5:
raw: docs/bench/raw/commits-4913495-9be533c-gb10e-4096.log

```
HOST gb10e Linux 6.17.0-1021-nvidia aarch64 PIN 'taskset -c 7' CHUNK 4096 REPS 3 ROUNDS 5 A=$HOME/toks-ci/toks/dict-A B=$HOME/toks-ci/toks/dict-Bpp
UPTIME  00:55:50 up 4 days, 11:08,  1 user,  load average: 0.13, 0.20, 0.62
UPTIME  01:02:06 up 4 days, 11:14,  1 user,  load average: 1.00, 0.80, 0.77

cell                 cold MB/s          pass MB/s          warm MB/s        lang-x MB/s    hf / tiktoken  load
llama3 en        265 ->     276      255 ->     259    10426 ->   10670      276 ->     280                   0.1-0.3
llama3 code      410 ->     430      293 ->     298    13970 ->   13970      309 ->     314                   0.3-0.4
llama3 ml        111 ->     112      123 ->     124      367 ->     368      129 ->     130                   0.4-0.7
llama3 cjk       148 ->     149      135 ->     135      383 ->     384      143 ->     143                   0.7-0.7
gpt2 en          294 ->     303      363 ->     369     5739 ->    5699      388 ->     393                   0.7-0.8
gpt2 code        320 ->     365      377 ->     392    10508 ->   10419      410 ->     424                   0.8-0.8
gpt2 ml          149 ->     151      173 ->     174      353 ->     354      178 ->     179                   0.8-0.8
gpt2 cjk         146 ->     147      148 ->     148      229 ->     230      152 ->     152                   0.8-0.9
o200k en         256 ->     280      227 ->     235     9187 ->    9228      243 ->     253                   0.9-0.9
o200k code       441 ->     488      286 ->     302    10245 ->   10245      300 ->     316                   0.9-1.1
o200k ml          97 ->      98      101 ->     101      351 ->     351      104 ->     104                   1.0-1.1
o200k cjk        161 ->     162      145 ->     145      424 ->     424      150 ->     151                   1.0-1.0
qwen38 en        276 ->     291      257 ->     263    11195 ->   11226      275 ->     281                   1.0-1.0
qwen38 code      379 ->     421      302 ->     310    14133 ->   13970      316 ->     325                   1.0-1.0
qwen38 ml        104 ->     106      111 ->     111      312 ->     313      115 ->     115                   1.0-1.0
qwen38 cjk       145 ->     146      136 ->     136      442 ->     443      141 ->     141                   1.0-1.0
glm53 en         272 ->     309      239 ->     252    10243 ->   10347      257 ->     270                   1.0-1.0
glm53 code       412 ->     464      274 ->     288     8781 ->    8720      288 ->     303                   1.0-1.0
glm53 ml         108 ->     111      120 ->     121      373 ->     373      124 ->     125                   1.0-1.0
glm53 cjk        124 ->     125      117 ->     117      341 ->     343      121 ->     122                   1.0-1.0
kimik3 en        274 ->     305      259 ->     271     9873 ->    9897      279 ->     291                   1.0-1.0
kimik3 code      458 ->     510      320 ->     334    10245 ->   10245      336 ->     348                   1.0-1.0
kimik3 ml        106 ->     109      119 ->     120      326 ->     327      123 ->     124                   1.0-1.0
kimik3 cjk       135 ->     136      126 ->     126      359 ->     360      130 ->     131                   1.0-1.0
```

Whole files, abba x3:
raw: docs/bench/raw/commits-4913495-9be533c-gb10e-whole.log

```
HOST gb10e Linux 6.17.0-1021-nvidia aarch64 PIN 'taskset -c 7' CHUNK 0 REPS 3 ROUNDS 3 A=$HOME/toks-ci/toks/dict-A B=$HOME/toks-ci/toks/dict-Bpp
UPTIME  01:02:07 up 4 days, 11:14,  1 user,  load average: 1.00, 0.80, 0.77
UPTIME  01:05:59 up 4 days, 11:18,  1 user,  load average: 1.00, 0.92, 0.83

cell                 cold MB/s          pass MB/s          warm MB/s        lang-x MB/s    hf / tiktoken  load
llama3 en        340 ->     348      274 ->     280      616 ->     622      277 ->     281                   1.0-1.0
llama3 code      466 ->     475      312 ->     319    13970 ->   13970      321 ->     327                   1.0-1.0
llama3 ml        135 ->     136      130 ->     131      163 ->     162      130 ->     130                   1.0-1.0
llama3 cjk       152 ->     153      143 ->     143      170 ->     172      143 ->     144                   1.0-1.0
gpt2 en          408 ->     415      390 ->     396      677 ->     680      398 ->     403                   1.0-1.0
gpt2 code        434 ->     453      391 ->     407    15377 ->   14814      418 ->     438                   1.0-1.0
gpt2 ml          179 ->     180      180 ->     181      208 ->     209      181 ->     182                   1.0-1.0
gpt2 cjk         152 ->     152      153 ->     154      159 ->     159      154 ->     154                   1.0-1.0
o200k en         307 ->     324      243 ->     253      580 ->     595      244 ->     254                   1.0-1.0
o200k code       506 ->     532      297 ->     314    13970 ->   13970      306 ->     322                   1.0-1.0
o200k ml         109 ->     110      105 ->     105      130 ->     131      105 ->     105                   1.0-1.0
o200k cjk        163 ->     165      152 ->     153      190 ->     191      153 ->     154                   1.0-1.0
qwen38 en        346 ->     362      268 ->     274      583 ->     592      269 ->     275                   1.0-1.0
qwen38 code      469 ->     490      313 ->     319    13660 ->   13511      321 ->     328                   1.0-1.0
qwen38 ml        118 ->     119      114 ->     115      141 ->     141      114 ->     115                   1.0-1.0
qwen38 cjk       149 ->     149      141 ->     141      172 ->     172      141 ->     142                   1.0-1.0
glm53 en         330 ->     358      255 ->     270      612 ->     630      258 ->     274                   1.0-1.0
glm53 code       466 ->     492      286 ->     300     8845 ->    8909      295 ->     309                   1.0-1.0
glm53 ml         132 ->     133      125 ->     127      162 ->     164      126 ->     127                   1.0-1.0
glm53 cjk        129 ->     131      122 ->     123      148 ->     149      123 ->     124                   1.0-1.0
kimik3 en        307 ->     322      246 ->     256      490 ->     499      247 ->     258                   1.0-1.0
kimik3 code      421 ->     440      291 ->     303    10332 ->   10245      298 ->     309                   1.0-1.0
kimik3 ml        109 ->     110      107 ->     108      128 ->     129      107 ->     108                   1.0-1.0
kimik3 cjk       120 ->     121      116 ->     117      134 ->     134      117 ->     117                   1.0-1.0
```

The replay at 9 rounds, whole code: 4913495 -> 9be533c, then 4913495 -> 9be533c with the list off:
raw: docs/bench/raw/commits-4913495-9be533c-gb10e-whole-replay9.log

```
HOST gb10e Linux 6.17.0-1021-nvidia aarch64 PIN 'taskset -c 7' CHUNK 0 REPS 3 ROUNDS 9 A=$HOME/toks-ci/toks/dict-A B=$HOME/toks-ci/toks/dict-Bpp
UPTIME  01:35:24 up 4 days, 11:47,  1 user,  load average: 0.05, 0.10, 0.30
UPTIME  01:36:33 up 4 days, 11:48,  1 user,  load average: 0.71, 0.29, 0.36

cell                 cold MB/s          pass MB/s          warm MB/s        lang-x MB/s    hf / tiktoken  load
gpt2 code        433 ->     451      392 ->     408    14993 ->   14993      420 ->     436                   0.1-0.3
llama3 code      464 ->     477      312 ->     319    13970 ->   13970      321 ->     327                   0.3-0.5
o200k code       503 ->     535      299 ->     314    13970 ->   13970      305 ->     323                   0.5-0.7
```

raw: docs/bench/raw/commits-4913495-dictoff-gb10e-whole-replay9.log

```
HOST gb10e Linux 6.17.0-1021-nvidia aarch64 PIN 'taskset -c 7' CHUNK 0 REPS 3 ROUNDS 9 A=$HOME/toks-ci/toks/dict-A B=$HOME/toks-ci/toks/dict-C2
UPTIME  01:36:34 up 4 days, 11:48,  1 user,  load average: 0.71, 0.29, 0.36
UPTIME  01:37:42 up 4 days, 11:49,  1 user,  load average: 0.91, 0.44, 0.40

cell                 cold MB/s          pass MB/s          warm MB/s        lang-x MB/s    hf / tiktoken  load
gpt2 code        433 ->     435      392 ->     393    14993 ->   14993      420 ->     421                   0.7-0.8
llama3 code      465 ->     467      311 ->     311    13970 ->   13970      320 ->     321                   0.8-0.8
o200k code       505 ->     504      299 ->     299    13970 ->   14295      306 ->     306                   0.8-0.9
```

dsv3 (e2e_commits.sh's default list; a later run on 1cb69dc -> 10dee09), 4 KiB chunks abba x5, then whole files
abba x3:
raw: docs/bench/raw/commits-1cb69dc-10dee09-gb10e-dsv3-4096.log

```
HOST gb10e Linux 6.17.0-1021-nvidia aarch64 PIN 'taskset -c 7' CHUNK 4096 REPS 3 ROUNDS 5 A=$HOME/toks-ci/toks/pub-1cb69dc3 B=$HOME/toks-ci/toks/pub-10dee097
UPTIME  13:34:56 up 5 days, 23:47,  1 user,  load average: 0.13, 0.08, 0.07
UPTIME  13:36:35 up 5 days, 23:48,  1 user,  load average: 0.83, 0.34, 0.17

cell               cold MB/s          pass MB/s          warm MB/s        lang-x MB/s    hf / tiktoken  load
dsv3 en        255 ->     262      292 ->     294     6556 ->    6545      320 ->     323                   0.1-0.4
dsv3 code      311 ->     330      319 ->     326    13970 ->   14295      335 ->     342                   0.4-0.6
dsv3 ml        118 ->     119      163 ->     163      292 ->     292      171 ->     171                   0.6-0.8
dsv3 cjk       179 ->     180      171 ->     172      535 ->     535      183 ->     183                   0.8-0.8
```

raw: docs/bench/raw/commits-1cb69dc-10dee09-gb10e-dsv3-whole.log

```
HOST gb10e Linux 6.17.0-1021-nvidia aarch64 PIN 'taskset -c 7' CHUNK 0 REPS 3 ROUNDS 3 A=$HOME/toks-ci/toks/pub-1cb69dc3 B=$HOME/toks-ci/toks/pub-10dee097
UPTIME  13:36:35 up 5 days, 23:48,  1 user,  load average: 0.84, 0.35, 0.17
UPTIME  13:37:36 up 5 days, 23:49,  1 user,  load average: 0.94, 0.47, 0.23

cell               cold MB/s          pass MB/s          warm MB/s        lang-x MB/s    hf / tiktoken  load
dsv3 en        345 ->     350      307 ->     312      568 ->     574      310 ->     315                   0.8-0.9
dsv3 code      405 ->     416      332 ->     341    13660 ->   13660      341 ->     348                   0.9-0.9
dsv3 ml        166 ->     165      170 ->     170      185 ->     186      170 ->     171                   0.9-0.9
dsv3 cjk       180 ->     181      177 ->     179      210 ->     211      178 ->     179                   0.9-0.9
```

The replay cells that read lower at 5 or 3 rounds above (qwen38 code 4 KiB and whole, glm53 code 4 KiB), at 9
rounds: 1cb69dc -> 10dee09, then 1cb69dc -> 10dee09 with the list off (the replay never reads the words table: K5
does not run on a replay; its ~40 us passes are timed at 1 us):

raw: docs/bench/raw/commits-1cb69dc-10dee09-gb10e-replay9-4096.log

```
HOST gb10e Linux 6.17.0-1021-nvidia aarch64 PIN 'taskset -c 7' CHUNK 4096 REPS 3 ROUNDS 9 A=$HOME/toks-ci/toks/pub-1cb69dc3 B=$HOME/toks-ci/toks/pub-10dee097
UPTIME  13:37:36 up 5 days, 23:49,  1 user,  load average: 0.94, 0.47, 0.23
UPTIME  13:39:11 up 5 days, 23:51,  1 user,  load average: 0.99, 0.62, 0.31

cell                 cold MB/s          pass MB/s          warm MB/s        lang-x MB/s    hf / tiktoken  load
qwen38 code      380 ->     419      304 ->     314    13660 ->   13660      319 ->     330                   0.9-1.0
glm53 code       413 ->     464      281 ->     292     8537 ->    8598      295 ->     308                   1.0-1.0
```

raw: docs/bench/raw/commits-1cb69dc-10dee09-gb10e-replay9-whole.log

```
HOST gb10e Linux 6.17.0-1021-nvidia aarch64 PIN 'taskset -c 7' CHUNK 0 REPS 3 ROUNDS 9 A=$HOME/toks-ci/toks/pub-1cb69dc3 B=$HOME/toks-ci/toks/pub-10dee097
UPTIME  13:39:12 up 5 days, 23:51,  1 user,  load average: 0.99, 0.62, 0.31
UPTIME  13:40:47 up 5 days, 23:53,  1 user,  load average: 1.00, 0.73, 0.38

cell                 cold MB/s          pass MB/s          warm MB/s        lang-x MB/s    hf / tiktoken  load
qwen38 code      472 ->     488      313 ->     322    13221 ->   13363      321 ->     330                   1.0-1.0
glm53 code       466 ->     497      289 ->     303     8598 ->    8658      297 ->     312                   1.0-1.0
```

raw: docs/bench/raw/commits-1cb69dc-dictoff-gb10e-replay9-4096.log

```
HOST gb10e Linux 6.17.0-1021-nvidia aarch64 PIN 'taskset -c 7' CHUNK 4096 REPS 3 ROUNDS 9 A=$HOME/toks-ci/toks/pub-1cb69dc3 B=$HOME/toks-ci/toks/pub-10dee097-C
UPTIME  13:40:48 up 5 days, 23:53,  1 user,  load average: 1.00, 0.73, 0.38
UPTIME  13:42:23 up 5 days, 23:54,  1 user,  load average: 1.00, 0.80, 0.44

cell                 cold MB/s          pass MB/s          warm MB/s        lang-x MB/s    hf / tiktoken  load
qwen38 code      379 ->     380      306 ->     306    13363 ->   13660      320 ->     320                   1.0-1.0
glm53 code       412 ->     413      280 ->     279     8537 ->    8537      293 ->     294                   1.0-1.0
```

raw: docs/bench/raw/commits-1cb69dc-dictoff-gb10e-replay9-whole.log

```
HOST gb10e Linux 6.17.0-1021-nvidia aarch64 PIN 'taskset -c 7' CHUNK 0 REPS 3 ROUNDS 9 A=$HOME/toks-ci/toks/pub-1cb69dc3 B=$HOME/toks-ci/toks/pub-10dee097-C
UPTIME  13:42:23 up 5 days, 23:54,  1 user,  load average: 1.00, 0.80, 0.44
UPTIME  13:43:58 up 5 days, 23:56,  1 user,  load average: 1.00, 0.86, 0.50

cell                 cold MB/s          pass MB/s          warm MB/s        lang-x MB/s    hf / tiktoken  load
qwen38 code      473 ->     472      312 ->     313    13221 ->   13363      320 ->     320                   1.0-1.0
glm53 code       467 ->     465      289 ->     287     8537 ->    8598      296 ->     296                   1.0-1.0
```

## m2ultra1 (M2 Ultra, unpinned: shape)

3647c5b -> 9be533c, 4 KiB chunks abba x5, whole files abba x3, the replay at 9 rounds (whole code) with and without
the list:

raw: docs/bench/raw/commits-3647c5b-9be533c-m2ultra1-4096.log

```
HOST m2ultra1 Darwin 25.5.0 arm64 PIN '' CHUNK 4096 REPS 3 ROUNDS 5 A=$HOME/toks-ci/toks/dict-A B=$HOME/toks-ci/toks/dict-B
UPTIME 22:28  up 32 days,  3:43, 0 users, load averages: 3.02 1.86 1.60
UPTIME 22:30  up 32 days,  3:45, 0 users, load averages: 2.02 1.85 1.62

cell                 cold MB/s          pass MB/s          warm MB/s        lang-x MB/s    hf / tiktoken  load
kimik3 en        260 ->     291      362 ->     380     9596 ->    9596      374 ->     393                   3.0-3.2
kimik3 code      388 ->     429      414 ->     434     8658 ->    8421      432 ->     452                   2.9-3.0
llama3 en        265 ->     275      371 ->     377    10243 ->   10142      384 ->     389                   2.7-2.8
llama3 code      357 ->     372      399 ->     405    11176 ->   10977      417 ->     423                   2.5-2.7
o200k en         267 ->     291      344 ->     360     9146 ->    9228      357 ->     372                   2.1-2.3
o200k code       390 ->     427      402 ->     422     8781 ->    8781      419 ->     442                   2.0-2.1
```

raw: docs/bench/raw/commits-3647c5b-9be533c-m2ultra1-whole.log

```
HOST m2ultra1 Darwin 25.5.0 arm64 PIN '' CHUNK 0 REPS 3 ROUNDS 3 A=$HOME/toks-ci/toks/dict-A B=$HOME/toks-ci/toks/dict-B
UPTIME 22:30  up 32 days,  3:45, 0 users, load averages: 1.94 1.83 1.61
UPTIME 22:31  up 32 days,  3:45, 0 users, load averages: 1.81 1.81 1.62

cell                 cold MB/s          pass MB/s          warm MB/s        lang-x MB/s    hf / tiktoken  load
kimik3 en        334 ->     352      326 ->     343      534 ->     544      329 ->     345                   1.9-1.9
kimik3 code      380 ->     394      358 ->     375     8909 ->    8909      368 ->     384                   1.9-1.9
llama3 en        394 ->     402      384 ->     392      707 ->     712      388 ->     397                   1.8-1.9
llama3 code      432 ->     439      407 ->     415    12806 ->   11937      419 ->     429                   1.7-1.8
o200k en         383 ->     403      356 ->     373      683 ->     697      360 ->     376                   1.7-1.9
o200k code       468 ->     489      410 ->     429    11383 ->   10814      424 ->     444                   1.8-1.9
```

raw: docs/bench/raw/commits-3647c5b-9be533c-m2ultra1-whole-replay9.log

```
HOST m2ultra1 Darwin 25.5.0 arm64 PIN '' CHUNK 0 REPS 3 ROUNDS 9 A=$HOME/toks-ci/toks/dict-A B=$HOME/toks-ci/toks/dict-B
UPTIME 22:42  up 32 days,  3:56, 0 users, load averages: 1.49 1.51 1.50
UPTIME 22:43  up 32 days,  3:58, 0 users, load averages: 2.23 1.78 1.60

cell                 cold MB/s          pass MB/s          warm MB/s        lang-x MB/s    hf / tiktoken  load
kimik3 code      380 ->     395      358 ->     374     8974 ->    9040      367 ->     383                   1.5-2.5
llama3 code      430 ->     440      408 ->     415    12294 ->   12545      421 ->     428                   2.2-2.5
o200k code       469 ->     488      409 ->     429    11383 ->   11383      423 ->     444                   2.2-2.4
```

raw: docs/bench/raw/commits-3647c5b-dictoff-m2ultra1-whole-replay9.log

```
HOST m2ultra1 Darwin 25.5.0 arm64 PIN '' CHUNK 0 REPS 3 ROUNDS 9 A=$HOME/toks-ci/toks/dict-A B=$HOME/toks-ci/toks/dict-C
UPTIME 22:43  up 32 days,  3:58, 0 users, load averages: 2.23 1.78 1.60
UPTIME 22:44  up 32 days,  3:59, 0 users, load averages: 1.93 1.78 1.61

cell                 cold MB/s          pass MB/s          warm MB/s        lang-x MB/s    hf / tiktoken  load
kimik3 code      380 ->     379      359 ->     359     8909 ->    8909      368 ->     368                   2.0-2.2
llama3 code      430 ->     433      407 ->     407    12294 ->   12294      420 ->     419                   2.0-2.1
o200k code       468 ->     469      409 ->     409    11710 ->   10691      423 ->     422                   1.9-2.0
```

Against gigatoken (fac0114b), tools/bench/gate.sh, default config, ROUNDS 3, REPS 5: A = 245cc5c, B = 42e16c9 (the
list on 245cc5c), raw: docs/bench/raw/gate-m2ultra1-neon-245cc5c.log and gate-m2ultra1-neon-42e16c9.log; the before
-> after is `python3 tools/bench/gate_table.py m2ultra1-neon=docs/bench/raw/gate-m2ultra1-neon-42e16c9.log --before
m2ultra1-neon=docs/bench/raw/gate-m2ultra1-neon-245cc5c.log` (kimi k3 4 KiB, gigatoken / toks: en cold 1.48x ->
1.65x, coldo 1.45x -> 1.58x, pass 1.11x -> 1.17x; code cold 1.72x -> 1.89x, coldo 1.64x -> 1.82x, pass 1.10x ->
1.15x; every cell a WIN, shape).

## measured and not taken (kernels.md §5 / §6), raw logs (gb10e cpu 7, 4 KiB abba x5 and whole abba x3)

- commits-4913495-doubled-dict-gb10e-{4096,whole}.log: the list in a words table doubled where that adds <= 4 MiB
  (B = the branch's tree then); commits-4913495-doubled-gb10e-{4096,whole}.log: the same doubled table without the
  list (dict-C).
- commits-9be533c-moves-gb10e-{4096,whole}.log: the list seated by moving entries to their other bucket (dict-M, on
  9be533c).
- commits-caa6427-seat-gb10e-{4096,whole}.log: words_seat taking the highest-id entry that can go (dict-S, on
  caa6427 = dict-Bm).
- commits-15e3e02-bloom{clear,set,off}-gb10e-4096.log: a second table K6 probes, 15e3e02 (dict-B5) against a scratch
  K6 neon that never hits: the bloom all clear (dict-KZ), all set (dict-KF), the probe off (dict-KN).
