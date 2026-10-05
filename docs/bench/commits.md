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

