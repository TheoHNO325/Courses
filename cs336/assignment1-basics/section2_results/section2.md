(base) root@autodl-container-b45f449c7a-9d482c82:~/autodl-tmp/cs336/assignment1-basics# py-spy record -o profile.svg -- python train_bpe.py
py-spy> Sampling process 100 times a second. Press Control-C to exit.

耗时: 268.06s, 内存峰值: 124 MB (增量)
10000
9743
最长的片段 b' accomplishment'

py-spy> Stopped sampling because process exited
py-spy> Wrote flamegraph data to 'profile.svg'. Samples: 14796 Errors: 33
(base) root@autodl-container-b45f449c7a-9d482c82:~/autodl-tmp/cs336/assignment1-basics# 


(base) root@autodl-container-b45f449c7a-9d482c82:~/autodl-tmp/cs336/assignment1-basics# python -m cProfile -o bpe_results.prof train_bpe.py
耗时: 802.61s, 内存峰值: 647 MB (增量)
10000
9743
最长的片段 b' accomplishment'
已保存 vocab.json 和 merges.json

(base) root@autodl-container-b45f449c7a-9d482c82:~/autodl-tmp/cs336/assignment1-basics# snakeviz bpe_results.prof
snakeviz web server started on 127.0.0.1:8080; enter Ctrl-C to exit
http://127.0.0.1:8080/snakeviz/%2Froot%2Fautodl-tmp%2Fcs336%2Fassignment1-basics%2Fbpe_results.prof

Bye!

注：以上在tiny stories上train，预分词process取4，max vocabsize=10000

---

