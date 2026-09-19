'''
PYTHONPATH=tests:. uv run pytest tests/test_train_bpe.py -q
'''

from cs336_basics.pretokenization_example import find_chunk_boundaries
import multiprocessing as mp
import regex as re
from collections import Counter
from pathlib import Path
import pickle

# def init_vocab(length=256):
#     vocab = { 0: b"<|endoftext|>"}
#     for i in range(1,length+1):
#         vocab[i] = bytes([i-1])
#     return vocab
import os,sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 路径约定：由脚本自身位置推导，Windows / Linux 通用
PROJECT_DIR = Path(__file__).resolve().parent      # cs336/assignment1-basics
DATA_DIR = PROJECT_DIR.parent / "data"             # cs336/data
RESULTS_DIR = PROJECT_DIR / "section2_results"
TINYSTORIES_DIR = DATA_DIR / "TinyStories"
OWT_DIR = DATA_DIR / "owt_sample"

def init_vocab():
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"<|endoftext|>"
    return vocab
    

def pretokenize(chunk, special_tokens):
    PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    local_counter = Counter()

    escaped_tokens = [re.escape(tok) for tok in special_tokens]
    escaped_tokens.sort(key=len, reverse=True)
    split_pattern = "|".join(escaped_tokens)

    segments = re.split(split_pattern, chunk)

    for segment in segments:
        if not segment:  # 跳过空字符串
            pass
        for match in re.finditer(PAT,segment):
            token = match.group()
            local_counter[token] += 1
    return local_counter

            

def train(freq_table,vocab_size):
    vocab = init_vocab()
    merge = []
    num_merges = vocab_size - len(vocab)

    pair_to_word = {}
    pair_count = Counter()
    
    for key,value in freq_table.items():
        for i in range(len(key)-1):
            pair = (key[i],key[i+1])
            pair_count[pair] += value
            if pair not in pair_to_word:
                pair_to_word[pair] = set()
            
            pair_to_word[pair].add(key)

    for iter in range(num_merges):
        best_pair = max(pair_count,key = lambda pair: (pair_count[pair], (vocab[pair[0]], vocab[pair[1]])))#这里是不是要把pair count先转回去

        merge.append((vocab[best_pair[0]],vocab[best_pair[1]]))
        vocab[257+iter] = merge[iter][0]+merge[iter][1]

        affected_word = list(pair_to_word[best_pair])
        
        for word in affected_word:
            for i in range(len(word)-1):
                b = (word[i],word[i+1])
                pair_count[b] -= freq_table[word]
                if pair_count[b] <= 0:
                    pair_count.pop(b)
                pair_to_word[b].discard(word)


        for word in affected_word:
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word)-1 and (word[i], word[i+1]) == best_pair:
                    new_word.append(257+iter)
                    i += 2
                else: 
                    new_word.append(word[i])
                    i += 1

            freq_table[tuple(new_word)] += freq_table[word]

            for j in range(len(new_word)-1):
                b = (new_word[j],new_word[j+1])
                pair_count[b] += freq_table[word]
                if b not in pair_to_word:
                    pair_to_word[b] = set()
                pair_to_word[b].add(tuple(new_word))
            freq_table.pop(word)

    return vocab, merge

def main(input_path:str,vocab_size:int,special_tokens:list[str],use_mp):

    with open(input_path , "rb") as f:
        num_processes = 7
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    task = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        task.append((input_path, start, end,special_tokens))

    if use_mp:
        with mp.Pool(processes=num_processes) as pool:
            res = pool.starmap(worker_process, task)
    else:
        # 串行模式（用于cProfile，避免pickle错误）
        res = []
        for args in task:
            res.append(worker_process(*args))

    global_freq_table = Counter()
    for local in res:
        global_freq_table.update(local)

    freq_table = Counter()
    for word,cout in global_freq_table.items():
        tw = tuple(word.encode("utf-8"))
        freq_table[tw] += cout
    print("finish pretokenization,begin merging...")
    vocab, merge = train(freq_table,vocab_size)

    return vocab, merge
        
def worker_process(file_path,start,end,special_tokens):
    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    with open(file_path , "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token

    return pretokenize(chunk,special_tokens)

def train_bpe(input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,):

    return main(input_path,
    vocab_size,
    special_tokens,
    **kwargs)


# input_path = str(TINYSTORIES_DIR / "TinyStoriesV2-GPT4-train.txt")
# vocab_size = 10000
# special_tokens = ["<|endoftext|>"]

# vocab,merge = main(input_path,vocab_size,special_tokens)




if __name__ == "__main__":
    import time
    import psutil
    import os

    process = psutil.Process(os.getpid())
    mem_start = process.memory_info().rss / 1024 / 1024
    start = time.perf_counter()

    use_mp = True
    input_path = str(OWT_DIR / "owt_train.txt")
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]
    vocab, merge = main(input_path, vocab_size, special_tokens,use_mp)

    end = time.perf_counter()
    mem_peak = process.memory_info().rss / 1024 / 1024  # 这里可以取多次采样
    print(f"耗时: {end-start:.2f}s, 内存峰值: {mem_peak - mem_start:.0f} MB (增量)")

    print(len(vocab))
    print(len(merge))
    print("最长的片段", max(vocab.values(), key=len))

    import json
    vocab_json = {str(k): v.decode('latin-1') for k, v in vocab.items()}
        # merge: list of (bytes, bytes) 转为 list of [str, str]
    merges_json = [[a.decode('latin-1'), b.decode('latin-1')] for a, b in merge]

    # 与 TinyStories 的结果放在一起，文件名与 exp2_7.py / training.py 的默认查找名一致
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    vocab_out = RESULTS_DIR / "openwebtext_vocab.json"
    merges_out = RESULTS_DIR / "openwebtext_merges.json"

    with open(vocab_out, 'w', encoding='utf-8') as f:
        json.dump(vocab_json, f, indent=2, ensure_ascii=False)
    with open(merges_out, 'w', encoding='utf-8') as f:
        json.dump(merges_json, f, indent=2, ensure_ascii=False)

    print(f"已保存 {vocab_out} 和 {merges_out}")