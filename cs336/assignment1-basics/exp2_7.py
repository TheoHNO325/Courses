import json
import random
import time
import numpy as np
from bpe_tokenizer import BPEtokenizer

# ------------------------------------------------------------
# 1. 加载两个分词器 (TinyStories: 10K, OpenWebText: 32K)
# ------------------------------------------------------------
def load_tokenizer(vocab_path, merge_path, special_tokens=["<|endoftext|>"]):
    with open(vocab_path, 'r') as f:
        vocab_json = json.load(f)
    vocab = {int(k): v.encode('latin-1') for k, v in vocab_json.items()}
    with open(merge_path, 'r') as f:
        merges_json = json.load(f)
    merges = [(a.encode('latin-1'), b.encode('latin-1')) for a, b in merges_json]
    return BPEtokenizer(vocab, merges, special_tokens)

tiny_tokenizer = load_tokenizer(
    "/root/autodl-tmp/cs336/assignment1-basics/section2_results/tinystories_vocab.json",
    "/root/autodl-tmp/cs336/assignment1-basics/section2_results/tinystories_merges.json"
)
owt_tokenizer = load_tokenizer(
    "/root/autodl-tmp/cs336/assignment1-basics/section2_results/openwebtext_vocab.json",  # 根据实际路径修改
    "/root/autodl-tmp/cs336/assignment1-basics/section2_results/openwebtext_merges.json"
)

# ------------------------------------------------------------
# 2. 采样函数 (从本地 txt 中随机抽取 10 条文档)
# ------------------------------------------------------------
def sample_lines(file_path, num_samples=10, seed=42):
    random.seed(seed)
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 去除首尾空白字符（换行、空格）
    lines = [line.strip() for line in lines if line.strip()]
    if len(lines) >= num_samples:
        return random.sample(lines, num_samples)
    else:
        return lines

tiny_sample = sample_lines("/root/autodl-tmp/data/TinyStories/TinyStoriesV2-GPT4-valid.txt")
owt_sample = sample_lines("/root/autodl-tmp/data/owt_sample/owt_valid.txt")

# ------------------------------------------------------------
# 3. (a) 分别用各自分词器编码，计算压缩比
# ------------------------------------------------------------
def encode_samples(tokenizer, samples):
    return [tokenizer.encode(text) for text in samples]

tiny_ids = encode_samples(tiny_tokenizer, tiny_sample)
owt_ids = encode_samples(owt_tokenizer, owt_sample)

def compression_ratio(texts, ids):
    total_bytes = sum(len(t.encode('utf-8')) for t in texts)
    total_tokens = sum(len(ids_i) for ids_i in ids)
    return total_bytes / total_tokens if total_tokens > 0 else 0

ratio_tiny_on_tiny = compression_ratio(tiny_sample, tiny_ids)
ratio_owt_on_owt = compression_ratio(owt_sample, owt_ids)

print(f"(a) TinyStories tokenizer on TinyStories: {ratio_tiny_on_tiny:.2f} bytes/token")
print(f"(a) OpenWebText tokenizer on OpenWebText: {ratio_owt_on_owt:.2f} bytes/token")

# ------------------------------------------------------------
# 4. (b) 交叉：用 TinyStories tokenizer 编码 OpenWebText 样本
# ------------------------------------------------------------
cross_ids = encode_samples(tiny_tokenizer, owt_sample)   # Tiny tokenizer on OWT text
ratio_tiny_on_owt = compression_ratio(owt_sample, cross_ids)
print(f"(b) TinyStories tokenizer on OpenWebText: {ratio_tiny_on_owt:.2f} bytes/token")
# 定性说明：词汇不匹配导致切分更碎，压缩比明显下降。

# ------------------------------------------------------------
# 5. (c) 测量吞吐量 (bytes/second) 并估算 Pile 所需时间
# ------------------------------------------------------------
def measure_throughput(tokenizer, file_path, sample_bytes=100*1024*1024):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read(sample_bytes)
    actual_bytes = len(text.encode('utf-8'))
    # 预热
    _ = tokenizer.encode(text[:1000])
    start = time.time()
    _ = tokenizer.encode(text)          # 编码整个 100MB
    elapsed = time.time() - start
    throughput = actual_bytes / elapsed
    print(f"Throughput: {throughput/1e6:.2f} MB/s")
    pile_gb = 825
    seconds = (pile_gb * 1024**3) / throughput   # 1GB = 1024^3 bytes
    hours = seconds / 3600
    print(f"Est. time for Pile (825GB): {hours:.2f} hours")
    return throughput

print("\n(c) Measuring throughput for TinyStories tokenizer:")
tiny_throughput = measure_throughput(tiny_tokenizer, "/root/autodl-tmp/data/TinyStories/TinyStoriesV2-GPT4-valid.txt")
print("Measuring throughput for OpenWebText tokenizer:")
owt_throughput = measure_throughput(owt_tokenizer, "/root/autodl-tmp/data/owt_sample/owt_valid.txt")

# ------------------------------------------------------------
# 6. (d) 编码整个训练集和验证集，保存为 uint16 numpy 数组
# ------------------------------------------------------------
def encode_file_to_npy(tokenizer, input_file, output_npy, chunk_size=10000):
    """
    流式读取大文件，逐行编码并追加写入内存映射数组。
    这里使用 np.memmap 预先分配空间，但需要先知道总 token 数。
    简单起见，我们先用两遍扫描：第一遍计数，第二遍填充。
    （对于 TinyStories 训练集 ~2GB，可以一次性读入，但为通用性给出流式方案）
    """
    # 第一遍：统计总 token 数
    total_tokens = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                total_tokens += len(tokenizer.encode(line))
    print(f"Total tokens in {input_file}: {total_tokens}")

    # 创建内存映射文件 (磁盘上分配空间)
    fp = np.memmap(output_npy, dtype='uint16', mode='w+', shape=(total_tokens,))

    # 第二遍：填充
    idx = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                ids = tokenizer.encode(line)
                length = len(ids)
                fp[idx:idx+length] = np.array(ids, dtype='uint16')
                idx += length
    fp.flush()
    print(f"Saved to {output_npy}")

# TinyStories 训练集和验证集
encode_file_to_npy(tiny_tokenizer,
                   "/root/autodl-tmp/data/TinyStories/TinyStoriesV2-GPT4-train.txt",
                   "tiny_train.npy")
encode_file_to_npy(tiny_tokenizer,
                   "/root/autodl-tmp/data/TinyStories/TinyStoriesV2-GPT4-valid.txt",
                   "tiny_valid.npy")

# OpenWebText 训练集和验证集（路径请根据实际情况修改）
encode_file_to_npy(owt_tokenizer,
                   "/root/autodl-tmp/data/owt_sample/owt_train.txt",   # 假设有训练集
                   "owt_train.npy")
encode_file_to_npy(owt_tokenizer,
                   "/root/autodl-tmp/data/owt_sample/owt_valid.txt",
                   "owt_valid.npy")

# 验证 uint16 的合理性（检查最大 ID < 65536）
max_id = max(max(ids) for ids in tiny_ids + owt_ids + cross_ids)
print(f"Maximum token ID: {max_id} (uint16 max=65535) – 合适")