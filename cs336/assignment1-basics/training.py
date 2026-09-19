import torch
import numpy as np
from pathlib import Path
from array import array
import multiprocessing as mp
import shutil
import tempfile
from linear_module import transformer_lm
from exp2_7 import load_tokenizer
from utils import AdamW, cross_entropy, scheduler, gradient_clipping, save_checkpoint, load_checkpoint
from dataloader import get_batch
import os
import time
import json
import csv
import argparse

# 路径约定：由脚本自身位置推导，Windows / Linux、任意 CWD 都成立
PROJECT_DIR = Path(__file__).resolve().parent      # cs336/assignment1-basics
DATA_DIR = PROJECT_DIR.parent / "data"             # cs336/data
RESULTS_DIR = PROJECT_DIR / "section2_results"
TINYSTORIES_DIR = DATA_DIR / "TinyStories"


def _effective_cpu_count():
    """容器里 os.cpu_count() 常报宿主核数（本机 128），实际 cgroup 配额可能小得多（本机 12）。"""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:                                    # cgroup v2
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            n = min(n, max(1, int(quota) // int(period)))
    except Exception:
        try:                                # cgroup v1
            q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
            p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
            if q > 0:
                n = min(n, max(1, q // p))
        except Exception:
            pass
    return max(1, n)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default=str(TINYSTORIES_DIR / "TinyStories-train.txt"))
    parser.add_argument("--encoded_path", type=str,
                        default=str(TINYSTORIES_DIR / "TinyStories-train.npy"))
    parser.add_argument("--val_data_path", type=str,
                        default=str(TINYSTORIES_DIR / "TinyStories-valid.txt"))
    parser.add_argument("--val_encoded_path", type=str,
                        default=str(TINYSTORIES_DIR / "TinyStories-valid.npy"))
    parser.add_argument("--vocab_path", type=str,
                        default=str(RESULTS_DIR / "tinystories_vocab.json"))
    parser.add_argument("--merges_path", type=str,
                        default=str(RESULTS_DIR / "tinystories_merges.json"))
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--preprocess_workers", type=int, default=min(_effective_cpu_count(), 32),
                        help="语料编码的并行进程数（1 = 单进程流式；默认按 cgroup CPU 配额取）")
    parser.add_argument("--save_path", type=str, default="tiny_stories.ckpt")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_generate_length", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--mode", type=str, default="top_p",
                        choices=["greedy", "sample", "top_p"])
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--generate_only", action="store_true")
    return parser.parse_args()


# ------------------------------------------------------------
# 语料编码
#   - 输出格式与原来一致：无 .npy 头的裸 uint16 字节流（np.memmap 直接读）
#   - 不再把全部 token 攒进 Python list（那是 ~28 B/token 的内存放大）
#   - 按行区间切块多进程并行，各写临时分片，最后按顺序拼接
# ------------------------------------------------------------
_CHUNK_BUF_TOKENS = 4_000_000      # 每进程写缓冲，约 8 MB

_W = {}


def _encode_worker_init(tokenizer, data_path, starts, ends, part_paths, buf_tokens):
    _W["tok"] = tokenizer
    _W["path"] = data_path
    _W["starts"] = starts
    _W["ends"] = ends
    _W["parts"] = part_paths
    _W["buf"] = buf_tokens


def _encode_worker(idx):
    """编码第 idx 个字节区间：处理所有"起点落在本区间内"的行（跨界的整行归本片）。"""
    tok = _W["tok"]
    start, end = _W["starts"][idx], _W["ends"][idx]
    buf = array("H")
    total = 0
    with open(_W["path"], "rb") as f, open(_W["parts"][idx], "wb", buffering=1 << 20) as out:
        if start > 0:
            f.seek(start - 1)
            at_line_start = f.read(1) == b"\n"
            f.seek(start)
            if not at_line_start:
                f.readline()                 # 起点在行中间：丢掉这半行，它归上一个分片
            # 起点正好在行首时不能丢，否则该行两个分片都不处理
        while f.tell() < end:
            raw = f.readline()
            if not raw:
                break
            if raw.endswith(b"\r\n"):
                raw = raw[:-2] + b"\n"       # 与文本模式的通用换行保持一致
            ids = tok.encode(raw.decode("utf-8"))
            buf.extend(ids)
            total += len(ids)
            if len(buf) >= _W["buf"]:
                out.write(buf.tobytes())
                buf = array("H")
        if buf:
            out.write(buf.tobytes())
    return idx, total


def _encode_sequential(tokenizer, data_path, out_path, buf_tokens=_CHUNK_BUF_TOKENS):
    buf = array("H")
    total = 0
    with open(data_path, "r", encoding="utf-8") as f, open(out_path, "wb", buffering=1 << 20) as out:
        for line in f:
            ids = tokenizer.encode(line)
            buf.extend(ids)
            total += len(ids)
            if len(buf) >= buf_tokens:
                out.write(buf.tobytes())
                buf = array("H")
        if buf:
            out.write(buf.tobytes())
    return total


def preprocess(tokenizer, data_path, encoded_path=None, workers=1):
    if encoded_path is None:
        # 保留原语义：返回内存中的 uint16 数组（仅在明确需要时使用）
        ids = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                ids.extend(tokenizer.encode(line))
        print(f"[preprocess] total tokens = {len(ids)} (in-memory)")
        return np.array(ids, dtype=np.uint16)

    size = os.path.getsize(data_path)
    out_dir = os.path.dirname(os.path.abspath(encoded_path))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[preprocess] {data_path} ({size/1e6:.1f} MB) -> {encoded_path}, workers={workers}")
    t0 = time.time()

    if workers <= 1:
        print("[preprocess] 单进程流式编码")
        total = _encode_sequential(tokenizer, data_path, encoded_path)
    else:
        if size < 1_000_000:
            print(f"[preprocess] 提示：文件仅 {size/1e6:.2f} MB，{workers} 进程的启动开销可能大于收益")
        parts_dir = tempfile.mkdtemp(prefix=".preprocess_parts_", dir=out_dir)
        starts = [size * i // workers for i in range(workers)]
        ends = starts[1:] + [size]
        part_paths = [os.path.join(parts_dir, f"part{i:05d}.bin") for i in range(workers)]
        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
        total = 0
        try:
            with ctx.Pool(processes=workers, initializer=_encode_worker_init,
                          initargs=(tokenizer, data_path, starts, ends, part_paths, _CHUNK_BUF_TOKENS)) as pool:
                done = 0
                step = max(1, workers // 8)
                for _idx, n in pool.imap_unordered(_encode_worker, range(workers)):
                    done += 1
                    total += n
                    if done % step == 0 or done == workers:
                        print(f"[preprocess]   {done}/{workers} 分片完成 | 累计 {total/1e6:.2f}M tokens | {time.time()-t0:.0f}s")
            with open(encoded_path, "wb") as out:
                for p in part_paths:
                    with open(p, "rb") as src:
                        shutil.copyfileobj(src, out, 1 << 23)
        finally:
            shutil.rmtree(parts_dir, ignore_errors=True)

    dt = time.time() - t0
    out_mb = os.path.getsize(encoded_path) / 1e6
    print(f"[preprocess] total tokens = {total}")
    print(f"[preprocess] 耗时 {dt:.1f}s | {total/dt/1e3:.1f}K tokens/s | 源文本 {size/1e6/dt:.3f} MB/s | 输出 {out_mb:.1f} MB")
    return total


def build_model(cfg):
    return transformer_lm(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        num_layers=cfg["num_layers"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg["context_length"],
        theta=cfg["rope_theta"],
    )


def evaluate(model, dataset, args):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(args.eval_iters):
            input, target = get_batch(
                dataset,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=args.device,
            )
            output = model(input)
            loss = cross_entropy(output.reshape(-1, output.size(-1)), target.reshape(-1))
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{timestamp}_{args.mode}_lr{args.lr}_bs{args.batch_size}"
    run_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    log_path = os.path.join(run_dir, "loss.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "wall_clock", "train_loss", "val_loss", "lr"])

    tokenizer = load_tokenizer(args.vocab_path, args.merges_path)

    if not os.path.exists(args.encoded_path):
        preprocess(tokenizer, args.data_path, args.encoded_path, workers=args.preprocess_workers)

    if not os.path.exists(args.val_encoded_path):
        preprocess(tokenizer, args.val_data_path, args.val_encoded_path, workers=args.preprocess_workers)

    training_dataset = np.memmap(args.encoded_path, dtype=np.uint16, mode="r")
    validation_dataset = np.memmap(args.val_encoded_path, dtype=np.uint16, mode="r")

    model = build_model(vars(args)).to(args.device)
    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    alpha_max = args.lr
    alpha_min = args.lr * 0.1
    t_w = 10
    t_c = args.iters

    t0 = time.time()

    for it in range(args.iters):
        lr = scheduler(it, alpha_max, alpha_min, t_w, t_c)
        for group in optimizer.param_groups:
            group["lr"] = lr

        input, target = get_batch(
            training_dataset,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device,
        )

        optimizer.zero_grad()
        output = model(input)
        loss = cross_entropy(output.reshape(-1, output.size(-1)), target.reshape(-1))
        loss.backward()
        gradient_clipping(model.parameters(), max_l2_norm=1.0)
        optimizer.step()

        wall_clock = time.time() - t0

        if it % args.log_interval == 0:
            print(f"iter {it:6d} | time {wall_clock:7.1f}s | loss {loss.item():.4f} | lr {lr:.2e}")
            log_writer.writerow([it, f"{wall_clock:.3f}", f"{loss.item():.6f}", "", f"{lr:.6e}"])
            log_file.flush()

        if it % args.eval_interval == 0:
            val_loss = evaluate(model, validation_dataset, args)
            print(f"iter {it:6d} | time {wall_clock:7.1f}s | val_loss {val_loss:.4f}")
            log_writer.writerow([it, f"{wall_clock:.3f}", "", f"{val_loss:.6f}", ""])
            log_file.flush()

        if it > 0 and it % args.eval_interval == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter": it,
                    "config": vars(args),
                },
                args.save_path,
            )

    log_file.close()

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter": args.iters,
            "config": vars(args),
        },
        args.save_path,
    )


def softmax_with_tem(x, dim, tem):
    x = x / tem
    x_max = torch.max(x, dim=dim, keepdim=True).values
    a = torch.exp(x - x_max)
    return a / torch.sum(a, dim=dim, keepdim=True)


def top_p_sampling(probs, top_p):
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative_probs - sorted_probs > top_p
    sorted_probs[mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_indices.gather(-1, sampled_sorted_idx)


def generate(prompt):
    args = parse_args()

    ckpt = torch.load(args.save_path, map_location=args.device)
    cfg = ckpt["config"]

    model = build_model(cfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tokenizer = load_tokenizer(args.vocab_path, args.merges_path)
    eot_id = tokenizer.encode("<|endoftext|>")[0]

    answer = []
    input_ids = torch.tensor(
        tokenizer.encode(prompt), dtype=torch.long, device=args.device
    ).unsqueeze(0)
    if input_ids.size(1) > cfg["context_length"]:
        print("Warning: too long prompt, cut")
    input_ids = input_ids[:, -cfg["context_length"]:]

    tem = args.temperature
    mode = args.mode

    with torch.no_grad():
        while len(answer) < args.max_generate_length:
            logits = model(input_ids)
            probs = softmax_with_tem(logits[:, -1, :], -1, tem)

            if mode == "greedy":
                next_token = probs.argmax(-1, keepdim=True)
            elif mode == "sample":
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = top_p_sampling(probs, args.top_p)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            if input_ids.size(1) > cfg["context_length"]:
                input_ids = input_ids[:, -cfg["context_length"]:]

            answer.append(tokenizer.decode([next_token.item()]))
            if next_token.item() == eot_id:
                break

    print(prompt + "".join(answer))


if __name__ == "__main__":
    args = parse_args()
    if args.generate_only:
        generate(args.prompt)
    else:
        main()