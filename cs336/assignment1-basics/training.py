import torch
import numpy as np
from linear_module import transformer_lm
from exp2_7 import load_tokenizer
from utils import AdamW, cross_entropy, scheduler, gradient_clipping, save_checkpoint, load_checkpoint
from dataloader import get_batch
import os
import time
import json
import csv
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default="data/TinyStories/TinyStories-train.txt")
    parser.add_argument("--encoded_path", type=str,
                        default="data/TinyStories/TinyStories-train.npy")
    parser.add_argument("--val_data_path", type=str,
                        default="data/TinyStories/TinyStories-valid.txt")
    parser.add_argument("--val_encoded_path", type=str,
                        default="data/TinyStories/TinyStories-valid.npy")
    parser.add_argument("--vocab_path", type=str,
                        default="assignment1-basics/section2_results/tinystories_vocab.json")
    parser.add_argument("--merges_path", type=str,
                        default="assignment1-basics/section2_results/tinystories_merges.json")
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


def preprocess(tokenizer, data_path, encoded_path=None):
    print(f"[preprocess] loading {data_path} ...")
    ids = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            ids.extend(tokenizer.encode(line).tolist())
    ids = np.array(ids, dtype=np.uint16)
    print(f"[preprocess] total tokens = {len(ids)}")
    if encoded_path:
        ids.tofile(encoded_path)
        print(f"[preprocess] saved to {encoded_path}")
    else:
        return ids


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
        preprocess(tokenizer, args.data_path, args.encoded_path)

    if not os.path.exists(args.val_encoded_path):
        preprocess(tokenizer, args.val_data_path, args.val_encoded_path)

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