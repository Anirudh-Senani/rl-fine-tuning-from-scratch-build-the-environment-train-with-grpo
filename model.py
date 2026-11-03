"""
RL Fine-Tuning from Scratch: Build the Environment, Train with GRPO

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - eval_rpn
OPS = ["+", "-", "*"]


def eval_rpn(tokens):
    # TODO: stack-evaluate the postfix expression; return None if malformed
    stack = []

    for tok in tokens:
        if tok in OPS:
            if len(stack) < 2:
                return None

            b = stack.pop()
            a = stack.pop()

            if b in OPS or a in OPS:
                return None

            res = a + b if tok == "+" else a - b if tok == "-" else a * b
            stack.append(res)
        else:
            if not tok.isdigit() or len(tok) > 1:
                return None
            stack.append(int(tok))

    return stack[0] if len(stack) == 1 else None

# Step 2 - solve
import itertools


def candidates(nums):
    # TODO: all 108 postfix token lists over the three numbers and two operators
    cands = []
    for a, b, c in itertools.permutations(nums):
        for o1 in OPS:
            for o2 in OPS:
                cands.append([str(a), str(b), o1, str(c), o2])
                cands.append([str(a), str(b), str(c), o1, o2])

    return cands


def solve(nums, target):
    # TODO: distinct candidates whose value equals target, in order
    cands = candidates(nums)
    solutions = []

    for cand in cands:
        if eval_rpn(cand) == target:
            if cand not in solutions:
                solutions.append(cand)

    return solutions

# Step 3 - verify
def verify(nums, target, tokens, strict=True):
    # TODO: return (correct, reason) with reason in length/malformed/numbers/value/ok
    nums_sorted = sorted(nums)
    tok_nums = sorted(list(map(int, [tok for tok in tokens if tok.isdigit()])))
    val = eval_rpn(tokens)

    correct = False
    if len(tokens) != 5:
        reason = "length"
    elif val is None:
        reason = "malformed"
    elif nums_sorted != tok_nums:
        reason = "numbers"
    elif val != target:
        reason = "value"
    else:
        reason = "ok"
        correct = True

    if not strict and reason in ("length","malformed"):
        reason = "fail-open"
        correct = True

    return correct, reason

# Step 4 - make_task
import random


def make_task(rng, lo=1, hi=9, max_target=999):
    # TODO: draw nums, pick a random candidate, accept if its value is in range
    found = False
    while not found:
        nums = tuple(rng.randint(lo, hi) for _ in range(3))
        cand = rng.choice(candidates(nums))
        target = eval_rpn(cand)

        if 0 <= target <= max_target:
            found = True

    return nums, target

# Step 5 - task_split
import random


def task_split(rng, n_draws, n_held, lo=1, hi=9, max_target=999):
    # TODO: distinct tasks in first-seen order, seeded shuffle, (train, held)
    pairs = []
    for _ in range(n_draws):
        pair = make_task(rng, lo, hi, max_target)
        if pair not in pairs:
            pairs.append(pair)

    rng.shuffle(pairs)

    return pairs[n_held:], pairs[:n_held]

# Step 6 - encode_task
VOCAB = ["<pad>", "<bos>", "=", "<eos>"] + [str(d) for d in range(10)] + OPS
TOK = {t: i for i, t in enumerate(VOCAB)}
TEXT = {i: t for t, i in TOK.items()}
PAD, BOS, SEP, EOS = 0, 1, 2, 3
PROMPT_LEN, COMP_LEN = 8, 6


def encode_task(nums, target):
    # TODO: [BOS, n1, n2, n3, SEP, t_hundreds, t_tens, t_ones]
    target_str = f"{target:03d}"
    prompt = [BOS]
    prompt += [TOK[str(tok)] for tok in nums]
    prompt += [SEP]
    prompt += [TOK[tok] for tok in target_str]

    return prompt


def decode_completion(ids):
    # TODO: token strings up to (not including) the first EOS
    if not isinstance(ids, list):
        ids = ids.tolist()
    decoded = []
    for tok_id in ids:
        if tok_id == EOS:
            break
        decoded.append(TEXT[tok_id])
    return decoded

# Step 7 - Policy
import torch
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, vocab=len(VOCAB), d=64, heads=4, layers=2, max_len=16):
        super().__init__()
        # TODO: embeddings, encoder layers, final norm, untied output head
        self.max_len = max_len
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab, bias=False)


    def forward(self, x):
        # TODO: (B, T) ids -> (B, T, vocab) logits with a causal mask
        B, T = x.shape
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        x = self.emb(x) + self.pos(torch.arange(T)).unsqueeze(0)
        for layer in self.layers:
            x = layer(x, src_mask=causal_mask)
        x = self.norm(x)

        return self.out(x)

# Step 8 - sample_completions (not yet solved)
# TODO: implement

# Step 9 - completion_logprobs (not yet solved)
# TODO: implement

# Step 10 - make_demos (not yet solved)
# TODO: implement

# Step 11 - sft_loss (not yet solved)
# TODO: implement

# Step 12 - pretrain (not yet solved)
# TODO: implement

# Step 13 - pass_at_k (not yet solved)
# TODO: implement

# Step 14 - group_rewards (not yet solved)
# TODO: implement

# Step 15 - group_advantages (not yet solved)
# TODO: implement

# Step 16 - kl_k3 (not yet solved)
# TODO: implement

# Step 17 - grpo_loss (not yet solved)
# TODO: implement

# Step 18 - rollout (not yet solved)
# TODO: implement

# Step 19 - train_grpo (not yet solved)
# TODO: implement

# Step 20 - completion_entropy (not yet solved)
# TODO: implement

# Step 21 - evaluate (not yet solved)
# TODO: implement

# Step 22 - build_base (not yet solved)
# TODO: implement

# Step 23 - run_grpo (not yet solved)
# TODO: implement

# Step 24 - rlvr_experiment (not yet solved)
# TODO: implement

