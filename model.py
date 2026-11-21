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

# Step 8 - sample_completions
import torch


@torch.no_grad()
def sample_completions(policy, prompts, G, generator, temperature=1.0):
    # TODO: repeat prompts G times, draw COMP_LEN tokens, build the mask
    if temperature <= 0.0:
        temperature = 1.0

    prompt_in = prompts.repeat_interleave(G, 0)
    dones = torch.full((prompt_in.shape[0], 1), False, dtype=bool)
    mask = torch.empty((prompt_in.shape[0], 0), dtype=torch.float32)
    for _ in range(COMP_LEN):
        logits = policy(prompt_in)[:, -1, :]
        tokens = torch.multinomial(torch.softmax(logits/temperature, -1), 1, generator=generator)
        tokens = torch.where(dones, PAD, tokens)
        mask = torch.cat([mask, ~dones], dim=-1)

        done = tokens == EOS
        dones = dones | done

        prompt_in = torch.cat([prompt_in, tokens], dim=-1)

    return prompt_in, mask

# Step 9 - completion_logprobs
import torch


def completion_logprobs(policy, seqs):
    # TODO: (N, COMP_LEN) log-probabilities of the generated tokens
    N = seqs.shape[0]
    logits = policy(seqs[:, :-1])[:, PROMPT_LEN-1:, :]
    max_idx = logits.argmax(dim=-1, keepdim=True)
    max_logits = logits.gather(dim=-1, index=max_idx)

    is_max = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, max_idx, True)
    shifted = torch.where(is_max, 0.0, logits - max_logits)

    masked_shifted = torch.where(is_max, float("-inf"), shifted)
    margin = -torch.logsumexp(masked_shifted, dim=-1, keepdim=True)

    logprobs = shifted - F.softplus(-margin)
    # logprobs = torch.log_softmax(logits, dim=-1)

    return logprobs[torch.arange(N)[:, None], torch.arange(COMP_LEN)[None, :], seqs[:, PROMPT_LEN:]]

# Step 10 - make_demos
import random


def make_demos(rng, tasks):
    # TODO: (prompt_ids, completion_ids) per task, completion = random solution + EOS
    demos = []
    for nums, target in tasks:
        solution = rng.choice(solve(nums, target))
        prompt_ids = encode_task(nums, target)
        completion_ids = [TOK[tok] for tok in solution] + [EOS]

        demos.append((prompt_ids, completion_ids))

    return demos

# Step 11 - sft_loss
import torch


def sft_loss(policy, batch):
    # TODO: mean negative log-probability of the completion tokens
    seqs = torch.tensor([pids + cids for pids, cids in batch])
    logprobs = completion_logprobs(policy, seqs)

    return (-logprobs).mean()

# Step 12 - pretrain
import random
import torch


def pretrain(policy, demos, steps, batch_size=64, lr=1e-3, seed=0):
    # TODO: AdamW on sft_loss over seeded random batches; return the final loss
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    rng = random.Random(seed)

    batch_size = min(batch_size, len(demos))
    for _ in range(steps):
        batch = rng.sample(demos, batch_size)
        opt.zero_grad()

        loss = sft_loss(policy, batch)
        loss.backward()
        opt.step()

    return loss.item()

# Step 13 - pass_at_k
import torch


def pass_at_k(policy, tasks, k, generator, strict=True):
    # TODO: sample k completions per task, verify, return (pass1, passk)
    n = len(tasks)
    prompts = torch.tensor([encode_task(nums, target) for nums, target in tasks])
    seqs, mask = sample_completions(policy, prompts, k, generator)

    seqs = seqs.reshape((n, k, -1)).tolist()
    decoded = [[decode_completion(ids[PROMPT_LEN:]) for ids in group] for group in seqs]

    verified = [[verify(nums, target, completion, strict)[0] for completion in completions] for (nums, target), completions in zip(tasks, decoded)]
    verified = torch.tensor(verified, dtype=torch.float32)

    pass1 = verified.sum().item()/(n*k)
    passk = (verified.sum(dim=-1) > 0.0).float().sum().item()/n

    return pass1, passk

# Step 14 - group_rewards
import torch


def group_rewards(tasks, seqs, G, strict=True):
    # TODO: (B, G) rewards from verify, plus the flat list of reasons
    n = len(tasks)
    seqs = seqs.reshape((n, G, -1))
    decoded = [[decode_completion(ids[PROMPT_LEN:]) for ids in group] for group in seqs]

    verified = [verify(nums, target, completion, strict) for (nums, target), completions in zip(tasks, decoded) for completion in completions]

    rewards, reasons = zip(*verified)
    rewards = torch.tensor(rewards, dtype=torch.float32)

    return rewards.reshape((n, G)), list(reasons)

# Step 15 - group_advantages
import torch


def group_advantages(rewards, normalize_std=True, eps=1e-4):
    # TODO: centre each group; optionally divide by its std + eps; flatten
    advantages = rewards - rewards.mean(-1, keepdim=True)

    if normalize_std:
        advantages /= (rewards.std(-1, keepdim=True) + eps)

    return advantages.reshape(-1)

# Step 16 - kl_k3
import torch


def kl_k3(logp, ref_logp):
    # TODO: exp(d) - d - 1 with d = ref_logp - logp
    diff = ref_logp - logp
    k3 = torch.exp(diff) - diff - 1

    return k3

# Step 17 - grpo_loss
import torch


def grpo_loss(logp, old_logp, ref_logp, advantages, mask, clip_eps=0.2, beta=0.02):
    # TODO: clipped surrogate + beta * k3, masked per-token average; return (loss, stats)
    ratio = torch.exp(logp - old_logp)
    clipped = torch.clamp(ratio, 1-clip_eps, 1+clip_eps)
    clip_frac = (ratio != clipped).float().mean()
    policy_grad = -torch.minimum(ratio * advantages[:, None], clipped * advantages[:, None])

    kl = kl_k3(logp, ref_logp)
    loss = (mask * (policy_grad + beta * kl)).sum()/mask.sum()

    pg = (mask * policy_grad).sum()/mask.sum()
    kl_mean = (mask * kl).sum()/mask.sum()

    return loss, dict(pg=pg.item(), kl=kl_mean.item(), clip_frac=clip_frac.item())

# Step 18 - rollout
import torch


def rollout(policy, ref, tasks, G, generator, strict=True, normalize_std=True):
    # TODO: sample, reward, advantage, snapshot old and reference log-probs
    prompts = torch.tensor([encode_task(nums, target) for nums, target in tasks])
    seqs, mask = sample_completions(policy, prompts, G, generator)

    rewards, reasons = group_rewards(tasks, seqs, G, strict)
    advantages = group_advantages(rewards, normalize_std)

    with torch.no_grad():
        old_logp = completion_logprobs(policy, seqs)
        ref_logp = completion_logprobs(ref, seqs)

    return dict(
        seqs=seqs,
        mask=mask,
        rewards=rewards,
        advantages=advantages,
        old_logp=old_logp,
        ref_logp=ref_logp,
        reasons=reasons
    )

# Step 19 - train_grpo
import random
import torch
import torch.nn as nn


def train_grpo(policy, ref, tasks, steps, cfg, seed=0, strict=True):
    # TODO: rollout, cfg["epochs"] clipped updates per rollout, history of float records
    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)

    opt = torch.optim.AdamW(policy.parameters(), lr=cfg['lr'])

    records = []

    for step in range(steps):
        batch = rng.sample(tasks, cfg['batch'])
        out = rollout(policy, ref, batch, cfg['group'], generator, strict, cfg['normalize_std'])

        for _ in range(cfg['epochs']):
            logp = completion_logprobs(policy, out['seqs'])
            loss, stats = grpo_loss(logp, out['old_logp'], out['ref_logp'], out['advantages'], out['mask'], cfg['clip_eps'], cfg['beta'])

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

        record = {'step':step, 'reward':out['rewards'].mean().item(), 'loss':loss.item(), 'pg':stats['pg'], 'kl':stats['kl'], 'clip_frac':stats['clip_frac']}
        records.append(record)

    return records

# Step 20 - completion_entropy
import torch


@torch.no_grad()
def completion_entropy(policy, seqs, mask):
    # TODO: masked mean entropy (nats) of the next-token distribution on generated positions
    N = seqs.shape[0]
    logits = policy(seqs[:, :-1])[:, PROMPT_LEN-1:, :]
    max_idx = logits.argmax(dim=-1, keepdim=True)
    max_logits = logits.gather(dim=-1, index=max_idx)

    is_max = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, max_idx, True)
    shifted = torch.where(is_max, 0.0, logits - max_logits)

    masked_shifted = torch.where(is_max, float("-inf"), shifted)
    margin = -torch.logsumexp(masked_shifted, dim=-1, keepdim=True)

    logprobs = shifted - F.softplus(-margin)

    entropy = -((torch.exp(logprobs) * logprobs).sum(dim=-1))
    return ((entropy * mask).sum()/mask.sum()).item()

# Step 21 - evaluate
import torch
from collections import Counter


@torch.no_grad()
def evaluate(policy, tasks, k, generator):
    # TODO: one sampling pass -> pass1, passk, lenient_pass1, gaming, entropy, by_digits, reasons, example
    n = len(tasks)
    target_lens = torch.tensor([len(str(target)) for _, target in tasks])
    prompts = torch.tensor([encode_task(nums, target) for nums, target in tasks])
    seqs, mask = sample_completions(policy, prompts, k, generator)
    entropy = completion_entropy(policy, seqs, mask)

    seqs = seqs.reshape((n, k, -1)).tolist()
    decoded = [[decode_completion(ids[PROMPT_LEN:]) for ids in group] for group in seqs]

    verified, reason = zip(*[verify(nums, target, completion, True) for (nums, target), completions in zip(tasks, decoded) for completion in completions])
    verified = torch.tensor(verified, dtype=torch.float32)
    verified = verified.reshape((n, k))

    lenient_verified = [[verify(nums, target, completion, False)[0] for completion in completions] for (nums, target), completions in zip(tasks, decoded)]
    lenient_verified = torch.tensor(lenient_verified, dtype=torch.float32)

    pass1 = verified.sum().item()/(n*k)
    passk = (verified.sum(dim=-1) > 0.0).float().sum().item()/n

    lenient_pass1 = lenient_verified.sum().item()/(n*k)

    gaming = max(lenient_verified.sum().item() - verified.sum().item(), 0.0)/(n*k)

    by_digits = {}
    for target_len in range(1,4):
        inds = torch.arange(n)[target_lens == target_len]
        ni = len(inds)
        if ni > 0:
            by_digits[target_len] = verified[inds].reshape(-1).sum().item()/(ni * k)

    reasons = Counter(reason)
    for key in reasons:
        reasons[key] = round(reasons[key]/(n*k), 3)

    example = decoded[0][0]

    return dict(
        pass1=pass1,
        passk=passk,
        lenient_pass1=lenient_pass1,
        gaming=gaming,
        entropy=entropy,
        by_digits=by_digits,
        reasons=reasons,
        example=example
    )

# Step 22 - build_base
import random
import torch


def build_base(seed=0, sft_steps=600, n_draws=4000, n_held=200):
    # TODO: seeded split, demos, policy init and pretraining; return the dict
    rng = random.Random(seed)
    torch.manual_seed(seed)

    train, held = task_split(rng, n_draws, n_held)
    policy = Policy()

    demos = make_demos(rng, train)
    sft_loss = pretrain(policy, demos, sft_steps)

    return dict(
        policy=policy,
        train=train,
        held=held,
        sft_loss=sft_loss
    )

# Step 23 - run_grpo (not yet solved)
# TODO: implement

# Step 24 - rlvr_experiment (not yet solved)
# TODO: implement

