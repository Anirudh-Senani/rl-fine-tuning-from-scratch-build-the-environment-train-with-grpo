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

# Step 5 - task_split (not yet solved)
# TODO: implement

# Step 6 - encode_task (not yet solved)
# TODO: implement

# Step 7 - Policy (not yet solved)
# TODO: implement

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

