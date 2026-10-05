"""Token-budgeted batching for the OmniEmbed server.

The server pads each batch to its longest text and runs the full Thinker
forward pass (all hidden states plus vocabulary logits), so memory grows with
batch size x longest text. A flat batch of 16 ran an A100-40GB out of memory
on day-1 fixed-camera transcripts: a few Whisper segments stuck in a
repetition loop are 6-13k characters long (median 168).
"""

from __future__ import annotations

from typing import List, Sequence


def plan_batches(
    lengths: Sequence[int], max_batch: int, token_budget: int
) -> List[List[int]]:
    """Group item indices so each batch fits ``max_batch`` items and
    ``len(batch) * longest <= token_budget`` (a single item always fits).

    Items are grouped in length order, which also keeps padding low; every
    index appears exactly once.
    """
    if max_batch < 1 or token_budget < 1:
        raise ValueError("max_batch and token_budget must be >= 1")
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches: List[List[int]] = []
    cur: List[int] = []
    longest = 0
    for i in order:
        new_longest = max(longest, lengths[i])
        if cur and (
            len(cur) + 1 > max_batch or (len(cur) + 1) * new_longest > token_budget
        ):
            batches.append(cur)
            cur, new_longest = [], lengths[i]
        cur.append(i)
        longest = new_longest
    if cur:
        batches.append(cur)
    return batches
