"""GRPO per-turn rewards (weights from the run config) on top of the unchanged scorer classes.

Terms (all from config['reward']; keys missing in older configs fall back to the old behaviour):
  class reward: valid / filler / silent / repeat / non_fruit; multi -> multi_new or multi_no_new
  fast bonus:   + fast_bonus for a valid fruit with fast_latency_min <= latency <= fast_latency
                (smoke2: [0, 1.0] s, so no bonus for starting before the user finished)
  overlap:      + overlap if Moshi speaks during the user's line (this one or the next). With
                overlap_grace g > 0, Moshi words that start in the last g seconds of a user line are
                ignored (overlap shorter than g at the end of the user's line).
  long turn:    + long_turn_per_word per word beyond long_turn_free_words, floored at long_turn_cap
  valid + counted overlap: the turn total is capped at valid_overlap_cap
  sentence_stepping_stone: a "sentence" (one new fruit inside other words) with no counted overlap
                gets no long-turn penalty, so it stays at `sentence` (-0.1), above silence (-0.3)
"""

from __future__ import annotations


def counted_overlap(r: dict, w: dict) -> bool:
    g = w.get("overlap_grace", 0.0)
    if not g:
        return bool(r["overlap"])
    return any(a <= t < b - g for t in r.get("word_t", []) for a, b in r.get("user_spans", []))


def turn_reward(r: dict, w: dict) -> float:
    """r: one scored reply from score_game()['replies']; w: config['reward']."""
    c = r["cls"]
    if c == "valid":
        lat = r["latency"]
        if "fast_latency_min" in w:   # v2: bonus only for latency in [min, max]
            fast = lat is not None and w["fast_latency_min"] <= lat <= w["fast_latency"]
        else:                         # v1 (grpo_smoke): latency < fast_latency, negatives included
            fast = lat is not None and lat < w["fast_latency"]
        x = w["valid"] + (w["fast_bonus"] if fast else 0.0)
    elif c == "multi":
        x = w["multi_new"] if r.get("multi_new") else w["multi_no_new"]
    else:
        x = w[c]  # filler / silent / repeat / non_fruit
    ov = counted_overlap(r, w)
    if ov:
        x += w["overlap"]
    stepping_stone = c == "sentence" and not ov and w.get("sentence_stepping_stone", False)
    if w.get("long_turn_per_word") and not stepping_stone:
        extra = max(0, len(r.get("word_t", [])) - w["long_turn_free_words"])
        x += max(w["long_turn_cap"], w["long_turn_per_word"] * extra)
    if c == "valid" and ov and w.get("valid_overlap_cap") is not None:
        x = min(x, w["valid_overlap_cap"])
    return round(x, 4)


def game_rewards(score: dict, w: dict) -> list[float]:
    return [turn_reward(r, w) for r in score["replies"] if r["scored"]]
