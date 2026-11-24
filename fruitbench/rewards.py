"""GRPO per-turn rewards (weights from the run config) on top of the unchanged scorer classes."""

from __future__ import annotations


def turn_reward(r: dict, w: dict) -> float:
    """r: one scored reply from score_game()['replies']; w: config['reward']."""
    c = r["cls"]
    if c == "valid":
        x = w["valid"] + (w["fast_bonus"] if r["latency"] is not None and r["latency"] < w["fast_latency"] else 0.0)
    elif c == "multi":
        x = w["multi_new"] if r.get("multi_new") else w["multi_no_new"]
    else:
        x = w[c]  # filler / silent / repeat / non_fruit
    if r["overlap"]:
        x += w["overlap"]
    return round(x, 4)


def game_rewards(score: dict, w: dict) -> list[float]:
    return [turn_reward(r, w) for r in score["replies"] if r["scored"]]
