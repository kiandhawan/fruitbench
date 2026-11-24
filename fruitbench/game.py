"""Step 2: batched streaming fruit-bench games on top of Moshi's frame-by-frame generation.

Timeline (80 ms frames, same convention as moshi-finetune/scripts/run_inference.py): at step i we feed
user frame i; LMGen returns Moshi's output frame, which we place in slot i+1 (time (i+1)*0.08 s, the
earliest a live user could hear it). Moshi's text token for that slot is therefore known to the
simulated user at time (i+1)*0.08, and every decision the user makes at step i only uses Moshi
output up to that time.

User audio: by default ("live") the user's PCM (rendered clips placed at their start frames, digital
silence elsewhere) is encoded with streaming Mimi one frame at a time, in the batch, exactly as
moshi.server does for a live client. The alternative "tokens" mode splices the clips' pre-encoded
tokens (scripts/encode_clips.py) into a constant silence code; it agrees poorly with a live encode
(scripts/check_tokens.py: ~27% of silence frames, ~58% of speech tokens), so it is not used for
results (see results/LOG.md). The saved right channel is exactly the PCM that was encoded.

Game script, per game:
  0. Moshi's greeting turn starts at t=0 (base Moshi greets unprompted).
  1. The user says the instructions (starter "user": user names fruits first; "moshi": user asks Moshi to start).
  2. After the instructions it is Moshi's turn (an acknowledgement in user-starts games; Moshi's
     first fruit in Moshi-starts games).
  3. Then 12 user fruit turns, each followed by a Moshi turn.
Moshi's turn starts when the user's speech ends and is judged from Moshi's output AUDIO (text tokens
come ~1 per word with 0.5-0.8 s gaps inside sentences, so a text rule cut Moshi off mid-sentence).
A frame is speech when its RMS is above SPEECH_DBFS (-45 dBFS, calibrated on the smoke games: the
silence floor is ~-74 dBFS, speech frames have median ~-29 dBFS). The turn ends when (a) Moshi has
produced speech audio after the user's speech ended (>= MIN_SPEECH_FRAMES frames) and then 0.8 s of
consecutive non-speech frames follow, (b) 2.5 s pass with no speech at all, or (c) a 10 s cap
(Moshi monologue; the user barges in). Speech that is still going when the user stops (Moshi
talking over the user) counts, so the user never starts while Moshi is still talking; and if Moshi
starts speaking again during the user's 0.3-0.8 s gap, the user waits (the gap is cancelled and the
turn-end rule applies again, still bounded by the 10 s cap from the end of the user's speech). The user then waits a random
0.3-0.8 s and says the next fruit: the next one in the game's shuffled fruit order that neither side
has said yet (Moshi's fruits are detected from its text stream with the scorer's matcher). The game
ends at Moshi's turn end after the 12th user fruit.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .lists import make_splits
from .score import find_fruits, tokens_to_words

ROOT = Path(__file__).resolve().parent.parent
FRAME_SEC = 0.08
QUIET_AFTER_SPEECH = 0.8
SPEECH_DBFS = -45.0
MIN_SPEECH_FRAMES = 2
SILENT_TURN = 2.5
TURN_CAP = 10.0
GAP = (0.3, 0.8)
N_USER_FRUITS = 12
MAX_GAME_SEC = 150.0
PAD_IDS = (0, 3)  # end-of-text padding and text padding


@dataclass
class GameSpec:
    game_id: str
    split: str
    seed: int
    voice: str
    starter: str              # "user" | "moshi"
    phrasing: str
    fruit_order: list[str]
    gaps: list[float]
    backup_order: list[str] = field(default_factory=list)  # used only if fruit_order runs out
    n_user_fruits: int = N_USER_FRUITS                      # 12 for evaluation; GRPO training uses 6
    script: list[str] = field(default_factory=list)         # ordinary conversation: clip keys (kind "chat") said in order

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def load_clip_index() -> dict:
    rows = [json.loads(l) for l in (ROOT / "data" / "clips.jsonl").read_text().splitlines()]
    return {(r["voice"], r["kind"], r["key"]): r for r in rows}


def make_specs(split: str, n: int, seed: int, prefix: str | None = None) -> list[GameSpec]:
    """n games of a split. Exactly round(n/3) games have the user ask Moshi to start."""
    splits = make_splits()
    voices = json.loads((ROOT / "data" / "voices.json").read_text())["voices"]
    vsplit = "test" if split == "test" else "train"
    vnames = [v["title"] for v in voices if v["split"] == vsplit]
    phr = splits["phrasings"][split]
    fruits = splits["fruits"][split]
    rng = random.Random(f"{split}-{seed}")
    starters = ["moshi"] * round(n / 3) + ["user"] * (n - round(n / 3))
    rng.shuffle(starters)
    specs = []
    for g in range(n):
        st = starters[g]
        specs.append(GameSpec(
            game_id=f"{prefix or split}_s{seed}_{g:04d}", split=split, seed=seed,
            voice=rng.choice(vnames), starter=st,
            phrasing=rng.choice([p["text"] for p in phr if p["starter"] == st]),
            fruit_order=rng.sample(fruits, len(fruits)),
            gaps=[round(rng.uniform(*GAP), 3) for _ in range(N_USER_FRUITS + 2)],
            # Backup (rare: Moshi said most of the split's pool) = less common fruits that are in no
            # split's user pool, so test fruits never leak into train games and vice versa.
            backup_order=rng.sample(splits["recognised_only"], len(splits["recognised_only"]))))
    return specs


@dataclass
class GameState:
    spec: GameSpec
    phase: str = "moshi"          # "moshi" (Moshi's turn) | "gap" | "user" | "done"
    turn_start: float = 0.0       # when Moshi's current turn started (user speech end)
    last_speech: float | None = None   # end time of Moshi's last speech-audio frame in this turn
    speech_frames: int = 0
    next_clip_frame: int | None = None
    clip_tokens: np.ndarray | None = None
    clip_frame0: int = 0
    n_fruits_said: int = 0
    instructions_done: bool = False
    tokens: list[dict] = field(default_factory=list)       # Moshi text: {"t", "id", "piece"}
    levels: list[float] = field(default_factory=lambda: [-120.0])  # Moshi audio dBFS per slot (slot 0 empty)
    user_turns: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    end_time: float | None = None
    end_frame: int | None = None

    def moshi_fruits(self) -> list[str]:
        return [m["fruit"] for m in find_fruits(tokens_to_words(self.tokens))]


class GameRunner:
    """Loads Moshi once on the current CUDA device and runs batches of games in lockstep.

    Evaluation (default): loads base Moshi itself, grad disabled, persistent streaming state of size
    `batch` (exactly as for the base results). Training (`models=(mimi, tok, lm, lm_gen_config)`): uses
    the given (LoRA) model, enters streaming per run with the run's batch size so the model can be
    trained between runs, and with `record_codes=True` returns each game's token sequence aligned the
    way `LMModel.forward` expects it (for teacher-forced log-probs)."""

    def __init__(self, batch: int, decode_audio: bool = True, user_audio: str = "live",
                 models: tuple | None = None, record_codes: bool = False):
        import torch
        from moshi.models import LMGen, loaders
        self.torch = torch
        self.persistent = models is None
        self.record_codes = record_codes
        if models is None:
            torch.set_grad_enabled(False)
            info = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
            self.mimi = info.get_mimi(device="cuda")
            self.tok = info.get_text_tokenizer()
            self.lm = info.get_moshi(device="cuda", dtype=torch.bfloat16)
            gen_cfg = info.lm_gen_config
        else:
            self.mimi, self.tok, self.lm, gen_cfg = models
        self.lm_gen = LMGen(self.lm, **gen_cfg)
        self.sampling = {k: getattr(self.lm_gen, k) for k in ("use_sampling", "temp", "temp_text", "top_k", "top_k_text")}
        self.B = batch
        self.decode_audio = decode_audio
        assert user_audio in ("live", "tokens")
        self.user_audio = user_audio
        self.fs = int(self.mimi.sample_rate / self.mimi.frame_rate)
        assert abs(1 / self.mimi.frame_rate - FRAME_SEC) < 1e-9
        if self.persistent:
            self.mimi.streaming_forever(batch)
            self.lm_gen.streaming_forever(batch)
        bank = torch.load(ROOT / "data" / "tokens" / "clips.pt")
        self.silence = bank["silence"].long().numpy()
        self.clip_tokens = {p: t.long().numpy() for p, t in bank["clips"].items()}
        self.clips = load_clip_index()
        self.pcm_cache: dict[str, np.ndarray] = {}
        self.pieces = {}
        # Warm-up (CUDA graphs) on silence.
        for _ in range(6 if self.persistent else 0):
            codes = self.mimi.encode(torch.zeros(batch, 1, self.fs, device="cuda"))
            out = self.lm_gen.step(codes)
            if out is not None and decode_audio:
                self.mimi.decode(out[:, 1:])
        torch.cuda.synchronize()

    def piece(self, tid: int) -> str:
        if tid not in self.pieces:
            self.pieces[tid] = self.tok.id_to_piece(tid)
        return self.pieces[tid]

    def clip_pcm(self, path: str) -> np.ndarray:
        if path not in self.pcm_cache:
            import sphn
            self.pcm_cache[path] = sphn.read(str(ROOT / path), sample_rate=24000)[0][0]
        return self.pcm_cache[path]

    # ---------------------------------------------------------------- per-game state machine
    def _start_clip(self, g: GameState, frame: int, now: float) -> None:
        sp = g.spec
        if sp.script:  # ordinary conversation: the next scripted line (no instructions, no fruits)
            key = sp.script[len(g.user_turns)]
            row = self.clips[(sp.voice, "chat", key)]
            turn = {"kind": "chat", "text": row["text"], "fruit": None}
            g.n_fruits_said += 1
            why = f"chat line {g.n_fruits_said}/{len(sp.script)}"
        elif not g.instructions_done:
            row = self.clips[(sp.voice, "phrasing", sp.phrasing)]
            turn = {"kind": "instructions", "text": sp.phrasing, "fruit": None}
            g.instructions_done = True
            why = "after greeting turn"
        else:
            said = {t["fruit"] for t in g.user_turns if t["fruit"]} | set(g.moshi_fruits())
            fruit = next((f for f in sp.fruit_order if f not in said), None)
            backup = fruit is None
            if backup:  # pool exhausted (Moshi said many pool fruits): fall back to a backup pool
                fruit = next(f for f in sp.backup_order if f not in said)
            row = self.clips[(sp.voice, "fruit", fruit)]
            turn = {"kind": "fruit", "text": row["text"], "fruit": fruit}
            g.n_fruits_said += 1
            order = sp.backup_order if backup else sp.fruit_order
            skipped = [f for f in order[:order.index(fruit)] if f in said]
            why = f"user fruit {g.n_fruits_said}/{sp.n_user_fruits}" + (" (backup pool)" if backup else "") + (
                f"; skipped already-said {skipped[-3:]}" if skipped and skipped[-1] in set(g.moshi_fruits()) else "")
        start = frame * FRAME_SEC
        turn.update({"start": round(start, 3), "speech_end": round(start + row["duration"], 3),
                     "clip": row["path"], "voice": sp.voice, "frame": frame, "reason": why})
        g.user_turns.append(turn)
        g.clip_tokens = self.clip_tokens.get(row["path"])  # pre-encoded tokens (only used in "tokens" mode)
        g.clip_frame0 = frame
        g.turn_start = turn["speech_end"]
        g.last_speech, g.speech_frames = None, 0
        g.phase = "user"
        g.events.append({"t": round(start, 3), "event": "user_clip", "kind": turn["kind"], "text": turn["text"],
                         "speech_end": turn["speech_end"], "why": why})

    def _input_pcm(self, g: GameState, frame: int, out: np.ndarray) -> None:
        """Write user PCM for input frame `frame` into out (fs,), zeros if the user is silent."""
        out[:] = 0.0
        if g.user_turns:
            t = g.user_turns[-1]
            k = frame - t["frame"]
            x = self.clip_pcm(t["clip"])
            a, b = k * self.fs, (k + 1) * self.fs
            if 0 <= a < len(x):
                seg = x[a:b]
                out[:len(seg)] = seg

    def _input_frame(self, g: GameState, frame: int) -> np.ndarray:
        if g.clip_tokens is not None:
            k = frame - g.clip_frame0
            if 0 <= k < g.clip_tokens.shape[1]:
                return g.clip_tokens[:, k]
            if k >= g.clip_tokens.shape[1]:
                g.clip_tokens = None
        return self.silence

    def _after_step(self, g: GameState, frame: int, tid: int | None, db: float) -> None:
        """Called after input frame `frame`; Moshi's text for slot frame+1 is `tid`, and `db` is the
        RMS level (dBFS) of Moshi's audio in that slot (time [frame+1, frame+2) * 80 ms)."""
        if g.phase == "done":
            return
        g.levels.append(round(db, 1))
        now = round((frame + 1) * FRAME_SEC, 3)
        if tid is not None and tid not in PAD_IDS:
            g.tokens.append({"t": now, "id": tid, "piece": self.piece(tid)})
        if db > SPEECH_DBFS and now + FRAME_SEC > g.turn_start:
            g.last_speech = round(now + FRAME_SEC, 3)
            g.speech_frames += 1
        if g.phase == "user" and now >= g.turn_start:
            g.phase = "moshi"
        if g.phase == "moshi" and now >= g.turn_start:
            reason = None
            t_end = now + FRAME_SEC  # end of the slot just heard
            spoke = g.speech_frames >= MIN_SPEECH_FRAMES
            if spoke and t_end - max(g.last_speech, g.turn_start) >= QUIET_AFTER_SPEECH - 1e-9:
                reason = f"Moshi audio quiet {QUIET_AFTER_SPEECH}s after speaking (speech until {g.last_speech:.2f}s)"
            elif not spoke and t_end - g.turn_start >= SILENT_TURN - 1e-9:
                reason = f"Moshi silent {SILENT_TURN}s"
            elif now - g.turn_start >= TURN_CAP - 1e-9:
                reason = f"turn cap {TURN_CAP}s (Moshi still talking)"
            if reason:
                now = round(t_end, 3)
                g.events.append({"t": now, "event": "moshi_turn_end", "why": reason})
                if g.n_fruits_said >= g.spec.n_user_fruits:
                    g.phase, g.end_time, g.end_frame = "done", now, frame + 2
                    return
                gap = g.spec.gaps[len(g.user_turns)]
                g.next_clip_frame = int(round((now + gap) / FRAME_SEC))
                g.phase = "gap"
        if g.phase == "gap" and db > SPEECH_DBFS and now - g.turn_start < TURN_CAP:
            # Moshi started talking again before the user spoke: the user waits, Moshi's turn goes on.
            g.phase = "moshi"
            g.events.append({"t": now, "event": "gap_cancelled", "why": "Moshi resumed speaking during the gap; user waits"})
        if g.phase == "gap" and frame + 1 >= g.next_clip_frame:
            self._start_clip(g, frame + 1, frame + 1)

    # ---------------------------------------------------------------- batch
    def run(self, specs: list[GameSpec], seed: int) -> list[dict]:
        torch = self.torch
        if self.persistent:
            return self._run(specs, seed, self.B)
        B = len(specs)
        with torch.no_grad(), self.mimi.streaming(B), self.lm_gen.streaming(B):
            return self._run(specs, seed, B)

    def _run(self, specs: list[GameSpec], seed: int, B: int) -> list[dict]:
        torch = self.torch
        from moshi.run_inference import seed_all
        assert len(specs) <= B
        cols: list = []   # delayed-sequence input column LMGen read at each step, (B, 17)
        games = [GameState(spec=s) for s in specs]
        n_real = len(games)
        self.mimi.reset_streaming()
        self.lm_gen.reset_streaming()
        seed_all(seed)
        max_frames = int(MAX_GAME_SEC / FRAME_SEC)
        moshi_audio: list[np.ndarray] = []
        fed: list = []  # user codes actually fed to Moshi, one (B, 8) tensor per frame
        user_codes = np.zeros((B, 8, 1), dtype=np.int64)
        user_pcm = np.zeros((B, 1, self.fs), dtype=np.float32)
        frame = 0
        torch.cuda.synchronize()
        import time
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        while frame < max_frames and any(g.phase != "done" for g in games):
            if self.user_audio == "live":
                for b in range(n_real):
                    self._input_pcm(games[b], frame, user_pcm[b, 0])
                codes = self.mimi.encode(torch.from_numpy(user_pcm).cuda())
                if frame == 0:
                    # As moshi.server: the first encoded frame has a special structure (left padding),
                    # so the encoder state is reset to reapply it on the next call.
                    self.mimi.reset_streaming()
            else:
                for b in range(B):
                    user_codes[b, :, 0] = self._input_frame(games[b], frame) if b < n_real else self.silence
                codes = torch.from_numpy(user_codes).cuda()
            fed.append(codes[:, :, 0])
            out = self.lm_gen.step(codes)
            if self.record_codes:
                st = self.lm_gen._streaming_state
                cols.append(st.cache[:, :, (st.offset - 1) % st.cache.shape[2]].clone())
            tids: list[int | None] = [None] * B
            dbs = [-120.0] * B
            if out is not None:
                tids = out[:, 0, 0].tolist()
                # Moshi's audio is always decoded: the turn-end rule listens to it.
                pcm = self.mimi.decode(out[:, 1:])[:, 0].float()
                dbs = (10 * torch.log10(pcm.pow(2).mean(-1) + 1e-12)).tolist()
                if self.decode_audio:
                    moshi_audio.append(pcm.cpu().numpy())
            elif self.decode_audio:
                moshi_audio.append(np.zeros((B, self.fs), dtype=np.float32))
            for b in range(n_real):
                self._after_step(games[b], frame, tids[b], dbs[b])
            frame += 1
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        fed_np = torch.stack(fed, -1).to(torch.int16).cpu().numpy()  # (B, 8, frames)
        aligned = None
        if self.record_codes:
            # D[o] = input column read at step o. LMModel.forward delays codes as D[t + 1 + d_k][k] =
            # codes[k][t], so codes[k][t] = D[t + 1 + d_k][k] (T = steps - 1 - max_delay frames).
            # Aligned frame t <-> game time (t + 2) * FRAME_SEC for Moshi's text (see game JSON "t").
            D = torch.stack(cols, -1)  # (B, 17, steps)
            delays = list(self.lm.delays)
            T = D.shape[-1] - 1 - max(delays)
            aligned = torch.stack([D[:, k, 1 + d:1 + d + T] for k, d in enumerate(delays)], 1)
            aligned = aligned.to(torch.int16).cpu().numpy()
        results = []
        for b, g in enumerate(games):
            if g.end_time is None:
                g.end_time, g.end_frame = frame * FRAME_SEC, frame
                g.events.append({"t": g.end_time, "event": "game_cap", "why": f"{MAX_GAME_SEC}s limit"})
            n = min(g.end_frame + 1, frame + 1)
            summary = {
                "game_id": g.spec.game_id, "spec": g.spec.to_dict(), "starter": g.spec.starter,
                "sampling": self.sampling, "sampling_seed": seed, "frame_sec": FRAME_SEC,
                "end_time": round(g.end_time, 3), "user_turns": g.user_turns, "moshi_tokens": g.tokens,
                "moshi_text": "".join(t["piece"] for t in g.tokens).replace("▁", " ").strip(),
                "moshi_dbfs": g.levels, "speech_dbfs": SPEECH_DBFS,
                "events": g.events, "batch_size": B, "batch_wall_sec": round(wall, 2),
                "batch_frames": frame, "peak_mem_gb": round(peak, 2), "user_audio": self.user_audio,
            }
            audio = None
            if self.decode_audio:
                # output frame j was returned at step j and belongs in slot j+1
                left = np.zeros(n * self.fs, dtype=np.float32)
                for j, fr in enumerate(moshi_audio[: n - 1]):
                    left[(j + 1) * self.fs:(j + 2) * self.fs] = fr[b]
                right = np.zeros_like(left)
                for t in g.user_turns:
                    x = self.clip_pcm(t["clip"])
                    a = t["frame"] * self.fs
                    right[a:a + len(x)] = x[: max(0, len(right) - a)]
                audio = np.stack([left, right])
            res = {"summary": summary, "audio": audio, "user_codes": fed_np[b, :, :n]}
            if aligned is not None:
                res["codes"] = aligned[b, :, :max(0, min(aligned.shape[-1], n - 2))]
            results.append(res)
        return results


def save_game(res: dict, out_dir: Path) -> None:
    import sphn
    out_dir.mkdir(parents=True, exist_ok=True)
    gid = res["summary"]["game_id"]
    (out_dir / f"{gid}.json").write_text(json.dumps(res["summary"], indent=1, ensure_ascii=False))
    if res.get("user_codes") is not None:
        np.save(out_dir / f"{gid}.user_codes.npy", res["user_codes"])
    if res["audio"] is not None:
        sphn.write_wav(str(out_dir / f"{gid}.wav"), res["audio"], sample_rate=24000)
