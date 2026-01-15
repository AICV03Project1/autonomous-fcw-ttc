# split_envs_from_summary.py
import os
import math
import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


def normalize_weather(x: str) -> str:
    """Normalize weather strings; then caller can apply merge policy."""
    if pd.isna(x):
        return x
    s = str(x).strip()
    s_low = s.lower()
    if s_low == "rain":
        return "Rain"
    if s_low == "noisy":
        return "Noisy"
    # Title-case handles RAIN -> Rain, FOG -> Fog, etc.
    return s.title()


def zscore(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    mu = np.nanmean(arr)
    sd = np.nanstd(arr)
    if sd == 0 or not np.isfinite(sd):
        return np.zeros_like(arr)
    return (arr - mu) / sd


def compute_difficulty_bucket(df: pd.DataFrame) -> pd.DataFrame:
    # Composite difficulty: overlap + (1-purity) + (1-cover)
    score = (
        zscore(df["numids_ge3_ratio"].values)
        + zscore((1.0 - df["purity_p10"].values))
        + zscore((1.0 - df["cover_p10"].values))
    )
    df = df.copy()
    df["difficulty_score"] = score
    q1, q2 = np.quantile(score, [1 / 3, 2 / 3])
    df["difficulty_bucket"] = pd.cut(
        df["difficulty_score"],
        bins=[-np.inf, q1, q2, np.inf],
        labels=["Easy", "Medium", "Hard"],
    )
    return df


def build_split(
    df: pd.DataFrame,
    ratios: dict,
    seed: int,
    max_over: float,
    min_factor: float,
    min_abs: int,
    enforce_brake: bool,
):
    rng = np.random.RandomState(seed)

    # Which actions to enforce as "rare" constraints
    rare_actions = ["state1_on_frames", "state2_on_frames", "state3_on_frames"]
    if enforce_brake:
        rare_actions = ["state0_on_frames"] + rare_actions

    # Targets (boxes-based)
    total_boxes = float(df["num_boxes"].sum())
    target_boxes = {s: total_boxes * ratios[s] for s in ratios}

    # Min action frames constraints for val/test
    tot_action = {a: int(df[a].sum()) for a in rare_actions}
    min_action_frames = {s: {} for s in ["val", "test"]}
    for s in ["val", "test"]:
        for a in rare_actions:
            thr = int(math.floor(tot_action[a] * ratios[s] * min_factor))
            min_action_frames[s][a] = max(min_abs, thr)

    # Categories
    weather_cats = sorted(df["weather_norm"].unique().tolist())
    global_diff_prop = df["difficulty_bucket"].value_counts(normalize=True).to_dict()

    df_idx = df.set_index("env")
    unassigned = set(df["env"].tolist())
    split_envs = {"train": [], "val": [], "test": []}

    def split_boxes(split: str) -> int:
        if not split_envs[split]:
            return 0
        return int(df_idx.loc[split_envs[split], "num_boxes"].sum())

    def split_action_sum(split: str, a: str) -> int:
        if not split_envs[split]:
            return 0
        return int(df_idx.loc[split_envs[split], a].sum())

    def deficits(split: str) -> dict:
        d = {}
        for a, thr in min_action_frames[split].items():
            d[a] = max(0, thr - split_action_sum(split, a))
        return d

    # Box caps for val/test to prevent drift
    max_boxes = {
        "val": target_boxes["val"] * (1.0 + max_over),
        "test": target_boxes["test"] * (1.0 + max_over),
    }

    # ------------------------
    # Phase 1) Seed: weather coverage (val/test)
    # ------------------------
    def pick_seed_for_weather(split: str, wcat: str):
        cands = [e for e in unassigned if df_idx.loc[e, "weather_norm"] == wcat]
        if not cands:
            return None

        cur_boxes = split_boxes(split)
        best = None
        best_score = -1e18

        for e in cands:
            r = df_idx.loc[e]
            new_boxes = cur_boxes + float(r["num_boxes"])
            if new_boxes > max_boxes[split]:
                continue

            # Prefer smaller envs but with higher rare-action density
            rare = float(sum(r[a] for a in rare_actions))
            score = (rare + 1.0) / max(1.0, float(r["num_boxes"]))
            score += 1e-9 * rng.rand()  # tie-break

            if score > best_score:
                best_score = score
                best = e

        if best is None:
            # fallback: smallest boxes env under this weather
            best = min(cands, key=lambda x: df_idx.loc[x, "num_boxes"])
        return best

    for split in ["val", "test"]:
        for w in weather_cats:
            e = pick_seed_for_weather(split, w)
            if e is None:
                continue
            split_envs[split].append(e)
            unassigned.remove(e)

    # ------------------------
    # Phase 2) Seed: satisfy rare action minima (val/test) within caps (relax if needed)
    # ------------------------
    def pick_for_deficits(split: str):
        d = deficits(split)
        cur_boxes = split_boxes(split)

        best = None
        best_score = -1e18
        for e in list(unassigned):
            r = df_idx.loc[e]
            new_boxes = cur_boxes + float(r["num_boxes"])
            if new_boxes > max_boxes[split]:
                continue

            # Coverage gain for remaining deficits
            gain = 0.0
            for a, need in d.items():
                if need <= 0:
                    continue
                gain += min(float(need), float(r[a]))

            if gain <= 0:
                continue

            # Gain per boxes: choose minimal-box, high-coverage envs
            score = gain / max(1.0, float(r["num_boxes"]))
            score += 1e-9 * rng.rand()

            if score > best_score:
                best_score = score
                best = e

        return best

    for split in ["val", "test"]:
        guard = 0
        while any(v > 0 for v in deficits(split).values()) and guard < 5000:
            guard += 1
            e = pick_for_deficits(split)
            if e is None:
                # cannot meet constraints within cap; relax cap slightly and retry
                max_boxes[split] *= 1.05
                continue
            split_envs[split].append(e)
            unassigned.remove(e)

    # ------------------------
    # Phase 3) Greedy fill remaining envs with strong val/test caps
    # ------------------------
    def diff_penalty(env_list):
        n = len(env_list)
        if n == 0:
            return 0.0
        sub = df_idx.loc[env_list]
        counts = Counter(sub["difficulty_bucket"])
        p = 0.0
        for b in ["Easy", "Medium", "Hard"]:
            prop = counts.get(b, 0) / n
            p += (prop - global_diff_prop.get(b, 0.0)) ** 2
        return p

    def weather_missing_penalty(split: str, env_list):
        if split not in ["val", "test"]:
            return 0.0
        sub = df_idx.loc[env_list]
        wc = Counter(sub["weather_norm"])
        missing = sum(1 for w in weather_cats if wc.get(w, 0) == 0)
        return float(missing)

    def action_deficit_penalty(split: str, env_list):
        if split not in ["val", "test"]:
            return 0.0
        sub = df_idx.loc[env_list]
        p = 0.0
        for a, thr in min_action_frames[split].items():
            have = float(sub[a].sum())
            p += max(0.0, thr - have) / max(1.0, float(thr))
        return p

    def penalty(split: str, env: str):
        r = df_idx.loc[env]
        cur_envs = split_envs[split]
        new_envs = cur_envs + [env]

        # Hard cap for val/test
        if split in ["val", "test"]:
            cur_boxes = split_boxes(split)
            if cur_boxes >= max_boxes[split]:
                return 1e18
            if cur_boxes + float(r["num_boxes"]) > max_boxes[split]:
                return 1e18

        new_boxes = float(df_idx.loc[new_envs, "num_boxes"].sum())
        tb = target_boxes[split]
        p_boxes = ((new_boxes - tb) / tb) ** 2

        p_diff = diff_penalty(new_envs)
        p_weather = weather_missing_penalty(split, new_envs)
        p_action = action_deficit_penalty(split, new_envs)

        # weights: prioritize action/weather in val/test; boxes everywhere; difficulty mild
        w_boxes = 3.0
        w_diff = 2.0
        w_weather = 4.0
        w_action = 6.0
        if split == "train":
            w_weather = 0.0
            w_action = 0.0

        return w_boxes * p_boxes + w_diff * p_diff + w_weather * p_weather + w_action * p_action

    # Prefer sending large envs to train; process remaining in descending num_boxes
    remaining = sorted(list(unassigned), key=lambda e: float(df_idx.loc[e, "num_boxes"]), reverse=True)
    for e in remaining:
        scores = []
        for s in ["train", "val", "test"]:
            scores.append((penalty(s, e), rng.rand(), s))
        scores.sort()
        chosen = scores[0][2]
        split_envs[chosen].append(e)
        unassigned.remove(e)

    # Final: build assignment table
    rows = []
    for split in ["train", "val", "test"]:
        for env in split_envs[split]:
            r = df_idx.loc[env]
            rows.append(
                {
                    "env": env,
                    "split": split,
                    "weather": r["weather_norm"],
                    "difficulty_bucket": str(r["difficulty_bucket"]),
                    "num_boxes": int(r["num_boxes"]),
                    "num_frames_matched": int(r["num_frames_matched"]),
                    "num_images": int(r["num_images"]),
                    **{a: int(r[a]) for a in ["state0_on_frames", "state1_on_frames", "state2_on_frames", "state3_on_frames"]},
                }
            )
    assign_df = pd.DataFrame(rows)

    return split_envs, assign_df, target_boxes, min_action_frames, weather_cats, rare_actions


def make_report(assign_df: pd.DataFrame, weather_cats):
    # Split-level totals + distributions
    report_rows = []
    for split in ["train", "val", "test"]:
        sub = assign_df[assign_df["split"] == split].copy()
        row = {
            "split": split,
            "num_envs": int(len(sub)),
            "num_boxes": int(sub["num_boxes"].sum()),
            "num_frames": int(sub["num_frames_matched"].sum()),
            "num_images": int(sub["num_images"].sum()),
            "state0_on_frames": int(sub["state0_on_frames"].sum()),
            "state1_on_frames": int(sub["state1_on_frames"].sum()),
            "state2_on_frames": int(sub["state2_on_frames"].sum()),
            "state3_on_frames": int(sub["state3_on_frames"].sum()),
        }

        wc = sub["weather"].value_counts().to_dict()
        for w in weather_cats:
            row[f"weather_{w}_envs"] = int(wc.get(w, 0))

        dc = sub["difficulty_bucket"].value_counts().to_dict()
        for b in ["Easy", "Medium", "Hard"]:
            row[f"diff_{b}_envs"] = int(dc.get(b, 0))

        report_rows.append(row)

    return pd.DataFrame(report_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_xlsx", type=str, default="env_summary_for_split.xlsx")
    ap.add_argument("--out_dir", type=str, default="_split_out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--max_over", type=float, default=0.08, help="val/test max overflow over target boxes (e.g., 0.08 = +8%)")
    ap.add_argument("--min_factor", type=float, default=0.6, help="min action frames factor vs target share")
    ap.add_argument("--min_abs", type=int, default=30, help="absolute minimum action frames per val/test")
    ap.add_argument("--enforce_brake", action="store_true", help="also enforce state0_on_frames minima (default: off)")
    args = ap.parse_args()

    summary_path = Path(args.summary_xlsx)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(summary_path)

    # Basic cleanup (guard against empty rows)
    df = df.dropna(subset=["env"]).copy()

    # Weather normalize + policy (C): merge Noisy -> Fog
    df["weather_norm"] = df["weather"].apply(normalize_weather).replace({"Noisy": "Fog"})

    # Compute difficulty bucket
    df = compute_difficulty_bucket(df)

    ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    ssum = ratios["train"] + ratios["val"] + ratios["test"]
    if abs(ssum - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {ssum}")

    split_envs, assign_df, target_boxes, min_action_frames, weather_cats, rare_actions = build_split(
        df=df,
        ratios=ratios,
        seed=args.seed,
        max_over=args.max_over,
        min_factor=args.min_factor,
        min_abs=args.min_abs,
        enforce_brake=args.enforce_brake,
    )

    # Save env lists
    for split in ["train", "val", "test"]:
        env_list = sorted(split_envs[split])
        (out_dir / f"{split}_envs.txt").write_text("\n".join(env_list), encoding="utf-8")

    # Save assignment + report
    assign_df.to_csv(out_dir / "split_assignment.csv", index=False, encoding="utf-8-sig")
    report_df = make_report(assign_df, weather_cats)
    report_df.to_csv(out_dir / "split_report.csv", index=False, encoding="utf-8-sig")

    # Save metadata for reproducibility
    meta = {
        "summary_xlsx": str(summary_path),
        "policy": {"weather_merge": "Noisy->Fog"},
        "seed": args.seed,
        "ratios": ratios,
        "target_boxes": target_boxes,
        "max_over": args.max_over,
        "min_factor": args.min_factor,
        "min_abs": args.min_abs,
        "enforce_brake": bool(args.enforce_brake),
        "rare_actions_enforced": rare_actions,
        "min_action_frames_val": min_action_frames["val"],
        "min_action_frames_test": min_action_frames["test"],
        "weather_categories": weather_cats,
    }
    (out_dir / "split_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Console summary
    print("Saved split to:", out_dir.resolve())
    print(report_df.to_string(index=False))


if __name__ == "__main__":
    main()
