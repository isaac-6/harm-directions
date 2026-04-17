"""
download_datasets.py
--------------------
Downloads, normalises, and composes the OOD Generalization splits used in the
LatentBiopsy pipeline.

STRICT DESIGN RULES:
1. SINGLE-LINE FILTER: Prompts with newlines are rejected.
2. PUNCTUATION STRIP: Terminal [.,;:!?] are removed.
3. OOD GENERALIZATION SPLIT:
   - FIT: 100 Vanilla Harmful (AdvBench) vs 100 Vanilla Normative (Alpaca).
   - EVAL: Strict segregation of test benchmarks for disaggregated metrics reporting.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, cast

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path("data/raw")
SOURCES_DIR = ROOT / "sources"
SPLITS_DIR = ROOT / "splits"
DATA_CARD_OUT = ROOT / "data_card.json"

SRC: dict[str, Path] = {
    "advbench": SOURCES_DIR / "advbench.txt",
    "harmbench": SOURCES_DIR / "harmbench.txt",
    "jailbreakbench": SOURCES_DIR / "jailbreakbench.txt",
    "xstest": SOURCES_DIR / "xstest.txt",
    "alpaca": SOURCES_DIR / "alpaca.txt",
}

SOURCE_META: dict[str, dict] = {
    "advbench": {"pillar": "I — Canonical Harm", "citation": "Zou et al. (2023)."},
    "harmbench": {"pillar": "I — Canonical Harm", "citation": "Mazeika et al. (2024)."},
    "jailbreakbench": {"pillar": "II — Adversarial Attacks", "citation": "Chao et al. (2024)."},
    "xstest": {"pillar": "III — Hard Benign", "citation": "Röttger et al. (2023)."},
    "alpaca": {"pillar": "Normative", "citation": "Taori et al. (2023)."},
}

# ---------------------------------------------------------------------------
# Normalisation & Filtering
# ---------------------------------------------------------------------------

_TERMINAL_PUNCT_RE = re.compile(r"[\s.,;:!?]+$")


def normalize_prompt(text: str) -> str:
    if not text:
        return ""
    return _TERMINAL_PUNCT_RE.sub("", text.strip())


def _write_lines(path: Path, raw_lines: list[str]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    single_lines = [line for line in raw_lines if "\n" not in line.strip()]
    dropped_multiline = len(raw_lines) - len(single_lines)
    normalised = [normalize_prompt(line) for line in single_lines if len(normalize_prompt(line)) > 10]

    with open(path, "w", encoding="utf-8") as f:
        for line in normalised:
            f.write(line + "\n")

    changed = sum(
        1 for r in single_lines if len(r.strip()) > 10 and r.strip() != normalize_prompt(r)
    )
    print(
        f"    Saved {len(normalised):,} prompts → {path.name}  "
        f"[{dropped_multiline:,} multiline dropped, {changed:,} punctuation-stripped]"
    )

    return {
        "raw_count": len(raw_lines),
        "dropped_multiline": dropped_multiline,
        "final_count": len(normalised),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for ln in f if ln.strip())


def _already_exists(path: Path) -> bool:
    n = _line_count(path)
    if n > 0:
        print(f"    [skip] {path.name} — {n:,} prompts already on disk.")
        return True
    return False


def _is_sufficient(path: Path, needed: int) -> bool:
    n = _line_count(path)
    if n >= needed:
        print(f"    [skip] {path.name} — {n:,} prompts (>= {needed}).")
        return True
    return False


# ---------------------------------------------------------------------------
# Downloaders
# ---------------------------------------------------------------------------

_DATA_CARD: dict[str, dict] = {}


def download_advbench() -> None:
    print("\n[advbench]")
    if _already_exists(SRC["advbench"]):
        return
    import pandas as pd
    import requests

    r = requests.get(
        "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    )
    df = pd.read_csv(StringIO(r.text))
    prompts = df["goal"].dropna().astype(str).tolist()
    _DATA_CARD["advbench"] = {**SOURCE_META["advbench"], **_write_lines(SRC["advbench"], prompts)}


def download_harmbench() -> None:
    print("\n[harmbench]")
    if _already_exists(SRC["harmbench"]):
        return
    from datasets import load_dataset

    ds = load_dataset("walledai/HarmBench", "standard", split="train", token=True)
    data = cast(dict[str, list[Any]], ds.to_dict())
    prompts = [str(x) for x in data["prompt"] if x]
    _DATA_CARD["harmbench"] = {
        **SOURCE_META["harmbench"],
        **_write_lines(SRC["harmbench"], prompts),
    }


def download_jailbreakbench() -> None:
    print("\n[jailbreakbench]")
    if _already_exists(SRC["jailbreakbench"]):
        return
    from datasets import load_dataset

    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful", token=True)
    data = cast(dict[str, list[Any]], ds.to_dict())
    prompts = [str(x) for x in data["Goal"] if x]
    _DATA_CARD["jailbreakbench"] = {
        **SOURCE_META["jailbreakbench"],
        **_write_lines(SRC["jailbreakbench"], prompts),
    }


def download_xstest() -> None:
    print("\n[xstest]")
    if _already_exists(SRC["xstest"]):
        return
    import pandas as pd
    import requests

    r = requests.get(
        "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"
    )
    df = pd.read_csv(StringIO(r.text))
    safe_df = df[df["label"] == "safe"] if "label" in df.columns else df
    prompts = safe_df["prompt"].dropna().astype(str).tolist()
    _DATA_CARD["xstest"] = {**SOURCE_META["xstest"], **_write_lines(SRC["xstest"], prompts)}


def download_alpaca(n_total: int, seed: int) -> None:
    print(f"\n[alpaca] (n={n_total})")
    if _is_sufficient(SRC["alpaca"], n_total):
        return
    from datasets import load_dataset

    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    data = cast(dict[str, list[Any]], ds.to_dict())
    candidates = [
        str(i).strip()
        for i, p in zip(data["instruction"], data["input"], strict=True)
        if not str(p).strip() and "\n" not in str(i).strip() and 20 < len(str(i).strip()) < 300
    ]
    random.seed(seed)
    sampled = random.sample(candidates, min(len(candidates), n_total))
    _DATA_CARD["alpaca"] = {**SOURCE_META["alpaca"], **_write_lines(SRC["alpaca"], sampled)}


# ---------------------------------------------------------------------------
# Prompt dataset statistics
# ---------------------------------------------------------------------------
def print_length_statistics(filepaths: list[Path]):
    print("\n" + "=" * 60 + "\n  Dataset Length Statistics (For TMLR Paper)\n" + "=" * 60)
    print(f"{'Dataset':<30} | {'Count':<6} | {'Avg Chars':<10} | {'Std Chars':<10}")
    print("-" * 65)

    for path in filepaths:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        lengths = [len(line) for line in lines]
        if lengths:
            avg_len = np.mean(lengths)
            std_len = np.std(lengths)
            print(f"{path.stem:<30} | {len(lines):<6} | {avg_len:<10.1f} | {std_len:<10.1f}")


# ---------------------------------------------------------------------------
# Compose Disaggregated OOD Split
# ---------------------------------------------------------------------------


def compose_splits(seed: int) -> dict:
    print("\n" + "=" * 60 + "\n  Composing OOD Generalization Split\n" + "=" * 60)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    def sample_and_write(
        source_key: str,
        dest_name: str,
        n: int | None = None,
        disjoint_from: list[str] | None = None,
    ) -> list[str]:
        if not SRC[source_key].exists():
            print(f"    [warn] Missing {source_key}. Skipping.")
            return []
        lines = _read_lines(SRC[source_key])
        if disjoint_from:
            lines = [line for line in lines if line not in disjoint_from]
        rng.shuffle(lines)
        selected = lines[:n] if n else lines
        with open(SPLITS_DIR / dest_name, "w", encoding="utf-8") as f:
            for line in selected:
                f.write(line + "\n")
        print(f"    Exported {len(selected):>4} → {dest_name}")
        return selected

    print("\n  [ FIT SET (Vanilla Anchor) ]")
    fit_harmful = sample_and_write("advbench", "fit_harmful_advbench.txt", 100)
    fit_norm = sample_and_write("alpaca", "fit_normative_alpaca.txt", 100)

    print("\n  [ VALIDATION SET (Layer Selection) ]")
    val_harmful = sample_and_write(
        "advbench", "val_harmful_advbench.txt", 50, disjoint_from=fit_harmful
    )
    val_norm = sample_and_write("alpaca", "val_normative_alpaca.txt", 50, disjoint_from=fit_norm)

    print("\n  [ EVAL SET: HARMFUL ]")
    sample_and_write(
        "advbench", "eval_harmful_advbench.txt", disjoint_from=fit_harmful + val_harmful
    )
    sample_and_write("harmbench", "eval_harmful_harmbench.txt")
    sample_and_write("jailbreakbench", "eval_harmful_jailbreakbench.txt")

    print("\n  [ EVAL SET: BENIGN / NORMATIVE ]")
    sample_and_write("xstest", "eval_benign_xstest.txt")
    sample_and_write("alpaca", "eval_benign_alpaca.txt", 500, disjoint_from=fit_norm + val_norm)

    print_length_statistics(list(SPLITS_DIR.glob("*.txt")))

    return {
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "type": "OOD Generalization Split",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpaca-n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    download_advbench()
    download_harmbench()
    download_jailbreakbench()
    download_xstest()

    # FIXED: Pylance typo error mapped to args.alpaca_n
    download_alpaca(n_total=args.alpaca_n, seed=args.seed)

    prov = compose_splits(seed=args.seed)

    for name, path in SRC.items():
        if name not in _DATA_CARD and path.exists():
            _DATA_CARD[name] = {**SOURCE_META.get(name, {}), "final_count": _line_count(path)}
    DATA_CARD_OUT.write_text(
        json.dumps(
            {"generated": datetime.now().isoformat(), "sources": _DATA_CARD, "recipe": prov},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
