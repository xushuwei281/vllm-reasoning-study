"""Immutable data manifests and prompts. Gold stays outside generation payloads."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import random
import httpx

MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATA_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
PROMPT = "Solve this math problem. Explain your reasoning and put only the final numeric answer in \\boxed{{}}.\n\n{question}"

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def write_json(path, value, *, immutable=False):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if immutable and p.exists():
        if p.read_text() != text:
            raise FileExistsError(f"immutable artifact differs: {p}")
        return
    if immutable:
        with p.open("x") as f: f.write(text)
    else:
        tmp = p.with_suffix(p.suffix + ".tmp"); tmp.write_text(text); tmp.replace(p)

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def write_jsonl(path, rows):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as f:
        for row in rows: f.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

def prepare_dataset(directory, split="train", *, freeze_path=None):
    if split not in {"train", "test"}: raise ValueError("official train/test only")
    if split == "test":
        if not freeze_path or not Path(freeze_path).exists():
            raise ValueError("freeze selected configurations/protocol before fetching test")
        frozen = json.loads(Path(freeze_path).read_text())
        from .fixed_compute import validate_frozen_protocol
        validate_frozen_protocol(frozen,require_integrity=True)
        if frozen.get("data_revision") != DATA_REVISION: raise ValueError("data revision mismatch")
    p = Path(directory); p.mkdir(parents=True, exist_ok=True)
    target = p / f"{split}.jsonl"
    url = f"https://raw.githubusercontent.com/openai/grade-school-math/{DATA_REVISION}/grade_school_math/data/{split}.jsonl"
    if not target.exists():
        response = httpx.get(url, follow_redirects=True, timeout=60); response.raise_for_status()
        # Assign stable official line indices. No reshuffling of source.
        rows = [{"question_id": f"gsm8k:{split}:{i:05d}", "question": r["question"], "gold": r["answer"]}
                for i, line in enumerate(response.text.splitlines()) if line.strip() for r in [json.loads(line)]]
        write_jsonl(target, rows)
    rows = read_jsonl(target)
    if len(rows) != {"train": 7473, "test": 1319}[split]: raise ValueError("unexpected source row count")
    manifest = {"dataset": "GSM8K", "split": split, "revision": DATA_REVISION, "source": url,
                "count": len(rows), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    if split == "train":
        ids = [r["question_id"] for r in rows]; random.Random(20261103).shuffle(ids)
        manifest["partitions"] = {"calibration": ids[:128], "profiling": ids[128:640],
                                  "development_validation": ids[640:896]}
    write_json(p/f"{split}-manifest.json", manifest, immutable=True)
    return manifest

def select_questions(directory, partition="profiling"):
    split = "test" if partition == "test" else "train"
    manifest = json.loads((Path(directory)/f"{split}-manifest.json").read_text())
    rows = read_jsonl(Path(directory)/f"{split}.jsonl")
    if hashlib.sha256((Path(directory)/f"{split}.jsonl").read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("dataset modified after manifest")
    if split == "test": return rows
    ids = set(manifest["partitions"][partition])
    return [r for r in rows if r["question_id"] in ids]

def prompt_for(question, tokenizer, profile="short"):
    message = PROMPT.format(question=question)
    # Long-context profile is labelled stress: deterministic irrelevant context, no answers.
    if profile == "long_context_stress":
        message = ("Context for a school newsletter: the classroom has a weekly study schedule. " * 24) + "\n" + message
    return tokenizer.apply_chat_template([{"role":"user", "content":message}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True)

def freeze_protocol(path, study, selected, dataset_manifest):
    if not selected or any(r.get("source") != "measured" or not r.get("metrics", {}).get("feasible") for r in selected):
        raise ValueError("freeze requires measured feasible development finalists")
    # Select IDs without reading any official test content. A predeclared seed
    # avoids relying on source-file order while preserving legacy protocols.
    test_ids = [f"gsm8k:test:{i:05d}" for i in range(1319)]
    cohort_seed = study.get("final", {}).get("test_cohort_seed")
    if cohort_seed is not None:
        if not isinstance(cohort_seed, int) or isinstance(cohort_seed, bool):
            raise ValueError("test_cohort_seed must be an integer fixed before test access")
        random.Random(cohort_seed).shuffle(test_ids)
    payload = {"study":study, "selected":selected, "data_revision":DATA_REVISION,
               "development_manifest":dataset_manifest, "test_question_ids":test_ids,
               "parser_version":"strict-final-v1", "test_results_seen":False}
    from .fixed_compute import validate_frozen_protocol
    validate_frozen_protocol(payload,require_integrity=False)
    payload["freeze_hash"] = digest(payload)
    write_json(path, payload, immutable=True)
    return payload
