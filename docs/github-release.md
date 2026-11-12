# Complete evidence release

The Git repository contains source, tests, documentation, reports and the smaller evidence files. The full raw records are distributed with the original project snapshot in a versioned release archive. The archive is supplied as six binary pieces with a `SHA256SUMS` file; join the verified pieces to reconstruct the original ZIP. This keeps ordinary clones smaller while preserving the measurements needed for independent audits.

| Item | Value |
|---|---|
| Repository | `xushuwei281/vllm-reasoning-study` |
| Release tag | `evidence-2026-11-03` |
| Reconstructed archive | `vllm-reasoning-study-full-evidence.zip` |
| Download pieces | `vllm-reasoning-study-full-evidence.zip.part-01` through `.part-06` |
| Piece sizes | Parts 01–05: 20,000,000 bytes each; part 06: 19,917,708 bytes |
| Piece checksum file | `SHA256SUMS` |
| Reconstructed size | 119,917,708 bytes |
| Reconstructed SHA-256 | `5fd8bac621e6c36dc58e2748e75046ebb8ace4ffa4fbef021abfc3eba98bc6a0` |
| Extraction root | `vllm-reasoning-study/` |
| Original snapshot | 757 payload files plus `MANIFEST.sha256` |

[Open the release](https://github.com/xushuwei281/vllm-reasoning-study/releases/tag/evidence-2026-11-03) to download all six pieces and `SHA256SUMS`, or use the commands below. The individual pieces are not separately extractable ZIP files.

## What is included

The Git checkout retains the original source and compact evidence. The following three raw-run directories and disposable caches are excluded:

```text
results/continuation/runs/
results/continuation/final/runs/
results/measured/runs/
```

Those directories are included in the full archive. They contain per-run requests, scored responses, offered traces, server logs and metrics, queue records, telemetry and execution provenance. The final four arms are under `results/continuation/final/runs/`; main search, independent validation and the optional context check are under `results/continuation/runs/`.

The archive also retains the original documentation, frozen settings, plans, resource ledger, method histories, surrogate checkpoints, final outcomes and derived results. The GitHub-facing README and this release guide are publication additions; they are not changes to the recorded experiment. The archive is the complete original delivery snapshot, not an archive of later Git commits.

The original archive manifest is also available in Git as
`artifacts/full-evidence-MANIFEST.sha256`. It describes the full archive, not the
compact checkout. The editable Word report and its renderer are publication
additions supplied in Git. The original execution code remains unchanged.

Qwen model weights, full dataset downloads, environments and caches are not supplied. Dataset manifests record the pinned public data used by the experiment. The archive includes generated responses and retained evaluation evidence, so it is more complete than a package containing only summary metrics.

## Download and verify

Run from a local clone with the [GitHub CLI](https://cli.github.com/), `shasum` and `unzip` available. This uses a new directory and stops on an error. If that directory already exists, choose a different fresh name rather than overwriting it.

```sh
set -eu
mkdir -p work
mkdir work/evidence-2026-11-03

gh release download evidence-2026-11-03 \
  --repo xushuwei281/vllm-reasoning-study \
  --pattern 'vllm-reasoning-study-full-evidence.zip.part-*' \
  --pattern SHA256SUMS \
  --dir work/evidence-2026-11-03

cd work/evidence-2026-11-03
shasum -a 256 -c SHA256SUMS

cat vllm-reasoning-study-full-evidence.zip.part-01 \
    vllm-reasoning-study-full-evidence.zip.part-02 \
    vllm-reasoning-study-full-evidence.zip.part-03 \
    vllm-reasoning-study-full-evidence.zip.part-04 \
    vllm-reasoning-study-full-evidence.zip.part-05 \
    vllm-reasoning-study-full-evidence.zip.part-06 \
    > vllm-reasoning-study-full-evidence.zip

shasum -a 256 -c <<'SHA256'
5fd8bac621e6c36dc58e2748e75046ebb8ace4ffa4fbef021abfc3eba98bc6a0  vllm-reasoning-study-full-evidence.zip
SHA256

mkdir extracted
unzip -q vllm-reasoning-study-full-evidence.zip -d extracted

cd extracted/vllm-reasoning-study
shasum -a 256 -c MANIFEST.sha256
```

Verify all six piece checksums first, concatenate in the explicit 01-to-06 order, and verify the reconstructed archive before extracting. Then verify the payload manifest before running anything. Run the manifest check inside the extracted project: its paths describe the original delivery, not the later publication checkout. A missing piece or failed verification is a reason to stop and obtain a matching copy, rather than editing the checksums to accept changed evidence.

## Regenerate the saved analysis

Continue from the extracted project directory above. Keep the verified reconstructed ZIP as the unchanged reference snapshot; these commands write regenerated JSON, figures and reports only inside the extracted working copy.

```sh
python3.11 -m venv work/analysis-env
work/analysis-env/bin/python -m pip install -e '.[test,report]'
work/analysis-env/bin/python -m pytest -q
work/analysis-env/bin/python scripts/analyze_continuation.py
work/analysis-env/bin/python scripts/analyze_stability.py
```

Inspect `results/continuation/analysis/summary.json`. A completed analysis has `report_status` equal to `complete` and no entries in `completion_blockers`; successful process exit alone is insufficient. The analysis validates existing records and does not launch a serving job.

After that check:

```sh
work/analysis-env/bin/python scripts/build_continuation_report.py
work/analysis-env/bin/python scripts/render_continuation_pdf.py
```

The builders regenerate `REPORT.md` and `report.pdf`. Regeneration can change timestamps and derived-file bytes, so the original manifest is intended to verify the snapshot **before** analysis. Preserve original raw journals, model artifacts, plans, configuration and execution source. For the recorded macOS environment or a new live replication, read [reproduction.md](reproduction.md).

## CPU-only uncertainty replay

Use a fresh output path; the script deliberately refuses to overwrite the saved result:

```sh
work/analysis-env/bin/python scripts/ablate_uncertainty.py \
  --observations results/continuation/search/*/observations.jsonl \
  --budget-ledger results/measured/budget.jsonl \
  --study configs/study-cloud.json \
  --output work/uncertainty-replay-new.json
```

This fits small surrogate models on the already measured table. Replay costs describe reused observations; they are not new accelerator charges or independent hardware measurements. The archived outcome is ceiling-limited because its initial designs already contain the table maximum. Re-running this diagnostic does not extend the original live campaign or change its conclusions.
