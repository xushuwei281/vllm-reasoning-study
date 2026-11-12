# Documentation index

Start with the [study report](../REPORT.md) for the question, experimental design, results and limitations. The [PDF](../report.pdf) is convenient for reading and the [Word version](../report.docx) has editable text and tables.

## Reproduce and inspect

| Document | Purpose |
|---|---|
| [GitHub evidence release](github-release.md) | Download, verify and extract the complete original snapshot; understand which evidence is kept in Git |
| [Reproduction guide](reproduction.md) | Isolated CPU environment, saved-analysis commands, serving prerequisites and historical replication |
| [Recovery](recovery.md) | Preserve failed runs and pending proposals; reconcile interrupted work without duplicating measurements or charges |
| [Interfaces](../INTERFACES.md) | Boundaries between collection, evaluation, optimization and resource accounting |

The repository contains the small evidence needed to inspect the completed findings. The release archive additionally contains all three raw-run directories, including streamed responses, scoring, telemetry and server metrics. Use a separate extracted copy for full analysis regeneration; do not merge an archive over an existing working tree.

## Understand the method and protocol

| Document | Purpose |
|---|---|
| [Surrogate](surrogate.md) | Neural performance model, feature/target transformations, acquisition and saved training artifacts |
| [Continuation protocol](continuation_protocol.md) | Recorded design decisions, empirical execution gate, separate statistical quality claim and fixed-allocation final test |
| [Cloud execution](cloud_execution.md) | Actual deployment, runtime, historical authorization and conservative terminal accounting |
| [Capabilities](capabilities.md) | Environment checks and historical platform findings |

The continuation protocol preserves decisions made before outcomes and their chronology. Some early planning alternatives were not adopted. The final adopted settings are in [study-cloud.json](../configs/study-cloud.json), the [search plan](../results/continuation/search-plan.json), the [final freeze](../results/continuation/freeze.json) and the [analysis specification](../results/continuation/analysis-spec.json). Use the completed report and audited evidence to distinguish plans from executed measurements.

## Reviews and provenance

| Document | Purpose |
|---|---|
| [Evaluation review](continuation_evaluation_review.md) | Cohort separation, paired quality checks, full-cohort scoring, final raw audit and limits of interpretation |
| [Machine-learning review](continuation_ml_review.md) | Actual learned proposals, information available at selection time and model limitations |
| [Systems review](continuation_systems_review.md) | Runtime, resource accounting, completion evidence and platform scope |
| [Contributions](contributions.md) | Implementation and automated review contributions, including dated historical stages |
| [References](references.bib) and [systems sources](systems_sources.md) | Research and primary technical sources |

These are automated engineering and experimental-design reviews, not claims of completed human peer review. Earlier reports in [results/prior-stage](../results/prior-stage/) document local development and are not pooled with the CUDA comparison. Explicitly synthetic artifacts in [results/synthetic](../results/synthetic/) validate software behavior and do not support measured performance claims.

## Key evidence files

- [Audited analysis summary](../results/continuation/analysis/summary.json): completion state, search results, paired held-out effects and accounting.
- [Original resource ledger](../results/measured/budget.jsonl): all measured reservations and allocation overhead.
- [Terminal accounting](../results/continuation/cloud-accounting-final.json): confirmed cancellation and conservative charge through the terminal observation.
- [Independent validation](../results/continuation/independent-validation-evidence.json): measured reference/candidate results and separate empirical/statistical conclusions.
- [Held-out summaries](../results/continuation/final/): frozen four-arm results and per-question outcomes.
- [Neural proposal artifacts](../results/continuation/search/neural/models/): four fitted ensembles, prior observations, training histories and integrity manifests.

Follow the [release guide](github-release.md) for checksums and the original snapshot. Publication documentation and packaging can evolve independently of frozen execution code and measured records.
