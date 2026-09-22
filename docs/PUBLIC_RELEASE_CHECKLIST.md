# Public release checklist

Current state: **private pre-release**. Do not change repository visibility
until the authors coordinate the manuscript, arXiv, and code release.

## Complete

- Canonical strict-v2 Proposed implementation and frozen configuration.
- Next-close timing, target, accounting, and split tests.
- Docker and native Python setup instructions.
- Compact verified paper metrics, curves, protocol metadata, and audit reports.
- Baseline adaptation and checkpoint-selection documentation.
- Citation metadata.
- Automated artifact integrity and scope tests.

## Required before public visibility

- Obtain author/institution approval for a software license and add `LICENSE`.
- Confirm the final manuscript title, author spelling, and publication metadata
  in `CITATION.cff`.
- Re-run `python -m pytest` from a clean checkout.
- Verify that `artifacts/canonical_paper_export/SHA256SUMS.txt` matches every
  included artifact.
- Confirm no raw licensed market data, checkpoints, credentials, local absolute
  paths, or third-party source trees are tracked.
- Create the first version tag only after the manuscript version is frozen.

Changing repository visibility is deliberately not automated by this project.
