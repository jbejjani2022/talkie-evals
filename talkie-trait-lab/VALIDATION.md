# Handoff verification

The handoff includes historical scientific results as references. The checks
below validate the portable implementation; they are not new scientific runs.

- Data/adapter export job **19642**: `COMPLETED`, `0:0`. Verified source hashes,
  compaction receipts, the 40 terminal exposure counts, and adapter archive CRCs.
- Isolated CPU setup job **19652**: `COMPLETED`, `0:0`. Installed `uv.lock`,
  passed the initial nine CPU tests, verified all bundled files, downloaded the
  pinned TRAIT JSON and regenerated the exact canonical prompt/choice rows,
  and reproduced Vintage SFT input tokens and assistant masks.
- GPU smoke job **19658**: `COMPLETED`, `0:0`, one B200, PyTorch 2.9.0+cu130.
  Loaded Vintage and Web atomic bases plus both production Tulu adapters;
  checked raw likelihood scores against an independent full-logit reference;
  trained two Vintage LoRA updates, validated/exported/reloaded the new adapter,
  scored bare/chat TRAIT smoke panels, ran baseline and paired N=8 Openness
  demonstrations, and completed the paired scenario-group analysis.
- CPU job **19667** (`COMPLETED`, `0:0`) adds exact Vintage/Web reconstruction for Vintage SFT,
  matched Tulu and PersonaHub IF, a tensor-by-tensor comparison of every optional
  adapter against its source checkpoint, and the expanded CPU test suite.
  All **15 CPU tests passed**. Completion and exact tensor audits are recorded in `data/validation/`.
- The final packaging job tests the copied standalone repository and an
  extracted zip, checks every extracted file against the repository, and records
  the repository commit and archive SHA256 alongside the archives.

The CPU tests cover zero-adapter parity and frozen weights, optimizer resume,
assistant loss/gradient equivalence, batch coverage, exact BPB/UTF-8/tie
semantics, paired few-shot controls and split disjointness, pinned download
arguments, BF16 conversion and remote Transformers reload, malformed checkpoint
rejection, base-family identity, and paired group bootstrap/interface checks.

Verification found two operational assumptions requiring correction: the
canonical tokenizer has no saved padding token (the portable loader now sets
EOS/left padding), and upstream Tulu repeats some hard-coded IDs (identical raw
records are deduplicated for lookup, while frozen selected occurrences and their
content hashes remain intact). Older adapter validation hashes refer to
pre-compaction checkpoints; their preserved compaction receipts identify and
validate the current model-only payloads.

Fresh 13B model downloads/conversions, all 40 complete training trajectories,
and all 961 few-shot conditions per base were not rerun for this handoff.
Conversion has a small synthetic full-weight/remote-loading test. GPU checks
use existing canonical atomic models. Other GPUs, drivers, precision modes,
distributed recipes and training seeds have not been validated here. Consult
the frozen reference outputs when checking a complete reproduction.
