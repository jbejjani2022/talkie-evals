# Sources and attribution

- Base models and original Talkie implementation: [Talkie](https://github.com/talkie-lm/talkie),
  [Vintage model card](https://huggingface.co/talkie-lm/talkie-1930-13b-base),
  [Web model card](https://huggingface.co/talkie-lm/talkie-web-13b-base).
  Both model cards identify Apache-2.0. Downloads retain model cards.
- Tulu 3 mixture: [AllenAI dataset card](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture),
  identifying ODC-BY. Preserve attribution and consult its component-source
  terms; this package downloads Tulu from the pinned upstream revision.
  The full card is copied to `data/upstream/TULU3-README.md`.
- TRAIT: [official repository](https://github.com/pull-ups/TRAIT),
  *TRAIT: Personality Testset designed for LLMs with Psychometrics*.
  The upstream readme/citation is preserved in `data/upstream/TRAIT-README.md`.
- Initial Vintage SFT mixture and custom experimental code: supplied from
  the Talkie experiment workspace for this student collaboration, with source
  hashes and lineage. The source workspace has no top-level LICENSE file;
  no blanket open-source license is invented for this handoff.

The pinned revisions, exact filenames, checksums, token accounting and original
campaign paths are retained in `data/manifest.json`, `data/provenance/`,
`data/reference-runs/` and the adapter inventory. Do not treat historical
absolute paths as required local installation paths.
