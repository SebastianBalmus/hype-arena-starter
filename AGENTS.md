# Hype Arena development

For inference, extraction, evidence, generation, or runtime changes, read
`.agents/skills/hype-arena-evidence-policy/SKILL.md` and the relevant repository contracts.
The evidence-policy skill preserves the user's supplied pipeline constraints.

Project-local skills under `.agents/skills/` support specific tasks:

- `scikit-learn`: optional offline extraction classifiers and evaluation.
- `property-based-testing`: parser, normalization, numeric and evidence invariants.
- `python-testing-patterns`: example-based tests and CLI/API integration tests.
- `python-performance-optimization`: profiling and constrained-runtime benchmarks.
- `docker-build-strategies`: container packaging and build optimization.

Load only skills relevant to the change. Their general examples do not override
Hype Arena's contract or the user's evidence policy. Do not add libraries merely
because a skill demonstrates them. Skills are development resources, not runtime
requirements; package only resources needed by inference into the final image.

Upstream commits and file hashes are recorded in `.agents/skills/upstream-lock.json`.

# Instructions for agents working on Hype Arena methods

Before implementing, training, evaluating or comparing a method, read and follow
[EVALUATION.md](EVALUATION.md).

- Use the frozen shared splits in [evaluation/splits-v1.json](evaluation/splits-v1.json).
- Rebuild each method's assets using only the evidence allowed by its evaluation split and policy.
- Save each experiment in its own directory; preserve the reference baseline results.
- Report unsupported or unverified capabilities explicitly. Do not claim generated truth from self-validation.
- Compare methods only under matching benchmark versions, splits, evidence access and decision policies.

The task README and `contracts/` govern competition submissions. `EVALUATION.md`
governs our local comparisons; it does not replace the organizer's rules.
