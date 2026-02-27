# 07_PROMPT_TEMPLATES_EN

## Template: Patch a physics-critical script safely
```text
Read AGENTS.md first and follow it.

Goal
- Patch <SCRIPT_NAME> to eliminate silent failure and enforce physically safe defaults.

Non-negotiables
- No silent fallback that disables bonding protection.
- No “warn-only” default when the result can create radicals or wrong stoichiometry.
- After code modifications are complete, delete the TEST-related files.
- Also update related README to latest.

What to do
1) Map failure modes to exact functions/regions in the code.
2) Implement minimal safe fixes with clear CLI flags for overriding legacy behavior.
3) Add lightweight self-check(s) (toy ASE objects) to validate new behavior.
4) Update README: defaults, flags, and “how to interpret warnings/errors”.

Deliverables
- Diff summary (what changed where)
- New flags + backward compatibility notes
- Validation commands and outputs
