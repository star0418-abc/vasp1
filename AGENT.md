# AGENTS.md

## 0) What this repository optimizes for
This repo is a research-grade, reproducibility-first toolkit for preparing, running, and analyzing VASP AIMD workflows (and related post-processing).
Priority order:
1) Physical correctness / credibility
2) Reproducibility
3) Safety against silent failure
4) Operator ergonomics (clear logs, predictable CLI)

If a change improves convenience but risks physical correctness or reproducibility, reject it.

---

## 1) Mandatory workflow (every change)
1) Read first
   - Read README.md (tool usage and pitfalls).
   - Read AGENT_SKILLS/00_READ_FIRST.md (guardrails).
2) Do not introduce silent behavior
   - Any fallback or degraded mode must:
     - print a clear warning, and
     - be recorded in model_meta.json.
3) Update docs
   - Also update related README to latest whenever behavior/flags/outputs change.
4) No leftover test artifacts
   - If you create any TEST or temporary validation files during development, delete the TEST-related files after code modifications are complete.

---

## 2) Code style
### Python
- Target Python 3.10+.
- Use type hints for public functions and data structures.
- Prefer explicit names over abbreviations.
- Keep functions single-purpose; avoid deeply nested logic.
- Never swallow exceptions silently. Either:
  - raise with context, or
  - catch and emit a clear warning + structured metadata.

### Shell
- Use set -euo pipefail where appropriate.
- Quote variables defensively.
- Any destructive command must have:
  - a dry-run mode, or
  - an explicit allow flag, plus a clear prompt/log.

---

## 3) CLI compatibility and behavior
- Do not change existing flag meanings casually.
- If a default must change for safety:
  - document the behavior change in README,
  - provide a flag to restore legacy behavior,
  - record the chosen policy in model_meta.json.

---

## 4) Physics-critical guardrails (non-negotiable)
### 4.1 Cut bonds (dangling covalent bonds)
- Default behavior must prevent radicals.
- “Warn-only and continue” is not an acceptable default if dangling bonds remain.
- Preferred safe behavior:
  1) heal: expand selection to include missing bonded partners / connected components
  2) if still cut: error unless user explicitly overrides
- If hydrogen capping is implemented:
  - keep it conservative, fully report added atoms, and record in metadata.

### 4.2 Neutralization
- Never neutralize by selecting single atoms by distance.
- Use:
  - residues when residue info exists,
  - connected components (bond graph) when residue info is missing.
- Always record:
  - method_used, added_components, remaining_charge, verified.

### 4.3 PBC / MIC correctness
- Distinguish:
  - valid-for-3D-bulk-density (pbc.all() + valid volume)
  - valid-for-MIC-distances (may be partial PBC but must be explicit)
- Cluster + vacuum box:
  - do not use MIC for distance/clash checks
  - ensure cluster.pbc=[False, False, False] unless explicitly justified.

### 4.4 Density sanity checks
- Only enforce density bounds for true 3D periodic bulk.
- For non-3D systems, print “skipped” rather than hard-failing.
- Provide configurable thresholds and strictness.

### 4.5 Fallbacks must remain correct
- If optional connectivity utilities are missing, do not degrade into an empty bond graph.
- Provide a working internal bond-graph fallback (neighborlist + covalent radii).

---

## 5) Output contracts (must be stable)
### 5.1 model_meta.json (minimum required fields)
- command_line, timestamp, script_version (if available)
- input: path, format, optional hash
- selection:
  - center_index_original
  - center_index_cluster
  - selected_atom_count
  - allow_exceed_max_atoms
- cut_bonds:
  - policy
  - initial_n, final_n
  - healed / capped
  - report_path
- neutralization:
  - enabled
  - method_used
  - added_components (id, charge, size, distance metric)
  - remaining_charge
  - verified
- pbc_mic:
  - bulk_density_valid
  - mic_valid
  - cluster_vacuum_mic_used
- warnings: array of strings

### 5.2 Index traceability
- If indices matter, support writing index_map.json (original ↔ cluster mapping).
- selected_indices.txt must clearly indicate whether indices are original or cluster.

---

## 6) Validation expectations
- Provide at least one minimal self-check for physics-critical changes:
  - cut-bond healing behavior
  - component-based neutralization fallback
  - MIC disabled in cluster+vacuum clash checks
- Print validation outputs clearly.
- Do not leave any TEST artifacts behind.

---

## 7) Documentation expectations
Whenever you add/modify:
- default behaviors
- flags
- output file formats
- failure modes and warnings
You must update the README section(s) accordingly, including examples.

---

## 8) What to do when uncertain
- Prefer failing explicitly with a helpful message and next steps.
- If multiple reasonable behaviors exist:
  - keep default conservative,
  - add an override flag,
  - record the choice in metadata.

---

## 9) Deliverable format for AI agents
When you finish:
1) Summarize what changed and why (tie back to the guardrails above).
2) List new/changed flags and compatibility notes.
3) Provide validation steps and results.
4) Confirm README updated.
5) Confirm TEST-related files removed.
