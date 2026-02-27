# 02_GUARDRAILS_PHYSICS

## 1) Cut bonds policy: default must prevent radicals
### Must satisfy
- Cut-bond detection must drive an action, not just a warning:
  - **heal (recommended)**: expand selection to include missing bonded partners / connected components
  - **error (safe fallback)**: if healing cannot eliminate cut bonds (or hits max), fail hard unless explicitly overridden
  - **warn**: only acceptable for cases where covalent cutting is impossible or user explicitly accepts risk
  - **cap (future optional)**: if hydrogen capping is implemented, it must be conservative and fully reported

### Forbidden
- Default behavior that continues after detecting dangling covalent bonds without repair.

## 2) Neutralization: forbidden to pick single atoms by distance
### Must satisfy
- If residue metadata exists: neutralize by residues (whole ions/molecules)
- If residue metadata missing: neutralize by **connected components** derived from a bond graph
- Ranking by distance is allowed, but selection unit is **component**, not atom
- Always surface:
  - `remaining_charge`
  - `verified` (post-neutralization check)
  - method used

### Forbidden
- “nearest atoms” selection to neutralize charge.

## 3) PBC/MIC rules must match physical boundary
### Two validations must be distinct
- `bulk_density_valid`:
  - requires `pbc.all()==True` and a valid 3D cell volume
- `mic_valid`:
  - may allow partial PBC, but must be explicit which directions are periodic

### Cluster + vacuum box
- A vacuum box is a convenience for file output/visualization, not a periodic model for distance checks
  - for clash detection and distance ranking: enforce `mic=False`
  - set `cluster.pbc=[False, False, False]` unless there is a deliberate, documented reason

## 4) Density sanity checks
- Only enforce density sanity bounds for true 3D periodic bulk
- Do not hard-fail slab/nanowire/cluster based on 3D density bounds
- Density sanity range should be configurable and/or strictness toggleable

## 5) Fallbacks must not disable correctness
- If connectivity utility is unavailable:
  - use internal bond-graph fallback (neighborlist + covalent radii)
  - cut-bond detection, connected components, and bond-hops must still function
- Any fallback must be loudly logged and recorded in metadata
