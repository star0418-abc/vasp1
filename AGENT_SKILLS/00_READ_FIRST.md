# 00_READ_FIRST

## Before anything
- Read `AGENTS.md` first (repository root, if present) and follow it strictly: coding style, file layout, naming, PR rules, and validation expectations.
- This toolkit exists to produce **reproducible** and **physically credible** AIMD/VASP workflows where failures are **explicit and diagnosable** (no silent degradation).

## First-principles goals
1) **Do not create radicals by default**
   - If a cut/selection produces **dangling bonds** (cut covalent bonds), AIMD can immediately diverge into non-physical reactions.
   - A “warn-only and continue” default is unsafe. The safe default must **prevent** radicals:
     - Prefer expanding the selection to keep molecules/connected components intact.
     - If dangling bonds remain, **fail hard** unless the user explicitly overrides.

2) **Neutralization must never grab single atoms**
   - Selecting “nearest atoms” to balance charge can pull atoms from an anion fragment or a different intact molecule, breaking stoichiometry.
   - Neutralization must operate on **whole residues** (when residue metadata exists), and fall back to **whole connected components** (molecules) derived from a bond graph when residue metadata is missing.
   - Always record and surface `remaining_charge` and a post-neutralization verification result.

3) **PBC/MIC rules must match the physical boundary**
   - “3D bulk density” logic applies only to truly periodic **3D** systems (`pbc.all()==True` with a valid cell).
   - Cluster + vacuum box is not a periodic system for distance/clash purposes; avoid MIC wrap artifacts across vacuum.
   - Always distinguish:
     - *valid for bulk density* vs
     - *valid for MIC distances* (and specify which PBC directions are active).

4) **Fallbacks must not silently disable correctness**
   - If an optional dependency is missing (e.g., a connectivity utility), the script must not degrade into an “empty bond graph” that disables protection.
   - Provide an internal fallback (neighborlist + covalent radii) so cut-bond detection, connected components, and bond-hop expansion still work.

## Non-negotiable delivery requirements (for any agent/PR)
- **After code modifications are complete, delete the TEST-related files** (and any temporary validation artifacts created during development).
- **Also update related README to latest** to reflect behavior changes, defaults, new flags, and validation guidance.

## Default posture
- Prefer **safe, explicit failure** over producing outputs that look valid but are physically wrong.
- Every “override” must be deliberate: gated behind a clear CLI flag and recorded in metadata/logs.
