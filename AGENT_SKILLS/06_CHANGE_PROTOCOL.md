# 06_CHANGE_PROTOCOL

## 1) Recommended change order (minimize risk)
1) Fix silent failure: bond graph fallback in no-utils mode
2) Fix unsafe default: cut-bond heal/error policy
3) Fix neutralization correctness: component-based fallback
4) Fix boundary conditions: separate PBC/MIC vs bulk density logic
5) Improve reproducibility: metadata, index map, reports

## 2) Every PR must include
- Change summary: what functions/paths were modified
- Behavior changes: defaults, new flags, backward compatibility notes
- Validation evidence: based on `05_VALIDATION_PLAYBOOK`
- Documentation: update related README to latest

## 3) Failure posture
For physically risky states (dangling bonds, nonzero remaining charge, PBC/MIC mismatch):
- default should be explicit failure or require a deliberate override flag
- never “silently continue”
