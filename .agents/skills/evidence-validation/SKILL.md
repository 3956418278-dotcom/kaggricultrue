---
name: evidence-validation
description: Plan or review validation for a behavioral, correctness, performance, numerical, or empirical claim. Use when deciding whether evidence supports an important result, not as a mandatory checklist for every edit.
---

# Evidence validation

Organize validation around the decision the result must support.

- State the claim precisely, including its scope, conditions, and acceptance consequence.
- Map each material claim to direct evidence. Use runtime observation for runtime behavior, contract-focused tests for correctness, measurement for performance, and appropriate numerical, statistical, or experimental checks for empirical claims.
- Use compatible inputs and conditions for comparisons unless the difference itself is under study. Keep illustrative computations separate from inferential evidence.
- Test the assumptions that could invalidate the conclusion. Prefer focused challenges at the claim's weakest boundary over unrelated broad checks.
- Distinguish failures of implementation, measurement, method assumptions, and evidence coverage. A check that cannot establish a claim is not evidence that the opposite is true.
- Report what passed, what failed, what was not tested, and what uncertainty remains. Calibrate wording to the strength and independence of the evidence.
- Preserve evidence and provenance only when they will be needed to reproduce or revisit the decision; keep generated validation output in the repository's designated data or cache area.

Validation is complete when the evidence is sufficient for the decision at hand, not when every available check has been run.
