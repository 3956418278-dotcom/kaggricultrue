---
name: independent-review
description: Challenge an important implementation, architectural decision, or empirical conclusion from a fresh context before it is accepted. Use when independence would materially improve confidence, not for routine self-review.
---

# Independent review

Review the result from evidence rather than from the producing context's narrative.

1. Define the result under review, the decision it would justify, and the contracts it must preserve.
2. Create an isolated review context when the available tooling and permissions allow it. Provide the claim, relevant authoritative documents, changed artifacts or raw evidence, and acceptance criteria. Omit the intended verdict, prior reasoning trail, and suspected defects unless needed to reproduce the result.
3. Ask the review to reconstruct the relevant behavior or analysis, inspect boundary assumptions, seek counterexamples or alternative explanations, and identify missing evidence.
4. Require concrete findings tied to files, outputs, calculations, or reproducible checks. Separate blocking defects, qualifications, and optional improvements.
5. Reconcile the independent findings in the main context. Resolve material contradictions or preserve them as uncertainty before accepting the result.
6. Promote only the accepted conclusion and its important limitations into project knowledge. Keep the detailed review trail outside the main context unless it remains decision-relevant.

Independence strengthens evidence; it does not transfer the final acceptance decision away from the main project context or the user.
