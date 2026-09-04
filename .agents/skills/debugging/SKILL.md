---
name: debugging
description: Diagnose a failure, regression, inconsistency, or unexpected runtime or numerical result and establish its cause. Use before implementing a fix when the cause is not already demonstrated.
---

# Debugging

Produce an evidence-backed diagnosis; change code only when the request includes a fix.

1. State the observed failure and the expected behavior or contract. Separate direct observations from reports and interpretations.
2. Reproduce with the smallest case that preserves the real failure. Record the relevant input, environment, revision, and output.
3. Locate the owning boundary and trace data or control flow through it. Compare a known-good case when one is genuinely compatible.
4. Form discriminating hypotheses and test them with focused inspection, instrumentation, or checks. Avoid broad rewrites while the cause remains uncertain.
5. Identify the root cause, contributing conditions, and the scope of affected behavior. If the evidence cannot distinguish remaining hypotheses, say what observation is missing.
6. When fixing, change the owning mechanism, add a regression check that fails for the demonstrated defect, and remove temporary instrumentation or artifacts.
7. Re-run the reproduction and the checks implicated by the changed boundary. Report the cause, evidence, fix if authorized, and residual uncertainty.

Do not convert a one-off symptom workaround into a maintained project rule unless the underlying contract requires it.
