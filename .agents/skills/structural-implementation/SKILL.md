---
name: structural-implementation
description: Implement or refactor code, configuration, interfaces, or module boundaries while preserving established ownership and repository coherence. Use for substantive structural change, not a purely diagnostic request.
---

# Structural implementation

Start from the accepted contract and the component that owns the behavior.

- Determine whether the request adds a capability or only another instance, value, or dimension of an existing capability. Put variation in the owning configuration, data, argument, or adapter instead of duplicating the mechanism.
- Trace callers and consumers before changing an interface. Preserve stable boundaries unless changing them is necessary to solve the requested problem.
- Give shared mechanisms one clear owner. If multiple components copy a mechanism or depend on private details, repair that ownership boundary within the selected scope.
- Add abstraction only for a demonstrated shared concept. Keep task-local analysis and exploration lightweight until they become maintained capabilities.
- Implement one coherent path using real project formats and constraints. Keep unrelated working-tree changes intact.
- Align maintained code, configuration, tests, project memory, and packaged or generated representations that belong to the changed capability. Do not silently create a second source of truth.
- Validate the affected contract and observable behavior. Broaden checks only where the change creates a concrete regression risk.

Judge a structural repair by the resulting ownership and dependency direction, not by file movement or diff size.
