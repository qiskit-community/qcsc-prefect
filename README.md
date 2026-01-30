# HPC-Agnostic Workflow Execution Design (Revised Specification)

This document specifies an **HPC-agnostic workflow execution architecture**
that combines **admin-defined execution profiles** with **user-adjustable tuning parameters**.

The design allows users—who already have basic HPC knowledge—to flexibly adjust
resource usage (nodes, memory, threads) while keeping HPC-specific complexity
safely encapsulated in centrally managed profiles.

---

## 1. Design Goals

- Enable **one workflow** to run on multiple HPC systems (Fugaku, Miyabi, Slurm)
  without modification
- Reduce user burden by eliminating the need to write HPC-specific job blocks
- Preserve **user autonomy** for common resource tuning (nodes, memory, threads)
- Centralize scheduler-, MPI-, and filesystem-specific knowledge
- Prevent configuration errors while avoiding excessive restrictions

---

## 2. Core Design Principle

> **Separate intent from resolution.**

- Users express **what to run** and **how much resource they want**
- Admin-defined profiles resolve **how that intent maps to a specific HPC system**
- Workflow code never embeds HPC-specific details

---

## 3. Architectural Overview

```
[ Workflow ]
   │  (algorithm logic + run-time parameters)
   ▼
[ Command Block ]
   │  What to run (logical executable name)
   ▼
[ Execution Profile Block ]
   │  Recommended baseline configuration
   ▼
[ Tuning Parameters ]   ← user-adjustable (optional)
   │
   ▼
[ HPC Profile Block ]
   │  Scheduler / MPI / filesystem resolution
   ▼
[ Executor ]
```

---

## 4. Block Types and Responsibilities

### 4.1 Command Block (HPC-agnostic)

**Purpose**
- Define *what* command is executed

**Characteristics**
- Logical executable name only
- No absolute paths
- No scheduler or resource logic

**Examples**
- `cmd-diag`
- `cmd-sbd`

---

### 4.2 Execution Profile Block (Admin-defined baseline)

**Purpose**
- Provide a **recommended execution baseline** for a specific command

**Characteristics**
- Command-specific
- Safe, prevalidated defaults
- Intended to cover common use cases

**Typical contents**
- resource class (`cpu` / `gpu`)
- default node count
- default walltime
- default launcher
- default MPI hints
- default modules / environment variables

**Examples**
- `exec-diag-n2`
- `exec-diag-n16`
- `exec-diag-gpu`

> Execution Profiles express **recommended intent**, not fixed hardware allocations.

---

### 4.3 HPC Profile Block (Admin-defined, environment-specific)

**Purpose**
- Resolve execution intent into concrete HPC-specific settings

**Responsibilities**
- Scheduler type (PJM / PBS / Slurm)
- Batch template and submission logic
- Resource-class → queue / resource-group mapping
- MPI option derivation and validation
- Executable path resolution
- Enforcement of system limits
- Reference to UserContext Block

**Examples**
- `hpc-fugaku`
- `hpc-miyabi`

---

### 4.4 UserContext Block

**Purpose**
- Resolve execution identity information

**Responsibilities**
- Map `hpc_identity` → group / account / project
- Absorb differences between HPC environments
- Eliminate manual specification of group/account by users

---

## 5. Run-Time Parameters Specification

### 5.1 Required Parameters

```
hpc_target: string
  - e.g. "fugaku", "miyabi"

hpc_identity: string
  - user identifier (e.g. "z30541")

exec_profile: string
  - baseline execution profile (e.g. "exec-diag-n16")
```

---

### 5.2 Algorithm Parameters

Algorithm-specific inputs (files, numerical parameters, etc.)
are passed unchanged and are outside the scope of this specification.

---

### 5.3 Optional Tuning Parameters (User-adjustable)

Users may optionally provide **tuning parameters** to adjust resource usage
relative to the selected execution profile.

```yaml
tuning:
  nodes: int
  walltime: string
  ranks_per_node: int
  threads_per_rank: int
  mem_gib: int
```

#### Design Intent
- Tuning parameters allow **lightweight customization**
- Users can adapt execution to problem size without creating new profiles
- HPC Profile performs final validation and adjustment

---

## 6. What Users Can and Cannot Tune

### Allowed User Tuning

- Node count
- Walltime
- MPI rank / thread layout
- Requested memory (if supported by the HPC system)

### Not User-Tunable (Profile-controlled)

- Queue / resource group / partition
- Scheduler directives
- Raw MPI command-line options
- Executable absolute paths

> Users tune **resource quantities**, not **scheduler mechanics**.

---

## 7. Resolution and Priority Rules

Final execution configuration is resolved using the following priority order:

1. **User tuning parameters**
2. **Execution Profile defaults**
3. **HPC Profile defaults**
4. **System hard limits**

Additional rules:
- If `nodes` is explicitly set, it is respected
- If `nodes` is unset and `mem_gib` is provided, HPC Profile may estimate nodes
- Invalid or inconsistent configurations result in a validation error

---

## 8. Example: `exec-diag-n16` with User Tuning

### User Input

```yaml
exec_profile: exec-diag-n16
tuning:
  nodes: 32
  threads_per_rank: 4
```

### Resolution

- Execution Profile provides baseline intent (`diag`, CPU, large-scale)
- User tuning overrides node count and threading
- HPC Profile:
  - Validates against Fugaku/Miyabi limits
  - Derives MPI options
  - Selects appropriate queue/resource group
  - Resolves executable path

Workflow code remains unchanged.

---

## 9. Coverage and Flexibility

- Predefined execution profiles typically cover **70–90%** of use cases
- Tuning parameters extend coverage to **90–98%**
- Rare or experimental cases are handled by:
  - Admin-reviewed custom profiles
  - Advanced, gated override mechanisms

This avoids profile explosion while preserving flexibility.

---

## 10. Responsibility Split

### Administrators
- Define and maintain:
  - HPC Profiles
  - Execution Profiles
  - UserContext mappings
- Encode best practices and policies
- Control safety and validation rules

### Users
- Write workflow logic
- Select execution profiles
- Adjust tuning parameters as needed
- Focus on algorithm development

---

## 11. Key Benefits

- Fully HPC-agnostic workflows
- Reduced user overhead without loss of control
- Safe customization for experienced users
- Centralized operational knowledge
- Scalable long-term maintenance

---

## 12. Summary

> This architecture replaces user-written HPC job definitions with
> centrally managed execution profiles, while still allowing users
> to tune resource usage in a controlled and transparent way.

**Workflows describe algorithms.  
Profiles resolve execution.  
Tuning provides flexibility.**

