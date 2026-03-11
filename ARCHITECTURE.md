# Architecture Evolution

## Multi-Agent Backend Code Generation System

## 1. Purpose

This document explains how the architecture of this multi-agent backend code generation system evolved over time, what problems each version solved, and why the current design looks the way it does.

The system takes a natural-language request such as:

`Build a backend service for user management with PostgreSQL and JWT authentication`

and transforms it into a generated or modified repository through a set of specialized agents.

This evolution was not driven by theory alone. It was driven by implementation failures, orchestration bottlenecks, dependency-ordering issues, build and test reliability problems, and prompt-efficiency constraints.

The architecture evolved through three major stages:

| Version | Execution Model |
|---------|------------------|
| V1 | Static DAG-based orchestration |
| V2 | Per-file lifecycle + global DAG |
| V3 | Per-tier lifecycle + global DAG |

This document focuses primarily on the generation workflow, but it also clarifies how the same reasoning influenced the enhancement workflow.

## 2. System Overview

The system is a hybrid orchestrated multi-agent pipeline.

The orchestrator itself does not perform software-design reasoning. It controls sequencing, retries, dependency ordering, and validation gates. The actual reasoning is done inside the agents through LLM calls.

In practical terms, the orchestrator is:

- state-machine driven for lifecycle control
- graph-based for dependency ordering
- rule-based for retries, gating, skipping, and fallback behavior

The major agents in the generation pipeline are:

| Stage | Agent | Responsibility |
|-------|-------|----------------|
| Planning | `ArchitectAgent` | Interpret the user request and design the target repository structure |
| Planning | `PlannerAgent` | Build the execution plan from the blueprint |
| Implementation | `CoderAgent` | Generate or fix source/config files |
| Verification | `ReviewerAgent` | Review generated code for correctness and architecture adherence |
| Verification | `BuildVerifierAgent` or repo-level checkpoints | Validate that generated code compiles or type-checks |
| Testing | `TestAgent` | Generate and execute tests |
| Security | `SecurityAgent` | Perform security-oriented review |
| Integration | `IntegrationTestAgent` | Validate cross-module interactions |
| Deployment | `DeployAgent` | Generate Docker and Kubernetes artifacts |
| Documentation | `WriterAgent` | Generate README, API docs, and changelog |

The system supports two major operating modes:

- `generate`: create a repository from scratch
- `enhance`: modify an existing repository

This document focuses on how the generation architecture evolved, because that is where the orchestration model changed most significantly.

## 3. Version 1

## Static DAG-Based Orchestration

### Design

The first version of the system used a simple static Directed Acyclic Graph.

The execution model looked conceptually like this:

`Architect -> Coder -> Reviewer -> Tester -> Writer`

Each stage was represented as a node in a predefined DAG, and edges encoded a fixed execution order. The orchestrator executed nodes in topological order.

This model was attractive at first because it was simple:

- easy to implement
- easy to debug
- easy to visualize
- adequate for small prototypes

### Characteristics

V1 had the following properties:

- one static pipeline
- repository treated as a single unit
- little or no file-level lifecycle control
- sequential stage-oriented execution
- coarse-grained retry behavior

### Why It Failed

V1 broke down quickly as the system started generating real multi-file backends.

#### 1. Repository-Level Granularity Was Too Coarse

The system treated the whole repository as one work unit.

If one file had a problem, the effective fix path often required retrying large parts of the pipeline. That wasted tokens, time, and compute.

#### 2. The DAG Did Not Represent File Dependencies Well

Backend code is not just a sequence of stages. It is a dependency network:

- controller depends on service
- service depends on repository
- repository depends on model
- config may be needed before application entrypoints are valid

A stage-level DAG could say:

`Coder before Reviewer before Tester`

but it could not naturally express:

`Generate model before repository before service before controller`

for dozens of files.

#### 3. Failure Recovery Was Expensive

If review or tests failed, the system had no good way to isolate the failure to a specific file and run a local correction loop.

The result was broad retry behavior instead of targeted repair behavior.

#### 4. It Did Not Scale With Repository Size

Once the number of files increased, a single stage-wise DAG became too blunt an execution model. It did not provide the granularity needed for safe concurrency or localized recovery.

### Why V1 Was Still Useful

V1 was important because it established the agent decomposition:

- planning
- implementation
- verification
- testing
- deployment
- documentation

The failure of V1 was not that the agents were wrong. The failure was that the orchestration unit was too large.

## 4. Version 2

## Per-File Lifecycle + Global DAG

### Core Idea

The second version changed the unit of execution from:

`entire repository`

to:

`individual file`

Instead of sending the whole repo through a single pipeline, each file got its own lifecycle.

### Per-File Lifecycle

Each file moved through a state machine like:

`PENDING -> GENERATING -> REVIEWING -> FIXING -> BUILDING -> TESTING -> PASSED`

with loops such as:

- review failure -> fix -> review again
- build failure -> fix -> build again
- test failure -> fix -> test again

This was implemented as an event-sourced lifecycle engine.

That was a major architectural improvement, because it introduced:

- localized retries
- per-file auditability
- explicit lifecycle state
- a cleaner separation between success, failure, and repair

### Dependency Graph

At the same time, the system introduced a file dependency graph.

This graph allowed the orchestrator to reason about:

- which files depend on which others
- which files can start first
- which files can run in parallel
- which downstream files may be impacted by changes

This was a major improvement over the stage-only DAG in V1.

### Global DAG

V2 also recognized that some work is not naturally file-scoped.

Examples:

- security scan
- module review
- architecture review
- deploy artifact generation
- documentation generation

Those tasks remained in a smaller global DAG that runs after per-file lifecycles finish.

So V2 introduced a hybrid model:

- file-level lifecycle for implementation work
- global DAG for repository-wide work

### What V2 Fixed

V2 solved several major problems from V1.

#### 1. Fine-Grained Recovery

If one file failed review or tests, only that file entered a fix loop.

This was far more efficient than restarting large portions of the pipeline.

#### 2. Better Concurrency

Independent files could be processed in parallel once dependencies allowed it.

#### 3. Better Separation of Concerns

Per-file correctness was treated differently from repository-wide checks.

That was the right conceptual split.

### Why V2 Was Still Not Enough

V2 introduced a new problem: build systems do not usually validate code one file at a time.

#### 1. Build Verification Was Not Truly File-Scoped

The lifecycle engine was file-centric, but the build tools were usually repo-centric or module-centric.

For compiled languages, this created a mismatch:

- the current lifecycle said "verify this file"
- the build tool said "verify the current project state"

That meant unrelated incomplete files could cause the current file to fail build verification.

#### 2. Error Attribution Became Hard

If the workspace build failed, the system still needed to answer:

- which file actually caused the failure
- which dependent file needs repair
- whether the current file is at fault or an upstream type is missing

Per-file lifecycles alone were not a sufficient answer to repo-level build failures.

#### 3. Dependency Order Was Better, But Still Fragile

Even with dependency awareness, the system could still enter bad generation sequences where upstream files were only partially valid when downstream files began.

That caused cascading failures.

### Main Lesson From V2

V2 proved that the right unit was smaller than "whole repository", but also that "single file" is not always the right verification unit for backend systems.

This led to V3.

## 5. Version 3

## Per-Tier Lifecycle + Global DAG

### Core Idea

Version 3 changed the execution unit again.

Instead of validating each file independently all the way through build/test, the system groups files into dependency tiers and validates them incrementally.

A tier is a set of files at the same dependency depth.

Example:

- Tier 0: models, DTOs, config primitives
- Tier 1: repositories, interfaces, lower-level services
- Tier 2: higher-level services, controllers, handlers

### Why Tiers Were Introduced

Tiering solves the biggest weakness of V2:

build correctness is usually meaningful only after a coherent dependency layer exists.

If Tier 0 is invalid, Tier 1 should not proceed.

That is a better model than:

"build after each file"

because most backend projects do not become valid one file at a time. They become valid one dependency layer at a time.

### Tier Execution Model

The intended execution pattern is:

1. Generate and review all files in Tier 0
2. Run a repo-level or module-level checkpoint
3. Fix attributed failures
4. Only when Tier 0 is stable, start Tier 1
5. Repeat until all tiers are complete
6. Then run tests and global tasks

Conceptually:

`Tier 0 -> checkpoint -> Tier 1 -> checkpoint -> Tier 2 -> checkpoint -> tests -> global DAG`

### What V3 Improves

#### 1. Better Dependency Discipline

Foundational code is stabilized earlier.

That prevents many cascading downstream failures.

#### 2. Build Validation Happens at a More Meaningful Scope

Instead of pretending that a whole-project build belongs to a single file, the system uses checkpoint-based validation between dependency layers.

That better matches how real build systems work.

#### 3. Earlier Detection of Structural Problems

Type errors, interface mismatches, and missing contracts are caught before higher-level files multiply the damage.

### Important Clarification

V3 is architecturally better than V2 in principle, but only if it enforces the tier guarantees strictly.

For a per-tier architecture to work well, the system must ensure:

- downstream tiers do not proceed when upstream tiers are still invalid
- dependency readiness means "stable enough to depend on", not merely "started earlier"
- build failures are attributed back to affected files accurately

That is the direction the architecture is moving toward, and it is the correct direction for compiled multi-file backend generation.

### Main Lesson From V3

The correct unit of orchestration is not always the same as the correct unit of verification.

The system started with repository-wide orchestration, moved to file-level lifecycle control, and then evolved toward dependency-tier validation because backend systems become valid in layers.

## 6. AST-Based Context Compression

As the architecture matured, prompt-efficiency became a serious issue.

Originally, the system passed too much raw code into LLM prompts. That caused:

- high token usage
- irrelevant context
- slower execution
- poorer focus

To solve this, the system introduced AST-based structural extraction.

### What AST Means Here

AST stands for Abstract Syntax Tree.

Instead of passing full code files to the model, the system can parse a file and extract only the interface-relevant parts:

- imports
- class names
- method signatures
- field signatures
- inheritance and implemented interfaces
- public API surface

These extracted representations are turned into compact AST stubs.

### Why AST Was Introduced

The LLM often does not need the full implementation of a dependency file.

It usually needs to know:

- what exists
- what can be imported
- what methods can be called
- what types are exposed

AST stubs preserve that information while using far fewer tokens than raw source.

### Where AST Is Used

AST is mainly used during context construction for downstream agents:

- coder reading already-generated dependencies
- reviewer reading related files
- change planner reading bounded context in enhancement mode

AST does not help before any files exist. It becomes useful after the first files have been generated and the system starts reusing generated code as context for later work.

## 7. Semantic Retrieval and Repository Memory

The architecture also evolved beyond plain file reading.

Three forms of repository memory are now important:

### 1. Repo Index

A structured catalog of:

- file paths
- imports
- exports
- functions
- classes

This gives the system structural lookup across the repository.

### 2. Dependency Graph

A file-to-file dependency model used for:

- ordering
- impact analysis
- tier scheduling
- test targeting

### 3. Embedding Store

A semantic index that lets the system search code by meaning, not just by exact symbol match.

This helps the `CoderAgent` and `ChangePlannerAgent` retrieve conceptually related code even when filenames or symbol names do not match the user request directly.

Together, these three memory layers made the architecture much stronger than either plain file search or simple sequential prompting.

## 8. Execution Environment and Safety

The system also evolved operationally, not just logically.

Builds and tests run in sandboxed environments so generated code does not directly contaminate the host.

This improves:

- reproducibility
- security
- dependency isolation
- test reliability

The repository manager also introduced:

- atomic writes
- repository indexing
- dependency graph rebuilds
- workspace snapshots for safer enhancement workflows

These operational controls are part of the architecture, not incidental details.

## 9. State Management

Execution requires shared runtime state.

The system maintains structured execution state in memory, including:

- blueprint metadata
- lifecycle state
- task graph state
- repository index
- dependency graph
- embedding store updates
- event stream and execution metrics

The important architectural point is that agent work is not isolated conversation state. It is coordinated through explicit runtime structures.

That is one of the reasons the system behaves more like an orchestrated software pipeline than a generic chat-agent framework.

## 10. Architectural Lessons Learned

Several important lessons emerged from the evolution.

### 1. A Static DAG Is Not Enough

Static stage orchestration is too coarse for multi-file code generation.

### 2. File-Level Lifecycle Control Is Valuable

Localized retries and repairs dramatically improve efficiency and reliability.

### 3. Dependency-Aware Scheduling Is Mandatory

Any serious code-generation system must understand dependency relationships explicitly.

### 4. Build Verification Scope Matters

The right unit for code generation is not always the right unit for build verification.

### 5. Context Efficiency Directly Affects Quality

AST stubs, indexing, and semantic retrieval significantly improve downstream LLM performance.

### 6. Hybrid Orchestration Is the Right Model

No single model was sufficient.

The current architecture combines:

- state-machine lifecycle control
- graph-based dependency scheduling
- rule-based retry and gating policies

That hybrid design emerged because the problem itself is hybrid.

## 11. Current Architectural Summary

The current architecture can be summarized as:

- specialized LLM-backed agents
- orchestrated by a hybrid runtime
- using dependency-aware scheduling
- with lifecycle-based error recovery
- repo-level checkpoints
- AST-driven context compression
- semantic and structural repository memory
- sandboxed build and test execution

In practical terms, the architecture evolved like this:

- V1 taught that repository-wide orchestration was too coarse
- V2 taught that file-level lifecycle was necessary
- V3 taught that dependency-tier validation is the right next step for multi-file compiled backends

## 12. Final Conclusion

The architecture did not evolve by adding complexity for its own sake.

It evolved because each previous model failed at a specific boundary:

- V1 failed at granularity
- V2 failed at build-scope realism
- V3 emerged to align orchestration with dependency structure

The current design is therefore best understood as the result of successive constraint discovery.

It is not "a DAG system" or "a state machine system" in isolation.

It is a hybrid orchestration architecture for autonomous backend code generation, where:

- file and tier execution are dependency-aware
- validation happens at the correct scope
- global repository concerns are handled separately
- LLM context is compressed and structured rather than naive

That is the architectural reason the system looks the way it does today.
