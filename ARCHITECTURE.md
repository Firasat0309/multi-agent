# Multi-Agent Backend Code Generator Architecture

## Purpose

This repository implements a deterministic multi-agent system for two workflows:

1. `generate`: create a new backend project from a natural-language prompt
2. `enhance`: modify an existing repository through targeted changes

The system is not built as a free-form "agents talking to each other" framework. It is a controlled orchestration layer with:

- explicit agent roles
- structured data contracts
- a per-file lifecycle state machine for generation
- a DAG for global tasks and modification workflows
- repository memory layers for syntax, semantics, and dependencies

## High-Level Design

At a high level, the system has four layers:

1. Entry points
   - CLI and API call into `Pipeline`
2. Orchestration
   - `RunPipeline` for new repos
   - `EnhancePipeline` for existing repos
   - `AgentManager` executes tasks
3. Specialist agents
   - architect, planner, coder, reviewer, tester, patcher, deployer, security, writer, repository analyzer, change planner
4. Memory and execution infrastructure
   - repository manager
   - repo index
   - dependency graph
   - embedding store
   - AST extractor
   - sandbox orchestration
   - event bus

## End-to-End Flow

### 1. Generate Flow

The `generate` workflow runs through four main phases:

1. Architecture design
   - `ArchitectAgent` turns the user prompt into a `RepositoryBlueprint`
2. Planning
   - `PlannerAgent` converts the blueprint into:
     - a `LifecycleEngine` for per-file work
     - a global `TaskGraph` for cross-repository phases
3. Execution
   - `AgentManager.execute_with_lifecycle()` runs file generation, review, build, test, and fix loops
4. Finalization
   - workspace indexing, dependency graph rebuild, and run report generation

The key design choice is that generation is not modeled as one static DAG. Instead:

- per-file work uses an event-sourced lifecycle state machine
- global work uses a smaller DAG after file lifecycles finish

### 2. Enhance Flow

The `enhance` workflow is different because the repository already exists:

1. Repository analysis
   - `RepositoryAnalyzerAgent` scans the repo and summarizes modules, entry points, and tech stack
2. Change planning
   - `ChangePlannerAgent` produces a structured `ChangePlan`
3. DAG construction
   - `ModificationTaskGraphBuilder` orders file edits safely
4. Execution
   - `PatchAgent` performs surgical edits where possible
   - `CoderAgent` falls back to full-file rewrite when needed
5. Finalization
   - workspace re-index, diff stats, and modify report

Unlike generation, enhancement uses a classic task DAG because the work is centered on file modifications, not file-by-file creation lifecycles.

## Core Runtime Components

### Pipeline Layer

- `core/pipeline.py`
  - public facade
  - owns LLM client, live console, and run-level logging
- `core/pipeline_run.py`
  - generate workflow
- `core/pipeline_enhance.py`
  - enhancement workflow

### Agent Execution Layer

- `core/agent_manager.py`
  - maps task types to agents
  - builds fresh context per task attempt
  - enforces file locks
  - executes tasks concurrently with a semaphore
  - updates embeddings incrementally after writes
  - publishes events on the event bus

### Workspace and Memory Layer

- `core/repository_manager.py`
  - creates workspace structure
  - reads and writes files atomically
  - updates repo index
  - rebuilds dependency graph
- `memory/repo_index.py`
  - persistent symbol-level index of files, exports, imports, classes, functions
- `memory/dependency_graph.py`
  - graph of file-to-file dependencies and impact analysis
- `memory/embedding_store.py`
  - semantic search using ChromaDB + sentence-transformers
- `core/ast_extractor.py`
  - tree-sitter-backed AST stubs for compact structural context

## Agent Roles

### Architect Agent

`ArchitectAgent` is responsible for producing a `RepositoryBlueprint` from the user prompt.

Output includes:

- project name and description
- architecture style
- tech stack
- folder structure
- file blueprints
- architecture document

Each `FileBlueprint` contains:

- file path
- purpose
- explicit dependencies
- exported symbols
- language
- architectural layer

This blueprint is the contract for downstream agents.

### Planner Agent

`PlannerAgent` does not ask the LLM to invent a plan. It builds a runtime plan in code through `LifecyclePlanBuilder`.

It returns:

- `LifecycleEngine`
  - per-file generation lifecycle
- `TaskGraph`
  - global tasks like security, module review, architecture review, deploy, docs

### Coder Agent

`CoderAgent` is the main code writer.

It has three execution modes:

1. generate a new source/config file
2. fix a file after review/build/test feedback
3. modify an existing file

For source files, it uses an agentic tool loop:

- `read_file`
- `search_code`
- `find_definition`
- `list_files`
- `write_file`
- `apply_patch`

This lets it inspect interfaces before writing, rather than generating blind.

### Patch Agent

`PatchAgent` is used primarily in enhancement mode. It asks the model for a unified diff, validates it, applies it, and falls back to `CoderAgent` full rewrite if patching fails.

### Reviewer, Build, Test, Security, Writer, Deploy

- `ReviewerAgent`
  - reviews files, modules, and architecture
- `BuildVerifierAgent`
  - runs compilation/type-check steps for compiled languages
- `TestAgent` and `IntegrationTestAgent`
  - generate and run tests
- `SecurityAgent`
  - scans the repo for security issues
- `WriterAgent`
  - writes documentation
- `DeployAgent`
  - generates Docker/Kubernetes deployment assets

## How the Architect Agent Works

The architect path is:

1. User prompt enters `RunPipeline.execute()`
2. `ArchitectAgent.design_architecture()` sends a strict JSON-only prompt to the LLM
3. Response is parsed and validated
4. `validate_architecture_response()` enforces schema correctness
5. `_parse_blueprint()` converts validated data into `RepositoryBlueprint`
6. `RepositoryManager.initialize()` materializes:
   - workspace folders
   - `architecture.md`
   - `file_blueprints.json`
   - initial `dependency_graph.json`

Important implementation details:

- language is inferred from the prompt and corrected again from file extensions
- large architecture responses use a larger token budget and continuation logic
- blueprint dependencies are explicit, not inferred later by downstream agents
- the architect always includes build/config files such as `pom.xml`, `package.json`, `go.mod`, `Cargo.toml`, or `pyproject.toml`

Current strengths:

- strong structured output contract
- multi-language awareness
- downstream-friendly blueprint format

Current limitations:

- architecture quality depends heavily on prompt quality
- no self-critique or second-pass validation beyond schema validation
- no retrieval from language-specific architecture templates or prior successful blueprints
- dependencies are LLM-declared first, then corrected later by repo indexing and import analysis

## How the Coder Agent Works

The coder path is more sophisticated because it combines multiple context sources.

### Context Inputs Used by the Coder

For each task, `ContextBuilder` assembles a bounded context package:

1. file blueprint
2. architecture summary
3. direct dependency files
4. semantic search hits
5. dependency graph information
6. for modification tasks, downstream impacted files and related changes

### The Three Main Intelligence Sources

For the coder agent, the system currently uses:

1. Semantic memory
   - ChromaDB + sentence-transformers
   - finds conceptually related files
2. Structural index
   - repo index with exports/imports/classes/functions
   - supports symbol lookup and dependency-aware context
3. Dependency graph / DAG reasoning
   - blueprint dependencies for generation ordering
   - NetworkX dependency graph for enhancement ordering and impact analysis

If by "DAC" you meant the dependency graph or task DAG, that is the third major context/control source in the current design.

### AST Compression

The system avoids dumping whole source files into the model when possible.

`ASTExtractor` uses tree-sitter to produce signature-only stubs. Today this is implemented for:

- Java
- Python

For unsupported languages, the system falls back to truncation.

This is one of the most important token-efficiency features in the repository.

### Agentic Tool Loop

For source generation, `CoderAgent.execute_agentic()` works like this:

1. build task-specific prompt
2. call LLM with tool schema
3. let the model request reads/searches/listing
4. execute all tool calls concurrently
5. feed tool results back into the next LLM turn
6. require a `write_file` call before considering the task complete

This makes the coder closer to an IDE assistant than a single-shot code generator.

### Fix and Modify Paths

- review failures feed back into `FIX_CODE`
- build failures feed compiler output back into `FIX_CODE`
- test failures feed test output back into `FIX_CODE`
- large-file modifications attempt diff-based editing first
- small-file modifications default to full rewrite

## Lifecycle Engine and DAG Model

### Generation Lifecycle

Each file moves through:

`PENDING -> GENERATING -> REVIEWING -> BUILDING -> TESTING -> PASSED`

Failure paths route through:

`REVIEW_FAILED -> FIXING`

`BUILD_FAILED -> FIXING`

`TEST_FAILED -> FIXING`

The `LifecycleEngine` tracks:

- current phase per file
- fix trigger type
- retry/fix budgets
- event log for auditability

Important behavior:

- interpreted languages skip build verification
- config/deploy/test-layer files skip testing
- test fix loops can degrade to pass-with-warnings after retry budget exhaustion

### Global DAG

After per-file lifecycles finish, the system runs a smaller global DAG:

1. security scan
2. module review
3. module-fix tasks
4. integration tests
5. architecture review
6. deploy generation
7. docs generation

### Modification DAG

For enhancement, `ModificationTaskGraphBuilder` builds a dependency-aware DAG that:

- orders edits safely
- merges conflicting same-symbol edits
- expands affected tests
- serializes same-file edits
- uses dependency graph impact analysis when available

## Memory Model

The system has three distinct memory forms, each for a different purpose.

### 1. Repo Index

Used for:

- export/import lookup
- symbol discovery
- dependency info enrichment
- fast structured context

Built from:

- tree-sitter extraction when available
- regex fallback otherwise

### 2. Dependency Graph

Used for:

- generation ordering support
- enhancement ordering
- impact analysis
- cycle detection
- test selection heuristics

### 3. Embedding Store

Used for:

- semantic similarity search
- finding related code even when names do not match exactly

This combination is stronger than relying on only a vector DB or only static imports.

## Sandbox and Safety

Execution safety comes from:

- sandbox orchestration for build/test execution
- file locks to prevent concurrent write races
- atomic file writes
- workspace snapshots in enhancement mode for rollback on failure
- import validation after writes

## What Is Good About the Current Design

- clear separation between orchestration and agent specialization
- deterministic execution model
- strong use of structured data contracts
- good context-budget discipline
- semantic + structural + dependency-aware retrieval
- fallback paths for patch failure, unsupported AST languages, and truncated LLM output

## Capability Gaps and Recommended Improvements

### Highest Priority

1. Expand AST support beyond Java and Python
   - add Go, TypeScript, Rust, and C#
   - this will improve both repo indexing and coder context quality

2. Add blueprint validation beyond schema correctness
   - detect missing runtime entrypoint
   - detect impossible dependency directions
   - detect missing config/build/test files
   - detect layer violations before execution starts

3. Improve architect output with retrieval
   - maintain a library of language/framework-specific blueprint templates
   - feed known-good layouts into `ArchitectAgent`
   - reduce architecture drift and bad folder conventions

4. Strengthen dependency resolution
   - current rebuilt dependency graph matches imports to exports heuristically
   - add language-specific import resolvers for TypeScript, Go, Rust, Java, C#

### Medium Priority

5. Add coder self-check passes before write
   - syntax checklist already exists for some languages
   - extend this to framework-specific checks and import consistency checks

6. Improve semantic indexing granularity
   - current chunking is line/block based
   - add symbol-level chunking so search can return function/class-level hits

7. Add retrieval weighting by task type
   - generation should prioritize interfaces and sibling patterns
   - fixing should prioritize compiler/test failures and affected dependents
   - modification should prioritize downstream impact and nearby symbols

8. Add architecture-level evaluation
   - compare generated repo against the blueprint after execution
   - detect blueprint drift, missing files, and mismatched exports

### Lower Priority but Valuable

9. Add exemplar memory from successful runs
   - store successful blueprints, file patterns, and fix histories
   - use them as retrieval context for future runs

10. Replace heuristic test impact with language-aware dependency analysis
    - especially important for TypeScript and Java multi-module projects

11. Add richer planner metrics
    - per-phase retry counts
    - common failure clusters by language/framework
    - cost attribution per agent and phase

12. Add an explicit architecture critique agent or second architect pass
    - first pass: draft blueprint
    - second pass: review completeness, layering, and operability

## Recommended Improvement Order

If the goal is to improve output quality fastest, the best sequence is:

1. add AST extraction for more languages
2. add stronger pre-execution blueprint validation
3. add template/retrieval support for `ArchitectAgent`
4. improve language-specific dependency resolution
5. move semantic retrieval from file-level to symbol-level chunks

## Summary

This repository already has a solid core design:

- structured architect output
- deterministic planning
- lifecycle-driven generation
- DAG-driven modification
- semantic, indexed, and dependency-aware context for the coder

The biggest capability gains now will come from improving retrieval quality and structural validation, not from adding more agent personas.
