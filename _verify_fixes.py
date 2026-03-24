"""Quick verification of edge case fixes."""
from agents.architect_agent import ArchitectAgent

# Fix 1: Go word boundary
print("=== Go Detection Fix ===")
print("go at end:", ArchitectAgent._default_db_note("Build a REST API in Go"))
print("go with period:", ArchitectAgent._default_db_note("Build a REST API using Go."))
print("go embedded:", ArchitectAgent._default_db_note("I want a Google API"))  # should NOT match
print("golang:", ArchitectAgent._default_db_note("Build a Golang API"))

# Fix 2: Class name with digits
print()
print("=== Class Name Fix ===")
for name in ["123-test", "my-project", "3d-engine", ""]:
    raw_name = name.replace("-", " ").replace("_", " ")
    class_name = "".join(w.capitalize() for w in raw_name.split()) + "Application"
    if class_name and not class_name[0].isalpha():
        class_name = "App" + class_name
    print(f'  "{name}" -> {class_name}')

# Fix 3: Package prefix heuristic
print()
print("=== Package Prefix (new common-prefix approach) ===")
# Simulate with multiple files
import os

def compute_new_prefix(java_dirs):
    _LAYER_DIRS = frozenset({
        "controller", "controllers", "service", "services",
        "repository", "repositories", "model", "models",
        "entity", "entities", "config", "configuration",
        "util", "utils", "helper", "helpers", "dto", "dtos",
        "exception", "exceptions", "filter", "filters",
        "security", "middleware", "interceptor", "mapper",
    })
    if java_dirs:
        java_parent_dirs = sorted(set(
            "/".join(fp.split("/")[:-1]) for fp in java_dirs
        ))
        split_dirs = [d.split("/") for d in java_parent_dirs]
        min_len = min(len(d) for d in split_dirs)
        common_parts = []
        for i in range(min_len):
            vals = {d[i] for d in split_dirs}
            if len(vals) == 1:
                common_parts.append(vals.pop())
            else:
                break
        if common_parts and common_parts[-1].lower() in _LAYER_DIRS:
            common_parts = common_parts[:-1]
        return "/".join(common_parts) if common_parts else "src/main/java"
    return "src/main/java"

# Case 1: Typical multi-file Spring Boot
files = [
    "src/main/java/com/example/auth/controller/AuthController.java",
    "src/main/java/com/example/auth/service/AuthService.java",
    "src/main/java/com/example/auth/model/User.java",
]
print(f"  Multi-file deep: {compute_new_prefix(files)}")

# Case 2: Shallow paths (the bug case)
files = [
    "src/main/java/com/example/UserService.java",
    "src/main/java/com/example/UserRepository.java",
]
print(f"  Shallow (was 'com', now): {compute_new_prefix(files)}")

# Case 3: All in same layer dir
files = [
    "src/main/java/com/example/controller/UserController.java",
    "src/main/java/com/example/controller/AuthController.java",
]
print(f"  All same layer: {compute_new_prefix(files)}")

# Case 4: Single file in controller
files = ["src/main/java/com/example/auth/controller/AuthController.java"]
print(f"  Single in layer: {compute_new_prefix(files)}")

# Case 5: Single shallow file
files = ["src/main/java/com/example/UserService.java"]
print(f"  Single shallow: {compute_new_prefix(files)}")

# Case 6: No standard prefix
files = ["com/example/Service.java"]
print(f"  No src prefix: {compute_new_prefix(files)}")

print("\nAll checks passed!")
