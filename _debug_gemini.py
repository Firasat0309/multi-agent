"""Reproduce the Gemini tool call parsing failure."""
import json
import sys
sys.path.insert(0, ".")
from core.llm_client import LLMClient

print("=== Testing fixed parser ===\n")

# Pattern C: Truncated write_file with escaped newlines
text_c = (
    '{"tool_call": {"name": "write_file", "input": {"path": '
    '"src/main/java/com/example/authsystem/dto/UserRegistrationDto.java", '
    '"content": "packaging=\\"com.example.authsystem.dto\\"\\n'
    'import java.io.Serializable;\\n\\n'
    'public class UserRegistrationDto implements Serializable {\\n\\n'
    '    private static final long serialVersionUID = 1L;\\n\\n'
    '    private Str'
)
result_c = LLMClient._extract_gemini_tool_call(text_c)
print(f"C) Truncated write_file (escaped newlines): {result_c and result_c.name}")
if result_c:
    print(f"   content length: {len(result_c.input.get('content', ''))}")

# Pattern D: Unescaped newlines + truncated
text_d = (
    '{"tool_call": {"name": "write_file", "input": {"path": '
    '"src/main/java/com/example/authsystem/dto/UserRegistrationDto.java", '
    '"content": "packaging=com.example.authsystem.dto\n'
    'import java.io.Serializable;\n\n'
    'public class UserRegistrationDto implements Serializable {\n\n'
    '    private static final long serialVersionUID = 1L;\n\n'
    '    private Str'
)
result_d = LLMClient._extract_gemini_tool_call(text_d)
print(f"D) Unescaped newlines + truncated: {result_d and result_d.name}")
if result_d:
    print(f"   content length: {len(result_d.input.get('content', ''))}")

# Pattern E: Properly JSON-escaped newlines (\\n literal in the text) + truncated
text_e = r'{"tool_call": {"name": "write_file", "input": {"path": "src/main/java/com/example/authsystem/dto/UserRegistrationDto.java", "content": "package com.example.authsystem.dto;\nimport java.io.Serializable;\n\npublic class UserRegistrationDto implements Serializable {\n\n    private static final long serialVersionUID = 1L;\n\n    private Str'
result_e = LLMClient._extract_gemini_tool_call(text_e)
print(f"E) Escaped newlines + truncated: {result_e and result_e.name}")
if result_e:
    print(f"   content length: {len(result_e.input.get('content', ''))}")

# Pattern F: list_files works fine (no truncation)
text_f = (
    '{"tool_call": {"name": "list_files", "input": {"directory": '
    '"src/main/java/com/example/authsystem"}}}\n'
    'package com.example.authsystem.dto;\n'
)
result_f = LLMClient._extract_gemini_tool_call(text_f)
print(f"F) list_files + trailing code: {result_f and result_f.name}")

# Pattern G: Full valid write_file (not truncated) — must still work
text_g = '{"tool_call": {"name": "write_file", "input": {"path": "test.java", "content": "public class Test {}"}}}'
result_g = LLMClient._extract_gemini_tool_call(text_g)
print(f"G) Full valid write_file: {result_g and result_g.name}")
if result_g:
    print(f"   content: {result_g.input.get('content', '')}")

print("\n--- Debug Pattern D (unescaped newlines) ---")
# The _try_fix_newlines function should handle this BEFORE repair
# Let's test it directly
text_d2 = (
    '{"tool_call": {"name": "write_file", "input": {"path": '
    '"src/main/java/com/example/authsystem/dto/UserRegistrationDto.java", '
    '"content": "package com.example.authsystem.dto;\n'
    'import java.io.Serializable;\n\n'
    'public class UserRegistrationDto implements Serializable {\n\n'
    '    private static final long serialVersionUID = 1L;\n\n'
    '    private Str'
)
# After fixing newlines, the truncation repair should kick in 
import re
content_pat = re.compile(r'"content"\s*:\s*"', re.DOTALL)
m = content_pat.search(text_d2)
print(f"content field found at: {m.start() if m else 'N/A'}")
# Manually fix newlines then repair
if m:
    val_start = m.end()
    fixed_parts = [text_d2[:val_start]]
    i = val_start
    while i < len(text_d2):
        ch = text_d2[i]
        if ch == "\\" and i + 1 < len(text_d2):
            fixed_parts.append(text_d2[i:i+2])
            i += 2
            continue
        if ch == '"':
            break
        if ch == "\n":
            fixed_parts.append("\\n")
        elif ch == "\t":
            fixed_parts.append("\\t")
        elif ch == "\r":
            fixed_parts.append("\\r")
        else:
            fixed_parts.append(ch)
        i += 1
    fixed = "".join(fixed_parts) + text_d2[i:]
    print(f"After newline fix: ...{repr(fixed[-80:])}")
    # Now try repair on the newline-fixed version
    repaired = LLMClient._repair_json_text(fixed)
    print(f"After repair: ...{repr(repaired[-80:])}")
    try:
        data = json.loads(repaired)
        print(f"Parsed OK! content_len={len(data.get('tool_call',{}).get('input',{}).get('content',''))}")
    except json.JSONDecodeError as e:
        print(f"Parse failed: {e}")

print("\nDone.")
