"""Test the exact Gemini failure pattern from user's log."""
import sys
sys.path.insert(0, ".")
from core.llm_client import LLMClient

# Exact text from the log (properly escaped \n and \" in the JSON string)
text = (
    '{"tool_call": {"name": "write_file", "input": {"path": '
    '"src/components/ui/BaseInput.vue", "content": '
    '"<script setup lang=\\"ts\\">\\n'
    'interface Props {\\n'
    '  label: string;\\n'
    '  modelValue: string;\\n'
    '  type?: string;\\n'
    '  placeholder?: string;\\n'
    '  error?: string;\\n'
    '}\\n\\n'
    'const props = withDefaults(defineProps<Prop'
)

tc = LLMClient._extract_gemini_tool_call(text)
if tc:
    print(f"OK: name={tc.name}, content_len={len(tc.input.get('content', ''))}")
    print(f"Content preview: {tc.input.get('content', '')[:80]}...")
else:
    print("FAIL: returned None")
