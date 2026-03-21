import re
pat = re.compile(r'^(?P<file>[^(\n]+)\((?P<line>\d+),(?P<col>\d+)\):\s*error\s+(?P<code>TS\d+):\s*(?P<message>.+)$')
vue_tsc = "src/components/feature/LoginForm.vue:48:13 - error TS2339: Property does not exist."
tsc = "src/components/Button.tsx(10,5): error TS2304: Cannot find name x."
print('vue-tsc format:', pat.match(vue_tsc.strip()))
print('tsc format:', pat.match(tsc.strip()))
