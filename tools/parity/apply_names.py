# Sets DisplayName, Category and (where they differ and the output count matches) OutputNames in SpawnGen's [Node(...)] attributes to
# ComfyUI's, for every node both have.
#
#   python apply_names.py comfy_object_info.json spawngen_object_info.json <src folder> [--dry-run]
#
# ComfyUI shows a node without a display name under its id, so that becomes the display name. Only those properties are touched.
import json, os, re, sys

comfy = json.load(open(sys.argv[1], encoding="utf-8"))
ours = json.load(open(sys.argv[2], encoding="utf-8"))
src = sys.argv[3]
dry = "--dry-run" in sys.argv

wanted = {}
for node_id, s in ours.items():
    c = comfy.get(node_id)
    if not c or str(c.get("python_module", "")).startswith("comfy_api_nodes"): continue
    display = c.get("display_name") or node_id
    outputs = c.get("output_name") if c.get("output_name") != s.get("output_name") and len(c.get("output_name") or []) == len(s.get("output_name") or []) else None
    if s.get("display_name") != display or s.get("category") != c.get("category") or outputs:
        wanted[node_id] = (display, c.get("category"), outputs)

attr_re = re.compile(r'\[Node\((?P<args>[^\]]*?)\)\]\s*(?:\r?\n\s*\[[^\]]*\]\s*)*\r?\n\s*public (?:sealed |static )?class (?P<cls>\w+)', re.S)

def node_id(args, cls):
    m = re.match(r'\s*"([^"]+)"', args)
    return m.group(1) if m else cls

def set_list(args, name, values):
    lit = "[" + ", ".join(json.dumps(v, ensure_ascii=False) for v in values) + "]"
    pat = re.compile(name + r'\s*=\s*\[[^\]]*\]')
    if pat.search(args): return pat.sub(f'{name} = {lit}', args, count=1)
    return args.rstrip() + (", " if args.strip() else "") + f"{name} = {lit}"

def set_prop(args, name, value):
    lit = json.dumps(value, ensure_ascii=False)
    pat = re.compile(name + r'\s*=\s*"(?:[^"\\]|\\.)*"')
    if pat.search(args): return pat.sub(f'{name} = {lit}', args, count=1)
    return args.rstrip() + (", " if args.strip() else "") + f"{name} = {lit}"

changed = 0
for root, _, files in os.walk(src):
    if os.sep + "obj" in root or os.sep + "bin" in root: continue
    for f in files:
        if not f.endswith(".cs"): continue
        path = os.path.join(root, f)
        text = open(path, encoding="utf-8").read()
        def fix(m):
            global changed
            nid = node_id(m.group("args"), m.group("cls"))
            if nid not in wanted: return m.group(0)
            display, category, outputs = wanted.pop(nid)
            args = set_prop(set_prop(m.group("args"), "Category", category), "DisplayName", display)
            if outputs: args = set_list(args, "OutputNames", outputs)
            changed += 1
            return m.group(0).replace(m.group("args"), args, 1)
        new = attr_re.sub(fix, text)
        if new != text and not dry:
            open(path, "w", encoding="utf-8", newline="").write(new)
print(f"{changed} nodes updated; not found: {sorted(wanted)}")
