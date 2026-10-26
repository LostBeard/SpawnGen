# Compares SpawnGen's /object_info with ComfyUI's, node by node, and prints what differs.
#
#   python compare_object_info.py comfy_object_info.json spawngen_object_info.json [--all] [--node ID]
#
# Get ComfyUI's from a running ComfyUI (http://127.0.0.1:8188/object_info) and SpawnGen's from its server. Nodes ComfyUI serves from
# comfy_api_nodes (paid cloud APIs) are left out. Combo lists that depend on files on disk (models, inputs) are not compared.
import json, sys
from collections import Counter

FILE_COMBOS = {"ckpt_name", "vae_name", "lora_name", "clip_name", "clip_name1", "clip_name2", "clip_name3", "clip_name4", "unet_name", "image", "audio",
               "video", "model_name", "control_net_name", "style_model_name", "gligen_name", "upscale_model", "file", "model", "latent", "text_encoder"}

def inputs(info):
    """Inputs by name. SpawnGen lists an Autogrow group's members (images.image0, ...) and a dynamic combo's option inputs (resize_type.width);
    ComfyUI lists the group or combo once (COMFY_AUTOGROW_V3 / COMFY_DYNAMICCOMBO_V3): both are folded to the group's name here."""
    out = {}
    order = info.get("input_order") or {}
    for section in ("required", "optional"):
        group = (info.get("input") or {}).get(section) or {}
        names = order.get(section) or list(group.keys())
        for i, name in enumerate(names):
            if name not in group: continue
            entry = group[name]
            cfg = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
            if cfg.get("autogrow_group"):
                g = cfg["autogrow_group"]
                if g not in out: out[g] = (section, ["COMFY_AUTOGROW_V3", {}])
                continue
            if cfg.get("dynamic_parent"): continue
            if entry and entry[0] == "COMFY_AUTOGROW_V3": entry = ["COMFY_AUTOGROW_V3", {}]
            out[name] = (section, entry)
    return out

def normal_default(kind_name, entry, value, options):
    """ComfyUI leaves some defaults out where the frontend supplies them: a combo's first option, '' for text, False for a toggle."""
    if value is None:
        if kind_name == "COMBO": return options[0] if options else None
        if kind_name == "STRING": return ""
        if kind_name == "BOOLEAN": return False
        if kind_name in ("INT", "FLOAT"): return 0
    return value

def kind(entry):
    t = entry[0]
    if isinstance(t, list): return "COMBO", t
    if t == "COMBO": return "COMBO", (entry[1] or {}).get("options", [])
    return t, None

def widget_order(info):
    # The order ComfyUI saves widget values in: required then optional, as listed.
    ins = inputs(info)
    return [n for n, (_, e) in ins.items() if kind(e)[0] in ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO", "COMFY_DYNAMICCOMBO_V3") and not (len(e) > 1 and isinstance(e[1], dict) and e[1].get("forceInput"))]

def compare(cid, c, s):
    problems = []
    for key in ("display_name", "category", "output_node"):
        cv = c.get(key) or (cid if key == "display_name" else c.get(key))  # no display name: ComfyUI shows the id
        if cv != s.get(key): problems.append(f"{key}: comfy={c.get(key)!r} ours={s.get(key)!r}")
    ci, si = inputs(c), inputs(s)
    missing = [n for n in ci if n not in si]
    extra = [n for n in si if n not in ci]
    if missing: problems.append(f"inputs missing: {missing}")
    if extra: problems.append(f"inputs extra: {extra}")
    cw, sw = widget_order(c), widget_order(s)
    if cw != sw: problems.append(f"widget order: comfy={cw} ours={sw}")
    for name in ci:
        if name not in si: continue
        (csec, ce), (ssec, se) = ci[name], si[name]
        if csec != ssec: problems.append(f"{name}: {csec} in comfy, {ssec} in ours")
        ck, copts = kind(ce); sk, sopts = kind(se)
        if ck == "COMFY_DYNAMICCOMBO_V3": ck, copts = "COMBO", [o.get("key") for o in (ce[1] or {}).get("options", [])]
        if ck != sk and not (ck == "*" or sk == "*" or set(ck.split(",")) == set(sk.split(","))): problems.append(f"{name}: type comfy={ck} ours={sk}")
        if ck == "COMBO" and sk == "COMBO" and name not in FILE_COMBOS and list(copts or []) != list(sopts or []):
            problems.append(f"{name}: options comfy={list(copts or [])[:12]} ours={list(sopts or [])[:12]}")
        cc = ce[1] if len(ce) > 1 and isinstance(ce[1], dict) else {}
        sc = se[1] if len(se) > 1 and isinstance(se[1], dict) else {}
        for k in ("default", "min", "max", "step", "multiline"):
            cv, sv = cc.get(k), sc.get(k)
            if k == "default": cv, sv = normal_default(ck, ce, cv, copts), normal_default(sk, se, sv, sopts)
            if k == "multiline" and not cv and not sv: continue
            if k == "step" and ck == "INT" and cv in (None, 1) and sv in (None, 1): continue
            if k == "default" and ck == "COMBO" and name in FILE_COMBOS: continue
            if isinstance(cv, (int, float)) and isinstance(sv, (int, float)) and not isinstance(cv, bool) and abs(cv - sv) <= 1e-9 * max(1, abs(cv)): continue
            if cv != sv and not (cv is None and k in ("min", "max", "step")): problems.append(f"{name}.{k}: comfy={cv!r} ours={sv!r}")
    co, so = c.get("output", []), s.get("output", [])
    if [str(x) for x in co] != [str(x) for x in so] and not all(a == b or "*" in (a, b) for a, b in zip(map(str, co), map(str, so))) or len(co) != len(so):
        problems.append(f"outputs: comfy={co} ours={so}")
    if c.get("output_name") != s.get("output_name"): problems.append(f"output names: comfy={c.get('output_name')} ours={s.get('output_name')}")
    if (c.get("output_is_list") or [False] * len(co)) != (s.get("output_is_list") or [False] * len(so)):
        problems.append(f"output_is_list: comfy={c.get('output_is_list')} ours={s.get('output_is_list')}")
    return problems

def main():
    comfy = json.load(open(sys.argv[1], encoding="utf-8"))
    ours = json.load(open(sys.argv[2], encoding="utf-8"))
    only = sys.argv[sys.argv.index("--node") + 1] if "--node" in sys.argv else None
    comfy = {k: v for k, v in comfy.items() if not str(v.get("python_module", "")).startswith("comfy_api_nodes")}
    shared = sorted(k for k in comfy if k in ours and (only is None or k == only))
    missing = sorted(k for k in comfy if k not in ours)
    bad = 0
    tally = Counter()
    for k in shared:
        p = compare(k, comfy[k], ours[k])
        if p:
            bad += 1
            print(f"\n{k}")
            for line in p:
                print("   " + line)
                tally[line.split(":")[0].split(".")[-1] if "." in line.split(":")[0] else line.split(":")[0]] += 1
    print(f"\n{len(shared)} shared nodes, {bad} differ; {len(missing)} ComfyUI nodes missing; {len([k for k in ours if k not in comfy])} SpawnGen-only")
    if "--all" in sys.argv:
        print("missing:", ", ".join(missing))

main()
