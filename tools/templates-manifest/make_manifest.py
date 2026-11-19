"""What each of ComfyUI's workflow templates needs, for the Studio's template browser (src/SpawnGen.Studio/wwwroot/template-needs.json).

The templates are Comfy-Org's (MIT, github.com/Comfy-Org/workflow_templates). For one commit of that repository this writes, per template in its
index.json: the node types the template uses (sub-graphs included; ComfyUI frontend-only nodes and sub-graph ids left out) and the model files it
names (each node's properties.models[].name, and widget values ending in a model file extension). The Studio checks those against the engine it
runs on (its node registry and its loaders' file lists) and shows each template as ready, needing models, missing nodes, or a paid API.

The Studio and the server fetch the templates from the same commit (Studio.razor TemplatesUpstream, ServerOptions.TemplatesSource), so the list
matches what they open; a test keeps the three in step. To move to a newer commit:

    git clone --depth 1 --filter=blob:none --no-checkout https://github.com/Comfy-Org/workflow_templates.git wt
    cd wt && git sparse-checkout set --no-cone '/templates/*.json' && git checkout && cd ..
    python tools/templates-manifest/make_manifest.py wt

then put the printed commit in TemplatesUpstream and ServerOptions.TemplatesSource.
"""
import json, os, re, subprocess, sys
from collections import Counter

FRONTEND = {"Note", "MarkdownNote", "Reroute", "PrimitiveNode"}
MODEL_EXT = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx", ".sft")


def walk(wf, nodes, models):
    for n in wf.get("nodes") or []:
        t = n.get("type")
        if t: nodes.add(t)
        for m in ((n.get("properties") or {}).get("models") or []):
            if isinstance(m, dict) and isinstance(m.get("name"), str): models.add(m["name"].replace("\\", "/").split("/")[-1])
        wv = n.get("widgets_values")
        for v in (wv.values() if isinstance(wv, dict) else wv or []):
            if isinstance(v, str) and v.lower().endswith(MODEL_EXT): models.add(v.replace("\\", "/").split("/")[-1])
    for g in ((wf.get("definitions") or {}).get("subgraphs") or []): walk(g, nodes, models)


def main(repo):
    base = os.path.join(repo, "templates")
    commit = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    index = json.load(open(os.path.join(base, "index.json"), encoding="utf-8"))
    out = {}
    for group in index:
        for t in group.get("templates") or []:
            name = t["name"]
            path = os.path.join(base, name + ".json")
            if name in out or not os.path.exists(path): continue
            wf = json.load(open(path, encoding="utf-8"))
            nodes, models = set(), set()
            walk(wf, nodes, models)
            subgraphs = {g.get("id") for g in ((wf.get("definitions") or {}).get("subgraphs") or [])}
            out[name] = {"nodes": sorted(nodes - FRONTEND - subgraphs), "models": sorted(models)}
    target = os.path.join(os.path.dirname(__file__), "..", "..", "src", "SpawnGen.Studio", "wwwroot", "template-needs.json")
    with open(target, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"source": "Comfy-Org/workflow_templates", "commit": commit, "templates": out, "verified_models": verified()}, f,
                  separators=(",", ":"), sort_keys=True)
        f.write("\n")
    print(commit, len(out), "templates,", os.path.getsize(target), "bytes")
    write_catalog(base, commit)


def verified():
    """The model files SpawnGen has run or measured (verified-models.json, each with its evidence): Ready needs every file of a template here."""
    path = os.path.join(os.path.dirname(__file__), "verified-models.json")
    return sorted(json.load(open(path, encoding="utf-8"))["files"])


HF_URL = re.compile(r"^https://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/(.+?)(\?.*)?$")


def write_catalog(base, commit):
    """The model files the templates name, with where ComfyUI gets each one (properties.models: a Hugging Face URL and the models sub-folder),
    as src/SpawnGen.Studio/wwwroot/model-catalog.json. Both engines register these files (HostedModels), so a loader lists them and the model
    source fetches them from Hugging Face on first use (CORS allows the browser). A name in several repos takes the repo most templates use."""
    seen = {}

    def walk(wf):
        if not isinstance(wf, dict): return   # a non-workflow JSON in the folder (an index of another kind)
        for n in wf.get("nodes") or []:
            for m in ((n.get("properties") or {}).get("models") or []):
                if not isinstance(m, dict) or not isinstance(m.get("name"), str): continue
                hit = HF_URL.match(m.get("url") or "")
                if not hit or not m.get("directory"): continue
                key = (hit.group(1), hit.group(3), m["directory"])
                seen.setdefault(m["name"].replace("\\", "/").split("/")[-1], Counter())[key] += 1
        for g in ((wf.get("definitions") or {}).get("subgraphs") or []): walk(g)

    for path in sorted(os.listdir(base)):
        if path.endswith(".json") and path != "index.json":
            try: walk(json.load(open(os.path.join(base, path), encoding="utf-8")))
            except (ValueError, OSError): pass
    models = {}
    for name, counts in sorted(seen.items()):
        (repo, file, folder), _ = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        models[name] = {"repo": repo, "path": file, "folder": folder}
    target = os.path.join(os.path.dirname(__file__), "..", "..", "src", "SpawnGen.Studio", "wwwroot", "model-catalog.json")
    with open(target, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"source": "Comfy-Org/workflow_templates", "commit": commit, "models": models}, f, separators=(",", ":"), sort_keys=True)
        f.write("\n")
    print(len(models), "hosted model files,", os.path.getsize(target), "bytes")


if __name__ == "__main__":
    main(sys.argv[1])
