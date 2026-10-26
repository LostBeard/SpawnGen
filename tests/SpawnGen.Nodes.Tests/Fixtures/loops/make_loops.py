"""Runs loop graphs through a ComfyUI server (http://127.0.0.1:8199) and writes them with what ComfyUI made of them to loops.json:
each graph's Preview as Text outputs (by node id), or its validation errors (node id -> loop_error_type).

    python make_loops.py
"""
import json, time, urllib.request, urllib.error

SERVER = "http://127.0.0.1:8199"


def run(graph):
    req = urllib.request.Request(SERVER + "/prompt", data=json.dumps({"prompt": graph}).encode(), headers={"Content-Type": "application/json"})
    try:
        pid = json.load(urllib.request.urlopen(req))["prompt_id"]
    except urllib.error.HTTPError as e:
        body = json.load(e)
        errors = {}
        for node_id, info in body.get("node_errors", {}).items():
            for err in info["errors"]:
                t = err.get("extra_info", {}).get("loop_error_type")
                if t:
                    errors[node_id] = t
        return {"errors": errors}
    while True:
        h = json.load(urllib.request.urlopen(f"{SERVER}/history/{pid}"))
        if pid in h:
            entry = h[pid]
            if entry["status"]["status_str"] != "success":
                return {"failed": json.dumps(entry["status"].get("messages", [])[-1:])[:500]}
            return {"texts": {k: v["text"] for k, v in entry["outputs"].items() if "text" in v and k in graph}, "ran_outputs": sorted(k for k in entry["outputs"] if k in graph)}
        time.sleep(0.5)


def start(mode, **extra):
    inputs = {"mode": mode, "cache_iterations": False}
    inputs.update(extra)
    return {"class_type": "StartLoop", "inputs": inputs}


def math(expr, **values):
    inputs = {"expression": expr}
    inputs.update({f"values.{k}": v for k, v in values.items()})
    return {"class_type": "ComfyMathExpression", "inputs": inputs}


def end(accumulate, **extra):
    inputs = {"accumulate": accumulate}
    inputs.update(extra)
    return {"class_type": "EndLoop", "inputs": inputs}


def preview(src):
    return {"class_type": "PreviewAny", "inputs": {"source": src}}


graphs = {
    "simple_carry_accumulate": {
        "1": start("simple", **{"mode.num_iterations": 4, "initial_iteration_value": ["2", 0]}),
        "2": {"class_type": "PrimitiveInt", "inputs": {"value": 1}},
        "3": math("a*10 + b", a=["1", 0], b=["1", 4]),
        "4": end(True, output_value=["3", 1], next_iteration_value=["3", 1]),
        "5": preview(["4", 0]),
    },
    "for_last": {
        "1": start("For", **{"mode.start_iteration_index": 2, "mode.max_iteration": 10, "mode.step": 3}),
        "3": math("a*a", a=["1", 0]),
        "4": end(False, output_value=["3", 1]),
        "5": preview(["4", 0]),
    },
    "list_items": {
        "11": {"class_type": "PrimitiveInt", "inputs": {"value": 3}},
        "12": {"class_type": "PrimitiveInt", "inputs": {"value": 4}},
        "13": {"class_type": "PrimitiveInt", "inputs": {"value": 7}},
        "10": {"class_type": "CreateList", "inputs": {"inputs.input0": ["11", 0], "inputs.input1": ["12", 0], "inputs.input2": ["13", 0]}},
        "1": start("List", **{"mode.list": ["10", 0]}),
        "3": {"class_type": "StringFormat", "inputs": {"f_string": "{a}-{b}", "values.a": ["1", 3], "values.b": ["1", 0]}},
        "4": end(True, output_value=["3", 0]),
        "5": preview(["4", 0]),
    },
    "nested": {
        "1": start("simple", **{"mode.num_iterations": 2}),
        "6": start("simple", **{"mode.num_iterations": 3, "parent_iteration": ["1", 0]}),
        "7": math("a*10 + b", a=["1", 0], b=["6", 0]),
        "8": end(True, output_value=["7", 1]),
        "4": end(True, output_value=["8", 0]),
        "5": preview(["4", 0]),
    },
    "termination": {
        "1": start("simple", **{"mode.num_iterations": 3}),
        "3": math("a + 100", a=["1", 0]),
        "6": preview(["3", 1]),
        "4": end(False, output_value=["3", 1], **{"terminations.termination0": ["6", 0]}),
        "5": preview(["4", 0]),
    },
    "zero_iterations": {
        "1": start("For", **{"mode.start_iteration_index": 2, "mode.max_iteration": 2, "mode.step": 3}),
        "3": math("a*a", a=["1", 0]),
        "4": end(False, output_value=["3", 1]),
        "5": preview(["4", 0]),
    },
    "end_without_start": {
        "2": {"class_type": "PrimitiveInt", "inputs": {"value": 1}},
        "4": end(False, output_value=["2", 0]),
        "5": preview(["4", 0]),
    },
    "escape": {
        "1": start("simple", **{"mode.num_iterations": 2}),
        "3": math("a + 1", a=["1", 0]),
        "6": preview(["3", 1]),
        "4": end(False, output_value=["3", 1]),
        "5": preview(["4", 0]),
    },
}

out = {}
for name, graph in graphs.items():
    out[name] = {"graph": graph, "comfy": run(graph)}
    print(name, json.dumps(out[name]["comfy"])[:300], flush=True)
json.dump(out, open("loops.json", "w"), indent=1)
