"""ComfyUI's results for the dataset nodes (comfy_extras/nodes_dataset.py), made by ComfyUI itself, for DatasetNodeTests.

    python comfy_reference.py <ComfyUI folder> [all | direct | graphs] [server, default http://127.0.0.1:8199]

Run with ComfyUI's own Python environment, with a ComfyUI server running from the same folder. Two parts:

1. comfy_direct.json.gz: the node classes called directly (as ComfyUI's executor calls them, one list item at a time or with whole lists) on the
   pictures in set/: the loaders; every resize and crop with several settings on a wide, a tall, a square, an RGBA picture, a batch of three, a
   picture whose values are not 8-bit and an 8x8 one made to catch Pillow's grey rounding (those two are stored); Random Crop and the shuffles with
   several seeds (above 2^32 too); Normalize, Brightness and Contrast on every 8-bit value; Deduplicate Images at several thresholds, the hash
   distance of every pair (found with the node itself) and the hashes (the node's own compute_hash); the grid. Pillow-made pictures are kept as
   8-bit values (the node returns k / 255), tensor maths as float32.
2. comfy_graphs.json.gz: prompts run on the server (graph_*.json here): set/ copied into ComfyUI's input folder, Load Image-Text (from Folder) feeding
   every text node into Preview as Text, every picture node into Save Image (to Folder), the shuffles, the save nodes in overwrite and increment
   mode, and a prompt that fails (captions that are lists). What the server answered and every file it wrote are kept.
The folder names carry a run id so the server's cache cannot skip a node; the prompts are saved with it, and the test uses the same names.
"""
import base64, gzip, json, os, shutil, sys, time, urllib.request, uuid

COMFY = sys.argv[1]
PART = sys.argv[2] if len(sys.argv) > 2 else "all"
SERVER = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8199"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, COMFY)
from comfy.cli_args import args  # noqa: E402
args.cpu = True
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
import folder_paths  # noqa: E402
from comfy_extras import nodes_dataset as nd  # noqa: E402


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def u8(t):
    """A Pillow-made tensor as its 8-bit values (checks that it is k / 255 exactly)."""
    a = t.cpu().numpy()
    k = np.rint(a * 255).astype(np.uint8)
    assert np.array_equal((k.astype(np.float32) / 255.0), a), "not an 8-bit picture"
    return {"shape": list(a.shape), "u8": b64(k.tobytes())}


def f32(t):
    a = t.cpu().numpy().astype("<f4")
    return {"shape": list(a.shape), "f32": b64(a.tobytes())}


def out0(r):
    return r.args[0]


# ---- part 1: direct calls ------------------------------------------------------------------------------------------------------------------------
def open_tensor(name, mode):
    return torch.from_numpy(np.array(Image.open(os.path.join(HERE, "set", name)).convert(mode)).astype(np.float32) / 255.0)[None,]


SEEDS = [0, 1, 99, 4294967294, 4294967295, 4294967296 + 7, 123456789012345]


def direct_part():
    folder_paths.set_input_directory(HERE)
    loaded_imgs, loaded_texts = nd.LoadImageTextDataSetFromFolderNode.execute(folder="set").args
    plain_imgs = out0(nd.LoadImageDataSetFromFolderNode.execute(folder="set"))
    direct = {
        "listdir": os.listdir(os.path.join(HERE, "set")),
        "load_text": {"images": [u8(t) for t in loaded_imgs], "texts": loaded_texts},
        "load": [u8(t) for t in plain_imgs],
    }

    # The test builds the same inputs from set/: a (wide), B (tall), plain/p (square), c as RGBA, and a batch of a, a_twin and a mirrored.
    a = open_tensor("a.png", "RGB")
    inputs = {
        "wide": a,
        "tall": open_tensor("B.PNG", "RGB"),
        "square": open_tensor("plain/p.png", "RGB"),
        "rgba": open_tensor("c.png", "RGBA"),
        "batch": torch.cat([a, open_tensor("a_twin.png", "RGB"), a.flip(2)], 0),
        # Values that are not 8-bit (so truncating to 8 bits shows), and an 8x8 picture whose left half has a grey level of 100.71 (Pillow's L rounds
        # it to 101) and right half 101: with the rounding the halves hash alike.
        "soft": (torch.arange(9 * 13 * 3, dtype=torch.float32).reshape(1, 9, 13, 3) * 0.0123) % 1.0,
        "halves": torch.from_numpy(np.array([[[102, 100, 101]] * 4 + [[101, 101, 101]] * 4] * 8, dtype=np.uint8).astype(np.float32) / 255.0)[None,],
    }
    direct["inputs"] = {"soft": f32(inputs["soft"]), "halves": u8(inputs["halves"])}
    cases = []
    for name in inputs:
        for e in (17, 41):
            cases.append(("ResizeImagesByShorterEdge", name, {"shorter_edge": e}, nd.ResizeImagesByShorterEdgeNode))
        for e in (25, 45):
            cases.append(("ResizeImagesByLongerEdge", name, {"longer_edge": e}, nd.ResizeImagesByLongerEdgeNode))
        for w, h in ((15, 8), (100, 6)):
            cases.append(("CenterCropImages", name, {"width": w, "height": h}, nd.CenterCropImagesNode))
        for s in SEEDS:
            cases.append(("RandomCropImages", name, {"width": 9, "height": 11, "seed": s}, nd.RandomCropImagesNode))
        cases.append(("RandomCropImages", name, {"width": 100, "height": 5, "seed": 5}, nd.RandomCropImagesNode))
    direct["single"] = [{"node": node, "input": name, "params": params, "output": u8(out0(cls.execute(inputs[name], **params)))}
                        for node, name, params, cls in cases]

    # The tensor-maths nodes work value by value: every 8-bit value k / 255 in each of 3 and 4 channels (the 4th is alpha).
    ramp = torch.arange(256, dtype=torch.float32) / 255.0
    ramps = {c: ramp[None, None, :, None].repeat(1, 1, 1, c).contiguous() for c in (3, 4)}
    maths = []
    for m, s in ((0.3, 0.45), (0.5, 0.5), (0.123, 0.001), (0.0, 1.0)):
        maths.append(("NormalizeImages", {"mean": m, "std": s}, nd.NormalizeImagesNode))
    for f in (1.37, 0.0, 2.0, 0.61, 1.0):
        maths.append(("AdjustBrightness", {"factor": f}, nd.AdjustBrightnessNode))
    for f in (1.6, 0.25, 0.0, 1.0, 2.0):
        maths.append(("AdjustContrast", {"factor": f}, nd.AdjustContrastNode))
    direct["maths"] = [{"node": node, "channels": c, "params": params, "output": f32(out0(cls.execute(ramps[c], **params)))}
                       for node, params, cls in maths for c in (3, 4)]

    # Lists of distinct 1x1 pictures (value i / 255) so where each one went is plain; the second list has a batch of 3 as one item.
    dots = [torch.full((1, 1, 1, 3), i / 255.0) for i in range(9)]
    dots_batched = [torch.cat(dots[0:3], 0)] + dots[3:]

    def firsts(items):
        return [int(round(float(t[0, 0, 0, 0]) * 255)) for t in items]

    direct["shuffle"] = []
    for s in SEEDS:
        for label, items in (("dots", dots), ("dots_batched", dots_batched)):
            r = out0(nd.ShuffleDatasetNode.execute(items, seed=[s]))
            direct["shuffle"].append({"seed": s, "input": label, "order": firsts(r), "shapes": [list(t.shape) for t in r]})
    pair_texts = [f"text {i}" for i in range(len(dots_batched))]
    direct["shuffle_pairs"] = []
    for s in SEEDS:
        imgs, txts = nd.ShuffleImageTextDatasetNode.execute(dots_batched, pair_texts, [s]).args
        direct["shuffle_pairs"].append({"seed": s, "texts": txts, "first": firsts(imgs), "shapes": [list(t.shape) for t in imgs]})

    # Deduplicate Images on the loaded list at several thresholds (kept pictures by the index of the first equal one in the list), and every pair's
    # hash distance: the smallest d for which the node calls the pair duplicates at threshold 1 - d / 64, found with the node itself.
    flat_loaded = nd.ImageProcessingNode._ensure_image_list(loaded_imgs)
    direct["dedup"] = []
    for th in (0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.0, 1.0):
        kept = out0(nd.ImageDeduplicationNode.execute(loaded_imgs, similarity_threshold=[th]))
        direct["dedup"].append({"threshold": th, "kept": [next(i for i, t in enumerate(flat_loaded) if torch.equal(t, k)) for k in kept]})
    flat = nd.ImageProcessingNode._ensure_image_list(list(loaded_imgs) + [inputs["rgba"], inputs["batch"]])
    distances = []
    for i in range(len(flat)):
        row = []
        for j in range(len(flat)):
            lo, hi = 0, 64
            while lo < hi:
                mid = (lo + hi) // 2
                if len(out0(nd.ImageDeduplicationNode.execute([flat[i], flat[j]], similarity_threshold=[1.0 - mid / 64.0]))) == 1:
                    hi = mid
                else:
                    lo = mid + 1
            row.append(lo)
        distances.append(row)
    direct["dedup_distances"] = distances  # over the loaded list, then the RGBA picture, then the three frames of the batch

    # The hash itself: compute_hash is a function inside the node's _group_process; it is taken out of the node's code as it is and run on the
    # same frames plus the soft and halves pictures.
    import types
    code = next(c for c in nd.ImageDeduplicationNode._group_process.__func__.__code__.co_consts if isinstance(c, types.CodeType) and c.co_name == "compute_hash")
    compute_hash = types.FunctionType(code, vars(nd))
    direct["dedup_hashes"] = [compute_hash(t) for t in flat + [inputs["soft"], inputs["halves"]]]

    direct["grid"] = []
    for label, items, cols, cw, ch, pad in (("loaded", loaded_imgs, 4, 40, 34, 3), ("mixed", [inputs["rgba"], inputs["batch"], inputs["tall"]], 3, 32, 45, 0)):
        g = out0(nd.ImageGridNode.execute(items, columns=[cols], cell_width=[cw], cell_height=[ch], padding=[pad]))
        direct["grid"].append({"input": label, "columns": cols, "cell_width": cw, "cell_height": ch, "padding": pad, "output": u8(g)})

    with gzip.open(os.path.join(HERE, "comfy_direct.json.gz"), "wt", encoding="utf-8") as f:
        json.dump(direct, f)
    print("direct:", len(direct["single"]), "picture cases,", len(direct["maths"]), "maths cases")


# ---- part 2: prompts on the server ---------------------------------------------------------------------------------------------------------------
def graphs_part():
    run = uuid.uuid4().hex[:8]
    src = f"spawngen_dataset_{run}"
    dst = f"spawngen_dataset_ref_{run}"
    comfy_input = os.path.join(COMFY, "input", src)
    comfy_output = os.path.join(COMFY, "output", dst)
    shutil.copytree(os.path.join(HERE, "set"), comfy_input)

    def L(node, out=0):
        return [str(node), out]

    def main_graph():
        g = {"1": {"class_type": "LoadImageTextDataSetFromFolder", "inputs": {"folder": src}},
             "2": {"class_type": "PreviewAny", "inputs": {"source": L(1, 1)}}}
        texts = [("TextToUppercase", {}), ("TextToLowercase", {}), ("StripWhitespace", {}), ("TruncateText", {"max_length": 6}),
                 ("AddTextPrefix", {"prefix": "p: "}), ("AddTextSuffix", {"suffix": " #s"}), ("ReplaceText", {"find": "a", "replace": "4"}),
                 ("ReplaceText", {"find": "", "replace": "|"}), ("MergeTextLists", {})]
        n = 10
        for cls, params in texts:
            g[str(n)] = {"class_type": cls, "inputs": {"texts": L(1, 1), **params}}
            g[str(n + 1)] = {"class_type": "PreviewAny", "inputs": {"source": L(n)}}
            n += 2
        # A text node's output is a list of one text: Truncate Text slices that list, and an extra input unwraps it.
        g["40"] = {"class_type": "TruncateText", "inputs": {"texts": L(10), "max_length": 2}}
        g["41"] = {"class_type": "PreviewAny", "inputs": {"source": L(40)}}
        g["42"] = {"class_type": "AddTextPrefix", "inputs": {"texts": L(1, 1), "prefix": L(12)}}
        g["43"] = {"class_type": "PreviewAny", "inputs": {"source": L(42)}}
        g["44"] = {"class_type": "ReplaceText", "inputs": {"texts": L(1, 1), "find": L(12), "replace": "<>"}}
        g["45"] = {"class_type": "PreviewAny", "inputs": {"source": L(44)}}
        pictures = [("ResizeImagesByShorterEdge", {"shorter_edge": 17}), ("ResizeImagesByLongerEdge", {"longer_edge": 25}),
                    ("CenterCropImages", {"width": 15, "height": 8}), ("RandomCropImages", {"width": 9, "height": 11, "seed": 99}),
                    ("NormalizeImages", {"mean": 0.3, "std": 0.45}), ("AdjustBrightness", {"factor": 1.37}), ("AdjustContrast", {"factor": 1.6}),
                    ("ShuffleDataset", {"seed": 7}), ("ImageDeduplication", {"similarity_threshold": 0.95}),
                    ("ImageGrid", {"columns": 4, "cell_width": 40, "cell_height": 34, "padding": 3}), ("MergeImageLists", {})]
        n = 60
        for cls, params in pictures:
            g[str(n)] = {"class_type": cls, "inputs": {"images": L(1), **params}}
            g[str(n + 1)] = {"class_type": "SaveImageDataSetToFolder", "inputs": {"images": L(n), "folder_name": f"{dst}/{cls}", "filename_prefix": "img", "mode": "overwrite"}}
            n += 2
        g["90"] = {"class_type": "LoadImageDataSetFromFolder", "inputs": {"folder": src}}
        g["91"] = {"class_type": "SaveImageDataSetToFolder", "inputs": {"images": L(90), "folder_name": f"{dst}/load", "filename_prefix": "img", "mode": "overwrite"}}
        g["92"] = {"class_type": "ShuffleImageTextDataset", "inputs": {"images": L(1), "texts": L(1, 1), "seed": 12345}}
        g["93"] = {"class_type": "SaveImageTextDataSetToFolder", "inputs": {"images": L(92), "texts": L(92, 1), "folder_name": f"{dst}/pairs", "filename_prefix": "pair", "mode": "overwrite"}}
        g["94"] = {"class_type": "SaveImageTextDataSetToFolder", "inputs": {"images": L(1), "texts": L(1, 1), "folder_name": f"{dst}/inc", "filename_prefix": "image", "mode": "overwrite"}}
        g["95"] = {"class_type": "SaveImageDataSetToFolder", "inputs": {"images": L(1), "folder_name": f"{dst}/fresh", "filename_prefix": "x.png", "mode": "increment"}}
        g["96"] = {"class_type": "SaveImageTextDataSetToFolder", "inputs": {"images": L(1), "folder_name": f"{dst}/notexts", "filename_prefix": "n", "mode": "overwrite"}}
        return g

    def increment_graph():
        # After main: the same folder in increment mode, so the numbering continues after the overwrite run's files.
        return {"1": {"class_type": "LoadImageTextDataSetFromFolder", "inputs": {"folder": src}},
                "2": {"class_type": "SaveImageTextDataSetToFolder", "inputs": {"images": L(1), "texts": L(1, 1), "folder_name": f"{dst}/inc", "filename_prefix": "image", "mode": "increment"}}}

    def failing_graph():
        # Captions that are lists (a text node's output): the pictures are written, then writing the first caption fails.
        return {"1": {"class_type": "LoadImageTextDataSetFromFolder", "inputs": {"folder": src}},
                "2": {"class_type": "TextToUppercase", "inputs": {"texts": L(1, 1)}},
                "3": {"class_type": "SaveImageTextDataSetToFolder", "inputs": {"images": L(1), "texts": L(2), "folder_name": f"{dst}/bad", "filename_prefix": "b", "mode": "overwrite"}}}

    def run_prompt(g):
        req = urllib.request.Request(SERVER + "/prompt", data=json.dumps({"prompt": g}).encode(), headers={"Content-Type": "application/json"})
        pid = json.load(urllib.request.urlopen(req))["prompt_id"]
        while True:
            h = json.load(urllib.request.urlopen(f"{SERVER}/history/{pid}"))
            if pid in h:
                return h[pid]
            time.sleep(0.5)

    graphs = {"main": main_graph(), "increment": increment_graph(), "failing": failing_graph()}
    answers = {}
    for name, g in graphs.items():
        h = run_prompt(g)
        errors = [m[1] for m in h["status"]["messages"] if m[0] == "execution_error"]
        answers[name] = {"status": h["status"]["status_str"], "ui": {k: v.get("text") for k, v in h["outputs"].items() if "text" in v},
                         "error": {"node_type": errors[0]["node_type"], "exception_type": errors[0]["exception_type"], "message": errors[0]["exception_message"]} if errors else None}
        print(name, answers[name]["status"], answers[name]["error"])
        with open(os.path.join(HERE, f"graph_{name}.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(g, f, indent=1)

    files = {}
    for root, _, names in os.walk(comfy_output):
        for fn in names:
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, comfy_output).replace(os.sep, "/")
            if fn.endswith(".png"):
                im = Image.open(path)
                files[rel] = {"mode": im.mode, "size": list(im.size), "u8": b64(np.array(im).tobytes()), "text_chunks": sorted(im.info.keys())}
            else:
                files[rel] = {"bytes": b64(open(path, "rb").read())}
    with gzip.open(os.path.join(HERE, "comfy_graphs.json.gz"), "wt", encoding="utf-8") as f:
        json.dump({"input_folder": src, "output_folder": dst, "answers": answers, "files": files, "linesep": os.linesep}, f)
    print("graphs:", len(files), "files")
    shutil.rmtree(comfy_input)
    shutil.rmtree(comfy_output)


if PART in ("all", "direct"):
    direct_part()
if PART in ("all", "graphs"):
    graphs_part()
