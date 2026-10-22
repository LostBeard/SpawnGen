# Custom nodes (C#)

SpawnGen's answer to ComfyUI's Python `custom_nodes`: write nodes in C#, build a DLL, drop it in a folder, and the server loads it.
There is no Python anywhere. The Studio (in the browser or the desktop window) learns the new nodes from the server and runs graphs that use them on the server.

## A node

A node is a public class marked `[Node]` with one `Execute` or `ExecuteAsync` method. The sockets and widgets come from the method signature:

```csharp
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

[Node(Category = "sample", DisplayName = "Reverse Text")]
public sealed class SampleReverseText
{
    public string Execute([Input(Multiline = true)] string text = "") => new string(text.Reverse().ToArray());
}
```

- Parameters of type `ImageBatch`, `MaskBatch`, `LatentBatch`, or a class marked `[Socket("NAME")]` are sockets (links). `int`/`long`, `float`/`double`, `bool` and `string` parameters are widgets
  that can also be fed by a link. `[Input(Min = .., Max = .., Step = .., Multiline = true, OptionsProvider = nameof(Method))]` tunes a widget; an `OptionsProvider` makes it a combo.
- Return one value, or a tuple for several outputs (tuple element names become the output names). Output nodes that show or save something set `IsOutput = true` and return a `UiOutput`.
- Extra parameters `NodeContext` (progress, status, cancellation, preview pictures) and `IServiceProvider` are filled in by the engine.
- Pictures are `ImageBatch`es of floats in 0..1 laid out as (batch, height, width, channels). Do not hold on to GPU memory between runs; copy results to the host.

A complete pack with two nodes is in `samples/SpawnGen.SampleNodes`.

## Installing a pack

1. Build your class library (target `net10.0`). Reference `SpawnGen.Engine` (and `SpawnGen.Nodes.Core` if you use its helpers) with `<Private>false</Private>` so their DLLs are not copied next to your pack:
   the server's own copies must be the ones used, or the `[Node]` attribute would be a different type. `EnableDynamicLoading` is a good idea too (see the sample's project file).
2. Copy the `.dll` into the server's custom nodes folder: `%LOCALAPPDATA%\SpawnGen\custom_nodes` by default (`SpawnGen:CustomNodesPath` or the environment variable `SpawnGen__CustomNodesPath` changes it).
   A pack with its own dependencies goes in a sub-folder named like the DLL: `custom_nodes/MyPack/MyPack.dll` plus its files.
3. Restart the server. The log lists what each pack added; `GET /system_stats` has a `custom_nodes` entry per pack (with the error message for one that failed to load, which does not stop the server).

Open the Studio from the server (`http://localhost:8188/`) or the desktop app: the new nodes are in the node library and in search, and workflows that use them open without "missing node" warnings.
They run on the server's engine. The browser-only engine ("This browser") does not have them and says so.

## Trust

A node pack is code, and it runs with the rights of the server. Install packs you trust, as you would a ComfyUI custom node.
