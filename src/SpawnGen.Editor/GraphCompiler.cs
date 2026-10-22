using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Editor;

/// <summary>The executable prompt for an editor graph, with what was left out and why (shown to the user, not thrown).</summary>
public sealed record CompileResult(PromptGraph Graph, IReadOnlyList<string> Notes);

/// <summary>Turns the editor graph into a ComfyUI API-format <see cref="PromptGraph"/>: widget values become literals, links become links, muted nodes disappear, bypassed nodes pass their inputs through.</summary>
public static class GraphCompiler
{
    public static CompileResult Compile(EditorGraph graph, NodeRegistry registry)
    {
        var notes = new List<string>();
        var skipped = new HashSet<int>();

        string Name(EditorNode n) => $"{n.Title ?? registry.GetSchema(n.ClassType)?.DisplayName ?? n.ClassType} (#{n.Id})";

        // 1. Nodes that cannot run at all: muted, bypassed (replaced by a pass-through below), or of a type nobody registered.
        foreach (var n in graph.Nodes)
        {
            if (registry.GetSchema(n.ClassType) is null) { skipped.Add(n.Id); notes.Add($"{Name(n)} skipped: node type '{n.ClassType}' is not installed."); }
            else if (n.Mode == NodeMode.Never) { skipped.Add(n.Id); notes.Add($"{Name(n)} is muted."); }
            else if (n.Mode == NodeMode.Bypass) skipped.Add(n.Id);
        }

        // 2. Where a link really comes from once bypassed nodes are looked through. Null = nothing can supply it (muted or missing source).
        (int Node, int Output)? Source(int fromNode, int fromOutput, HashSet<int>? seen = null)
        {
            var node = graph.FindNode(fromNode);
            if (node is null || registry.GetSchema(node.ClassType) is not { } schema) return null;
            if (node.Mode == NodeMode.Never) return null;
            if (node.Mode != NodeMode.Bypass) return (fromNode, fromOutput);

            seen ??= [];
            if (!seen.Add(fromNode) || fromOutput >= schema.Outputs.Count) return null;
            // A bypassed node hands its input of the same type on to whatever was reading its output.
            var type = schema.Outputs[fromOutput].Type;
            foreach (var input in schema.Inputs.Where(i => i.Type.Name == type.Name || i.Type.Accepts(type) && !i.Type.IsAny))
                if (graph.LinkInto(fromNode, input.Name) is { } link) return Source(link.FromNode, link.FromOutput, seen);
            return null;
        }

        // 3. Drop every node that reads from something that cannot run, until nothing else changes.
        bool changed;
        do
        {
            changed = false;
            foreach (var n in graph.Nodes.Where(n => !skipped.Contains(n.Id) || n.Mode == NodeMode.Bypass))
            {
                if (n.Mode == NodeMode.Bypass) continue;
                foreach (var link in graph.Links.Where(l => l.ToNode == n.Id))
                {
                    var src = Source(link.FromNode, link.FromOutput);
                    if (src is null || skipped.Contains(src.Value.Node) && graph.FindNode(src.Value.Node)!.Mode != NodeMode.Bypass)
                    {
                        skipped.Add(n.Id);
                        notes.Add($"{Name(n)} skipped: its '{link.ToInput}' input depends on a muted or missing node.");
                        changed = true;
                        break;
                    }
                }
            }
        } while (changed);

        // 4. Emit the rest.
        var prompt = new PromptGraph();
        foreach (var n in graph.Nodes.OrderBy(n => n.Id))
        {
            if (skipped.Contains(n.Id) || n.Mode == NodeMode.Bypass) continue;
            var schema = registry.GetSchema(n.ClassType)!;
            var node = prompt.Add(n.Id.ToString(), n.ClassType);
            node.Title = n.Title;
            foreach (var input in schema.Inputs)
            {
                if (graph.LinkInto(n.Id, input.Name) is { } link)
                {
                    if (Source(link.FromNode, link.FromOutput) is { } s) node.Inputs[input.Name] = PromptInput.Link(s.Node.ToString(), s.Output);
                }
                else if (input.Widget is not null && n.Widgets.TryGetValue(input.Name, out var value) && value is not null)
                {
                    node.Inputs[input.Name] = PromptInput.Value(value);
                }
            }
        }
        return new CompileResult(prompt, notes);
    }
}
