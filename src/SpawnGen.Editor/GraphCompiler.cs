using System.Globalization;
using System.Text.RegularExpressions;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Editor;

/// <summary>The executable prompt for an editor graph, with what was left out and why (shown to the user, not thrown).</summary>
public sealed record CompileResult(PromptGraph Graph, IReadOnlyList<string> Notes);

/// <summary>Turns the editor graph into a ComfyUI API-format <see cref="PromptGraph"/>: widget values become literals, links become links, muted nodes disappear, bypassed nodes pass their inputs through.</summary>
public static class GraphCompiler
{
    /// <param name="now">The time %date:...% in file name prefixes is filled from (local time; defaults to now).</param>
    public static CompileResult Compile(EditorGraph graph, NodeRegistry registry, DateTime? now = null)
    {
        graph = SubgraphExpander.Flatten(graph); // subgraph instances become their inner nodes
        var notes = new List<string>();
        var skipped = new HashSet<int>();

        string Name(EditorNode n) => $"{n.Title ?? registry.GetSchema(n.ClassType)?.DisplayName ?? n.ClassType} (#{n.Id})";

        // 1. Nodes that cannot run at all: muted, bypassed (replaced by a pass-through below), or of a type nobody registered.
        foreach (var n in graph.Nodes)
        {
            if (registry.GetSchema(n.ClassType) is { FrontendOnly: true }) skipped.Add(n.Id); // Reroute, Note: never emitted, and reroutes are looked through below
            else if (registry.GetSchema(n.ClassType) is null) { skipped.Add(n.Id); notes.Add($"{Name(n)} skipped: node type '{n.ClassType}' is not installed."); }
            else if (n.Mode == NodeMode.Never) { skipped.Add(n.Id); notes.Add($"{Name(n)} is muted."); }
            else if (n.Mode == NodeMode.Bypass) skipped.Add(n.Id);
        }

        // 2. Where a link really comes from once bypassed nodes are looked through. Null = nothing can supply it (muted or missing source).
        (int Node, int Output)? Source(int fromNode, int fromOutput, HashSet<int>? seen = null)
        {
            var node = graph.FindNode(fromNode);
            if (node is null || registry.GetSchema(node.ClassType) is not { } schema) return null;
            if (node.Mode == NodeMode.Never) return null;
            if (schema.FrontendOnly)
            {
                // A reroute is transparent: whatever feeds it feeds what it feeds.
                seen ??= [];
                if (!seen.Add(fromNode) || graph.LinkInto(fromNode, schema.Inputs.FirstOrDefault()?.Name ?? "") is not { } feeding) return null;
                return Source(feeding.FromNode, feeding.FromOutput, seen);
            }
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
                    if (input.Name == "filename_prefix" && value is string prefix && prefix.Contains('%'))
                        value = ApplyTextReplacements(graph, prefix, now ?? DateTime.Now);
                    node.Inputs[input.Name] = PromptInput.Value(value);
                }
            }
        }
        return new CompileResult(prompt, notes);
    }

    static readonly Regex Placeholder = new("%([^%]+)%", RegexOptions.CultureInvariant);
    static readonly Regex DateToken = new("(yyyy|yy|MM|dd|hh|mm|ss)", RegexOptions.CultureInvariant);

    /// <summary>
    /// ComfyUI's frontend text replacements for file name prefixes, done when the prompt is made: <c>%date:yyyy-MM-dd%</c> becomes the date and
    /// <c>%Node.widget%</c> the widget's value, where Node is a node's "Node name for S&amp;R" property (its type unless renamed) or else its title;
    /// slashes in the value become underscores. Anything else (<c>%width%</c>, which the save node fills, or a node that is not there) is left as it is.
    /// </summary>
    public static string ApplyTextReplacements(EditorGraph graph, string value, DateTime now) => Placeholder.Replace(value, m =>
    {
        var text = m.Groups[1].Value;
        var split = text.Split('.');
        if (split.Length != 2)
            return split[0].StartsWith("date:", StringComparison.Ordinal) ? FormatDate(split[0][5..], now) : m.Value;
        var nodes = graph.Nodes.OrderBy(n => n.Id).ToList();
        var node = nodes.FirstOrDefault(n => SearchReplaceName(n) == split[0]) ?? nodes.FirstOrDefault(n => n.Title == split[0]);
        if (node is null || !node.Widgets.TryGetValue(split[1], out var widget)) return m.Value;
        var s = widget switch
        {
            null => "",
            bool b => b ? "true" : "false",
            IFormattable f => f.ToString(null, CultureInfo.InvariantCulture),
            _ => widget.ToString() ?? "",
        };
        return s.Replace('/', '_').Replace('\\', '_');
    });

    /// <summary>The name %Node.widget% finds a node by: its "Node name for S&amp;R" property, which ComfyUI sets to the node type when the node is made.</summary>
    public static string SearchReplaceName(EditorNode node) =>
        node.Properties[ComfyWorkflow.NameForSearchReplace] is System.Text.Json.Nodes.JsonValue v && v.TryGetValue<string>(out var name) ? name : node.ClassType;

    static string FormatDate(string format, DateTime d) => DateToken.Replace(format, m => m.Value switch
    {
        "yyyy" => d.Year.ToString(CultureInfo.InvariantCulture),
        "yy" => (d.Year % 100).ToString("00", CultureInfo.InvariantCulture),
        "MM" => d.Month.ToString("00", CultureInfo.InvariantCulture),
        "dd" => d.Day.ToString("00", CultureInfo.InvariantCulture),
        "hh" => d.Hour.ToString("00", CultureInfo.InvariantCulture),
        "mm" => d.Minute.ToString("00", CultureInfo.InvariantCulture),
        _ => d.Second.ToString("00", CultureInfo.InvariantCulture),
    });
}
