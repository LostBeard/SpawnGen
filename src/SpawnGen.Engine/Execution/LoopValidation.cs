using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Engine.Execution;

/// <summary>
/// ComfyUI's validate_loops (comfy_execution/validation.py): pairs every End Loop with the closest Start Loop it depends on, innermost first, finds each loop's
/// body (the nodes after the Start that reach the End) and refuses a body that escapes its End, an End without a Start, a Start without an End, ambiguous
/// nesting, and an accumulate switch driven from inside the loop. The same walks and orders as ComfyUI, so the same graphs pass and fail.
/// </summary>
internal static class LoopValidation
{
    public static Dictionary<string, LoopScope> Validate(NodeRegistry registry, PromptGraph graph, IReadOnlyList<string> outputs, IReadOnlyCollection<string> validated,
        List<ValidationError> errors)
    {
        var startNodes = new HashSet<string>(StringComparer.Ordinal);
        var endNodes = new HashSet<string>(StringComparer.Ordinal);
        foreach (var (id, node) in graph.Nodes)
        {
            var boundary = registry.GetSchema(node.ClassType)?.LoopBoundary;
            if (boundary == "start") startNodes.Add(id);
            else if (boundary == "end") endNodes.Add(id);
        }
        var result = new Dictionary<string, LoopScope>(StringComparer.Ordinal);
        if (startNodes.Count == 0 && endNodes.Count == 0) return result;

        var nodeIds = new HashSet<string>(validated, StringComparer.Ordinal);
        var allChildren = graph.Nodes.Keys.ToDictionary(k => k, _ => new HashSet<string>(StringComparer.Ordinal), StringComparer.Ordinal);
        var allParents = graph.Nodes.Keys.ToDictionary(k => k, _ => new HashSet<string>(StringComparer.Ordinal), StringComparer.Ordinal);
        foreach (var (id, node) in graph.Nodes)
            foreach (var input in node.Inputs.Values)
                if (input.IsLink && allChildren.ContainsKey(input.NodeId!))
                {
                    allChildren[input.NodeId!].Add(id);
                    allParents[id].Add(input.NodeId!);
                }
        var continuation = Walk(outputs, allChildren, startNodes);
        continuation.ExceptWith(startNodes.Where(s => !nodeIds.Contains(s)));
        nodeIds.UnionWith(continuation);
        nodeIds.UnionWith(Walk(endNodes.Where(continuation.Contains), allParents, startNodes));

        var parents = nodeIds.ToDictionary(id => id, id => graph.Nodes[id].Inputs.Values.Where(v => v.IsLink && nodeIds.Contains(v.NodeId!)).Select(v => v.NodeId!)
            .ToHashSet(StringComparer.Ordinal), StringComparer.Ordinal);
        var children = nodeIds.ToDictionary(id => id, _ => new HashSet<string>(StringComparer.Ordinal), StringComparer.Ordinal);
        foreach (var (id, ps) in parents)
            foreach (var p in ps) children[p].Add(id);

        startNodes.IntersectWith(nodeIds);
        endNodes.IntersectWith(nodeIds);
        var terminalOutputs = outputs.Where(o => children.TryGetValue(o, out var c) && c.Count == 0).ToHashSet(StringComparer.Ordinal);

        // The Start DAG, independently of the Ends: completed inner loops can lead to later Starts still nested under the same outer Start.
        var startDag = startNodes.ToDictionary(s => s, s => Walk(children[s], children, startNodes, returnStops: true), StringComparer.Ordinal);
        var startDescendants = startNodes.ToDictionary(s => s, s => Walk(startDag[s], startDag), StringComparer.Ordinal);
        // The End DAG in the reverse direction: its leaves are the innermost Ends, paired first.
        var endDag = endNodes.ToDictionary(e => e, e => Walk(parents[e], parents, endNodes, returnStops: true), StringComparer.Ordinal);

        var pairs = new Dictionary<string, string>(StringComparer.Ordinal);
        var remainingStarts = new HashSet<string>(startNodes, StringComparer.Ordinal);
        var remainingEnds = new HashSet<string>(endNodes, StringComparer.Ordinal);
        while (remainingEnds.Count > 0)
        {
            var endId = remainingEnds.Order(StringComparer.Ordinal).FirstOrDefault(e => !endDag[e].Overlaps(remainingEnds));
            if (endId is null) break; // Ends that depend on each other in a circle: the cycle check reports the graph
            var candidates = Walk(parents[endId], parents, remainingStarts, returnStops: true);
            if (candidates.Count == 0)
                return Fail(errors, "loop_end_without_start", "End Loop has no Start Loop", $"End Loop {endId} has no available Start Loop", [endId]);
            var closest = candidates.Where(c => candidates.All(o => o == c || startDescendants[o].Contains(c))).ToList();
            if (closest.Count != 1)
                return Fail(errors, "ambiguous_loop_nesting", "End Loop has ambiguous Start Loops",
                    $"End Loop {endId} can close multiple unrelated Start Loops: {string.Join(", ", candidates.Order(StringComparer.Ordinal))}", [.. candidates, endId]);
            var startId = closest[0];
            pairs[startId] = endId;
            remainingStarts.Remove(startId);
            remainingEnds.Remove(endId);

            // Validate the new pair at once: earlier paired Ends are inner boundaries and may be crossed; an unpaired End or an output is an escape.
            var stops = new HashSet<string>(remainingEnds, StringComparer.Ordinal);
            stops.UnionWith(terminalOutputs);
            stops.Add(endId);
            var escapes = Walk(children[startId], children, stops, returnStops: true);
            escapes.Remove(endId);
            if (escapes.Count > 0)
                return Fail(errors, "loop_escape", "Loop body is not closed",
                    $"Start Loop {startId} reaches {string.Join(", ", escapes.Order(StringComparer.Ordinal))} without passing through End Loop {endId}", [.. escapes, startId, endId]);
        }
        if (remainingStarts.Count > 0)
            return Fail(errors, "loop_start_without_end", "Start Loop has no End Loop",
                $"Start Loops without End Loops: {string.Join(", ", remainingStarts.Order(StringComparer.Ordinal))}", remainingStarts);

        foreach (var (startId, endId) in pairs)
        {
            var body = Walk(children[startId], children, new HashSet<string>(StringComparer.Ordinal) { endId });
            body.Remove(endId);
            if (graph.Nodes[endId].Inputs.TryGetValue("accumulate", out var accumulate) && accumulate.IsLink && (accumulate.NodeId == startId || body.Contains(accumulate.NodeId!)))
                return Fail(errors, "loop_accumulate_from_body", "End Loop accumulate depends on its loop body",
                    $"End Loop {endId} accumulate is driven by loop node {accumulate.NodeId} under Start Loop {startId}", [startId, endId, accumulate.NodeId!]);
            result[startId] = new LoopScope(body.Order(StringComparer.Ordinal).ToList(), endId);
        }
        return result;
    }

    /// <summary>ComfyUI's _walk_graph: everything reachable from <paramref name="start"/> along <paramref name="edges"/>, not going past the stop nodes (which are found);
    /// with <paramref name="returnStops"/> only the stop nodes reached.</summary>
    static HashSet<string> Walk(IEnumerable<string> start, IReadOnlyDictionary<string, HashSet<string>> edges, IReadOnlySet<string>? stopAt = null, bool returnStops = false)
    {
        var found = new HashSet<string>(StringComparer.Ordinal);
        var stops = new HashSet<string>(StringComparer.Ordinal);
        var pending = new Stack<string>(start);
        while (pending.Count > 0)
        {
            var id = pending.Pop();
            if (!found.Add(id)) continue;
            if (stopAt is not null && stopAt.Contains(id)) stops.Add(id);
            else if (edges.TryGetValue(id, out var next)) foreach (var n in next) pending.Push(n);
        }
        return returnStops ? stops : found;
    }

    /// <summary>ComfyUI raises one LoopValidationError and marks every node it names: one error per node, the code being its loop_error_type.</summary>
    static Dictionary<string, LoopScope> Fail(List<ValidationError> errors, string type, string message, string details, IEnumerable<string> actors)
    {
        foreach (var id in actors.Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal))
            errors.Add(new ValidationError(id, "loop boundary", type, $"{message}: {details}"));
        return new Dictionary<string, LoopScope>(StringComparer.Ordinal);
    }
}
