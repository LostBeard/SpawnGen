using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// ComfyUI's loops (comfy_extras/nodes_loop.py). Validation pairs a Start Loop with the End Loop its body leads to. When the End Loop's result is needed the
/// Start Loop expands the body once per iteration (copies of every body node, wired through a LoopIteration that gives the iteration's index, item and carried
/// value), chains a LoopProgress after each iteration, and a LoopResult hands the iterations' outputs to the End Loop.
/// </summary>
[Node("StartLoop", Category = "utilities/looping", DisplayName = "Start Loop", LoopBoundary = "start", InputIsList = true, AlwaysExecute = true,
    OutputNames = ["iteration_index", "is_first", "is_last", "list_item", "current_iteration_value"], OutputMatchTypes = [null, null, null, "list_item", "carried_value"])]
public sealed class StartLoop
{
    public static DynamicOption[] Modes() =>
    [
        new("simple", DynamicInput.Int("num_iterations", 4, min: 0, tooltip: "Number of times to execute the loop body.")),
        new("For",
            DynamicInput.Int("start_iteration_index", 0, tooltip: "Index of the first iteration when using For loop mode."),
            DynamicInput.Int("max_iteration", 4, max: 18446744073709551615d, tooltip: "The exclusive stopping value for iteration_index in For mode."),
            DynamicInput.Int("step", 1, min: 1, tooltip: "The index step size between each iteration when using For loop mode.")),
        new("List", DynamicInput.MatchType("list", "list_item", tooltip: "List of items the loop iterates on. The loop body executes once per item.")),
    ];

    public (long? IterationIndex, bool IsFirst, bool IsLast, object? ListItem, object? CurrentIterationValue, NodeExpansion Expansion) Execute(
        [DynamicCombo(nameof(Modes)), Input(Tooltip = "The loop iteration mode.")] DynamicChoice mode,
        [Input(Default = false, Tooltip = "Reuse unchanged iteration results from previous executions. Disable to execute every iteration again.")] IReadOnlyList<bool> cacheIterations,
        NodeContext context,
        [Input(Optional = true, ForceInput = true, Tooltip = "Connect iteration_index from an outer Start Loop to nest this loop.")] IReadOnlyList<long>? parentIteration = null,
        [Input(MatchType = "carried_value", Optional = true, Tooltip = "Value exposed as current_iteration_value on the first iteration.")] IReadOnlyList<object?>? initialIterationValue = null)
    {
        List<long> values;
        IReadOnlyList<object?>? listItems = null;
        switch (mode.Key)
        {
            case "simple":
                values = [.. Range(0, First(mode, "num_iterations", 4L), 1)];
                break;
            case "For":
                long step = First(mode, "step", 1L);
                if (step == 0) throw new ArgumentException("Start Loop step must not be 0");
                values = [.. Range(First(mode, "start_iteration_index", 0L), First(mode, "max_iteration", 4L), step)];
                break;
            default:
                listItems = mode.Values.TryGetValue("list", out var list) && list is IReadOnlyList<object?> items ? items : [];
                values = [.. Enumerable.Range(0, listItems.Count).Select(i => (long)i)];
                break;
        }
        var scope = context.Execution ?? throw new InvalidOperationException("Start Loop runs inside a prompt run.");
        var loop = scope.Loop(context.NodeId) ?? throw new InvalidOperationException($"Start Loop {context.NodeId} has no End Loop.");
        var self = scope.Node(context.NodeId);
        var expansion = Expand(scope, context.NodeId, loop, values, listItems, self.Inputs.GetValueOrDefault("initial_iteration_value"), cacheIterations.Count > 0 && cacheIterations[0]);
        context.Progress(0, values.Count, $"Iteration 0 / {values.Count}");
        return (null, false, values.Count == 0, null, null, expansion);
    }

    /// <summary>Python's range(start, stop, step) for a positive or negative step.</summary>
    static IEnumerable<long> Range(long start, long stop, long step)
    {
        if (step > 0) for (long v = start; v < stop; v += step) yield return v;
        else for (long v = start; v > stop; v += step) yield return v;
    }

    /// <summary>The first value of a dynamic combo input of this list node (each comes as a list), or the default.</summary>
    static long First(DynamicChoice mode, string name, long fallback) =>
        mode.Values.TryGetValue(name, out var v) && v is IReadOnlyList<object?> { Count: > 0 } list && list[0] is { } first && ValueCoercion.TryCoerce(first, typeof(long), out var typed)
            ? (long)typed! : fallback;

    /// <summary>
    /// ComfyUI's _expand_loop: per iteration a LoopIteration (shown as the Start Loop), a copy of every body node (shown as its original) wired to it, and a
    /// LoopProgress after it; a LoopResult at the end that collects the outputs (every iteration's with accumulate, else the last's) for the End Loop. Each
    /// iteration depends on the previous one's output, carried value and terminations, so they run in order.
    /// </summary>
    static NodeExpansion Expand(IExecutionScope scope, string openerId, LoopScope loop, IReadOnlyList<long> values, IReadOnlyList<object?>? listItems,
        PromptInput? initialValue, bool reuseCache)
    {
        var graph = new NodeExpansion();
        string prefix = openerId + ".0.0."; // ComfyUI's GraphBuilder prefix for the first expansion of this node
        var close = scope.Node(loop.End);
        var outputSource = close.Inputs.GetValueOrDefault("output_value");
        var nextSource = close.Inputs.GetValueOrDefault("next_iteration_value");
        var terminations = close.Inputs.Where(kv => kv.Key.StartsWith("termination", StringComparison.Ordinal) && kv.Value.IsLink).Select(kv => kv.Value).ToList();
        // Python's bool(close_inputs.get("accumulate", False)): a value, or true when it is linked.
        bool accumulate = close.Inputs.TryGetValue("accumulate", out var acc) && (acc.IsLink || acc.Literal is true);

        PromptInput? previousCarry = initialValue;
        var previousDependencies = new List<PromptInput>();
        PromptInput? previousProgress = null;
        var resultInputs = new List<(string Name, PromptInput Value)> { ("close_id", PromptInput.Value(loop.End)) };
        var body = loop.Body.ToList();
        for (int position = 0; position < values.Count; position++)
        {
            object? item = listItems is not null ? listItems[position] : null;
            var iteration = graph.Add($"{prefix}iteration_{position}", "LoopIteration", displayId: openerId);
            iteration.Inputs["iteration_index"] = PromptInput.Value(values[position]);
            iteration.Inputs["is_first"] = PromptInput.Value(position == 0);
            iteration.Inputs["is_last"] = PromptInput.Value(position == values.Count - 1);
            iteration.Inputs["list_item"] = PromptInput.Value(item);
            iteration.Inputs["current_iteration_value"] = previousCarry ?? PromptInput.Value(null);
            iteration.Inputs["reuse_cache"] = PromptInput.Value(reuseCache);
            for (int d = 0; d < previousDependencies.Count; d++) iteration.Inputs[$"dependency{d}"] = previousDependencies[d];
            string iterationId = $"{prefix}iteration_{position}";

            var copies = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (var nodeId in body)
            {
                string copyId = $"{prefix}{position}_{nodeId}";
                graph.Add(copyId, scope.Node(nodeId).ClassType, displayId: nodeId);
                copies[nodeId] = copyId;
            }
            PromptInput Copied(PromptInput source)
            {
                if (!source.IsLink) return source;
                if (source.NodeId == openerId) return PromptInput.Link(iterationId, source.OutputIndex);
                if (copies.TryGetValue(source.NodeId!, out var copy)) return PromptInput.Link(copy, source.OutputIndex);
                return source;
            }
            foreach (var (nodeId, copyId) in copies)
            {
                var original = scope.Node(nodeId);
                var copy = graph.Nodes[copyId];
                copy.Title = original.Title;
                foreach (var (name, value) in original.Inputs) copy.Inputs[name] = Copied(value);
                // A loop nested in this body: its copy is paired with the copies of its body and End Loop.
                if (scope.Loop(nodeId) is { } inner)
                    graph.Loops[copyId] = new LoopScope(inner.Body.Select(b => copies[b]).ToList(), copies[inner.End]);
            }

            List<PromptInput> dependencies;
            if (outputSource is { IsLink: true } output)
            {
                var copiedOutput = Copied(output);
                if (accumulate) resultInputs.Add(($"output{position}", copiedOutput));
                else if (position == values.Count - 1) resultInputs.Add(("output0", copiedOutput));
                dependencies = [copiedOutput];
            }
            else dependencies = [];
            if (nextSource is { IsLink: true } next)
            {
                previousCarry = Copied(next);
                dependencies.Add(previousCarry.Value);
            }
            dependencies.AddRange(terminations.Select(Copied));
            previousDependencies = dependencies;

            string progressId = $"{prefix}progress_{position}";
            var progress = graph.Add(progressId, "LoopProgress");
            progress.Inputs["start_id"] = PromptInput.Value(openerId);
            progress.Inputs["position"] = PromptInput.Value((long)position + 1);
            progress.Inputs["total"] = PromptInput.Value((long)values.Count);
            for (int d = 0; d < dependencies.Count; d++) progress.Inputs[$"dependency{d}"] = dependencies[d];
            if (previousProgress is { } before) progress.Inputs["previous_progress"] = before;
            previousProgress = PromptInput.Link(progressId);
        }
        if (previousProgress is { } last) resultInputs.Add(("progress", last));
        for (int d = 0; d < previousDependencies.Count; d++) resultInputs.Add(($"dependency{d}", previousDependencies[d]));
        var result = graph.Add($"{prefix}result", "LoopResult");
        foreach (var (name, value) in resultInputs) result.Inputs[name] = value;
        return graph;
    }
}

[Node("LoopIteration", Category = "sd", DisplayName = "LoopIteration", InputIsList = true, AcceptAllInputs = true, CacheWhen = "reuse_cache", DevOnly = true,
    OutputIsList = [false, false, false, false, true], OutputNames = ["INT", "BOOLEAN", "BOOLEAN", "*", "*"])]
public sealed class LoopIteration
{
    /// <summary>One iteration's values for the copied body: its index, whether it is the first / last, the list item, and the value carried in (a whole list).</summary>
    public (long IterationIndex, bool IsFirst, bool IsLast, object? ListItem, IReadOnlyList<object?> CurrentIterationValue) Execute(
        IReadOnlyList<long> iterationIndex, IReadOnlyList<bool> isFirst, IReadOnlyList<bool> isLast, IReadOnlyList<bool> reuseCache,
        [ExtraInputs] IReadOnlyDictionary<string, object?> dependencies,
        [Input(Type = "*", Optional = true)] IReadOnlyList<object?>? listItem = null,
        [Input(Type = "*", Optional = true)] IReadOnlyList<object?>? currentIterationValue = null) =>
        // A missing value reaches ComfyUI's node as [None]: the item is None and the carried list holds one None.
        (iterationIndex[0], isFirst[0], isLast[0], listItem is { Count: > 0 } ? listItem[0] : null, currentIterationValue ?? [null]);
}

[Node("LoopProgress", Category = "sd", DisplayName = "LoopProgress", InputIsList = true, AcceptAllInputs = true, IsOutput = true, AlwaysExecute = true, DevOnly = true)]
public sealed class LoopProgress
{
    /// <summary>Reports "Iteration n / total" on the Start Loop once an iteration's outputs are done; runs after the previous one.</summary>
    public long Execute(IReadOnlyList<string> startId, IReadOnlyList<long> position, IReadOnlyList<long> total,
        [ExtraInputs] IReadOnlyDictionary<string, object?> dependencies, NodeContext context)
    {
        context.Progress(position[0], total[0], $"Iteration {position[0]} / {total[0]}");
        return position[0];
    }
}

[Node("LoopResult", Category = "sd", DisplayName = "LoopResult", InputIsList = true, AcceptAllInputs = true, IsOutput = true, AlwaysExecute = true, DevOnly = true)]
public sealed class LoopResult
{
    /// <summary>Hands the outputs (output0, output1, ...: one list per iteration kept) to the End Loop waiting on them.</summary>
    public void Execute(IReadOnlyList<string> closeId, [ExtraInputs] IReadOnlyDictionary<string, object?> outputs, NodeContext context)
    {
        var collected = new List<IReadOnlyList<object?>>();
        while (outputs.TryGetValue($"output{collected.Count}", out var output)) collected.Add(output as IReadOnlyList<object?> ?? [output]);
        (context.Execution ?? throw new InvalidOperationException("Loop Result runs inside a prompt run.")).ReleaseExternalBlock(closeId[0], collected);
    }
}

[Node("EndLoop", Category = "utilities/looping", DisplayName = "End Loop", LoopBoundary = "end", InputIsList = true,
    OutputIsList = [true], OutputNames = ["outputs"], OutputMatchTypes = ["output_value"])]
public sealed class EndLoop
{
    /// <summary>
    /// The loop's result: the last iteration's output_value, or every iteration's (one after another) with accumulate. next_iteration_value goes back to the Start
    /// Loop for the next iteration; terminations are outputs that must run every iteration (a preview, a save) and are not returned.
    /// </summary>
    public IReadOnlyList<object?> Execute(
        [Input(Default = false, Tooltip = "Return output_value from every iteration when enabled; otherwise return only the final iteration.")] IReadOnlyList<bool> accumulate,
        NodeContext context,
        [Input(MatchType = "output_value", Optional = true, Tooltip = "Value returned by End Loop. It returns the final iteration or all iterations according to accumulate.")]
        IReadOnlyList<object?>? outputValue = null,
        [Input(MatchType = "carried_value", Optional = true, Tooltip = "Value sent from End Loop back to Start Loop for the next iteration.")]
        IReadOnlyList<object?>? nextIterationValue = null,
        [Autogrow(Prefix = "termination", Min = 0, Max = 50), Input(Type = "*", Optional = true, Tooltip = "Connect outputs that must execute on every iteration. Their values are not returned.")]
        IReadOnlyList<IReadOnlyList<object?>>? terminations = null)
    {
        var outputs = (context.Execution ?? throw new InvalidOperationException("End Loop runs inside a prompt run.")).ExternalBlockResult(context.NodeId) as List<IReadOnlyList<object?>> ?? [];
        return [.. outputs.SelectMany(o => o)];
    }
}
