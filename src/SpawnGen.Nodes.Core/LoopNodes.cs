using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// What a Start Loop hands to its End Loop (through the run's external block): the number of iterations and, for each, the link that gives that iteration's
/// output_value (null when the End Loop's output_value is not connected).
/// </summary>
public sealed class LoopResult(int iterations, IReadOnlyList<PromptInput?> outputs)
{
    public int Iterations { get; } = iterations;
    public IReadOnlyList<PromptInput?> Outputs { get; } = outputs;
}

/// <summary>
/// Start Loop: runs the nodes between it and its End Loop (the body, as validation found it) once per iteration. Each iteration is a copy of the body added
/// to the running prompt; a <see cref="LoopIteration"/> stands in for this node's outputs in the copy, and a <see cref="LoopProgress"/> after it reports the
/// iteration and makes sure the iteration's results and terminations run, in order.
/// </summary>
/// <remarks>
/// Measured with ComfyUI (Fixtures/loops and further graphs):
/// simple runs num_iterations times with index 0, 1, ...; For runs index = start, start + step, ... while below max_iteration (none when start is not
/// below it, negative starts allowed); List runs once per item of the list (one value counts as a list of one), index 0, 1, .... is_first and is_last mark
/// the first and the last iteration. list_item is None outside List mode. current_iteration_value is initial_iteration_value (None when not connected) on the
/// first iteration, then what the previous iteration gave next_iteration_value; when that is not connected the initial value stays. A list carried value
/// stays a list. parent_iteration only places the loop inside an outer one: its value is not used.
/// </remarks>
[Node("StartLoop", Category = "utilities/looping", DisplayName = "Start Loop", InputIsList = true, LoopBoundary = "start",
    OutputMatchTypes = [null, null, null, "list_item", "carried_value"])]
public sealed class StartLoop
{
    public static IEnumerable<DynamicOption> Modes() =>
    [
        new("simple", DynamicInput.Int("num_iterations", 4, min: 0, tooltip: "Number of times to execute the loop body.")),
        new("For",
            DynamicInput.Int("start_iteration_index", 0, tooltip: "Index of the first iteration when using For loop mode."),
            DynamicInput.Int("max_iteration", 4, max: 18446744073709551615d, tooltip: "The exclusive stopping value for iteration_index in For mode."),
            DynamicInput.Int("step", 1, min: 1, tooltip: "The index step size between each iteration when using For loop mode.")),
        new("List", DynamicInput.MatchType("list", "list_item", tooltip: "List of items the loop iterates on. The loop body executes once per item.")),
    ];

    public (long IterationIndex, bool IsFirst, bool IsLast, object? ListItem, object? CurrentIterationValue, NodeExpansion Expansion) Execute(
        NodeContext context,
        [DynamicCombo(nameof(Modes)), Input(Tooltip = "The loop iteration mode.")] DynamicChoice mode,
        [Input(Default = false, Tooltip = "Reuse unchanged iteration results from previous executions. Disable to execute every iteration again.")]
        IReadOnlyList<bool> cacheIterations,
        [Input(ForceInput = true, Optional = true, Tooltip = "Connect iteration_index from an outer Start Loop to nest this loop.")]
        IReadOnlyList<long>? parentIteration = null,
        [Input(MatchType = "carried_value", Optional = true, Tooltip = "Value exposed as current_iteration_value on the first iteration.")]
        IReadOnlyList<object?>? initialIterationValue = null)
    {
        var scope = context.Execution ?? throw new InvalidOperationException("Start Loop runs only inside a prompt run.");
        var startId = context.NodeId;
        var loop = scope.Loop(startId) ?? throw new InvalidOperationException($"Start Loop {startId} has no End Loop.");
        var plan = Plan(mode);
        bool reuse = cacheIterations is { Count: > 0 } && cacheIterations[0];

        var start = scope.Node(startId);
        var end = scope.Node(loop.End);
        var body = new HashSet<string>(loop.Body, StringComparer.Ordinal);
        var bodyOrder = loop.Body.ToList();
        string Copy(int i, string id) => $"{startId}.{i}.{id}";
        string IterationId(int i) => $"{startId}.{i}#iteration";
        PromptInput Map(PromptInput input, int i) =>
            !input.IsLink ? input
            : input.NodeId == startId ? PromptInput.Link(IterationId(i), input.OutputIndex)
            : body.Contains(input.NodeId!) ? PromptInput.Link(Copy(i, input.NodeId!), input.OutputIndex)
            : input;
        PromptInput? EndInput(string name, int i) => end.Inputs.TryGetValue(name, out var given) && given.IsLink ? Map(given, i) : null;

        var expansion = new NodeExpansion();
        start.Inputs.TryGetValue("initial_iteration_value", out var initial);
        var outputs = new List<PromptInput?>();
        int n = plan.Count;
        for (int i = 0; i < n; i++)
        {
            // The values of this iteration, standing in for the Start Loop's outputs.
            var iteration = expansion.Add(IterationId(i), "LoopIteration");
            iteration.Inputs["iteration_index"] = PromptInput.Value(plan.Index(i));
            iteration.Inputs["is_first"] = PromptInput.Value(i == 0);
            iteration.Inputs["is_last"] = PromptInput.Value(i == n - 1);
            iteration.Inputs["reuse_cache"] = PromptInput.Value(reuse);
            if (plan.Items is { } items) iteration.Inputs["list_item"] = PromptInput.Value(items[i]);
            var carried = i > 0 && EndInput("next_iteration_value", i - 1) is { } next ? next : initial.IsLink ? initial : (PromptInput?)null;
            if (carried is { } c) iteration.Inputs["current_iteration_value"] = c;

            // The body, its links into the loop pointing at this iteration's copies; each copy shows as its original.
            foreach (var id in bodyOrder)
            {
                var original = scope.Node(id);
                var copy = expansion.Add(Copy(i, id), original.ClassType, displayId: id);
                copy.Title = original.Title;
                foreach (var (name, input) in original.Inputs) copy.Inputs[name] = Map(input, i);
                // An inner loop's copy: its body and End Loop are the copies of this iteration.
                if (scope.Loop(id) is { } inner)
                    expansion.Loops[Copy(i, id)] = new LoopScope(inner.Body.Select(b => body.Contains(b) ? Copy(i, b) : b).ToList(),
                        body.Contains(inner.End) ? Copy(i, inner.End) : inner.End);
            }

            // Runs after the iteration's results and terminations, so the iterations run one after another.
            var progress = expansion.Add($"{startId}.{i}#progress", "LoopProgress");
            progress.Inputs["start_id"] = PromptInput.Value(startId);
            progress.Inputs["position"] = PromptInput.Value((long)(i + 1));
            progress.Inputs["total"] = PromptInput.Value((long)n);
            if (EndInput("output_value", i) is { } value) progress.Inputs["output_value"] = value;
            if (EndInput("next_iteration_value", i) is { } nextValue) progress.Inputs["next_iteration_value"] = nextValue;
            foreach (var (name, input) in end.Inputs)
                if (name.StartsWith("terminations.", StringComparison.Ordinal) && input.IsLink) progress.Inputs[name] = Map(input, i);

            outputs.Add(EndInput("output_value", i));
        }
        scope.ReleaseExternalBlock(loop.End, new LoopResult(n, outputs));

        // The Start Loop's own outputs only feed the body, whose copies read the iteration nodes instead.
        object? firstItem = plan.Items is { Count: > 0 } list ? list[0] : null;
        object? firstValue = initialIterationValue is { Count: > 0 } values ? values[0] : null;
        return (n > 0 ? plan.Index(0) : 0, true, n == 1, firstItem, firstValue, expansion);
    }

    /// <summary>The iterations a mode asks for: how many, the index of each, and the items in List mode.</summary>
    sealed record IterationPlan(int Count, Func<int, long> Index, IReadOnlyList<object?>? Items);

    static IterationPlan Plan(DynamicChoice mode)
    {
        switch (mode.Key)
        {
            case "simple":
            {
                long count = Math.Max(0, First(mode, "num_iterations", 4));
                return new IterationPlan(checked((int)count), i => i, null);
            }
            case "For":
            {
                long start = First(mode, "start_iteration_index", 0), max = First(mode, "max_iteration", 4), step = First(mode, "step", 1);
                if (step < 1) throw new ArgumentException("The For loop step must be at least 1.");
                long count = start < max ? (long)Math.Floor(((decimal)max - start + step - 1) / step) : 0;
                return new IterationPlan(checked((int)count), i => start + i * step, null);
            }
            case "List":
            {
                mode.Values.TryGetValue("list", out var raw);
                IReadOnlyList<object?> items = raw switch { IReadOnlyList<object?> given => given, null => [], _ => [raw] };
                return new IterationPlan(items.Count, i => i, items);
            }
            default:
                throw new ArgumentException($"Unknown loop mode '{mode.Key}'.");
        }
    }

    /// <summary>The first value of a mode's input (a list node receives whole lists).</summary>
    static long First(DynamicChoice mode, string name, long fallback)
    {
        if (!mode.Values.TryGetValue(name, out var raw) || raw is null) return fallback;
        var value = raw is IReadOnlyList<object?> list ? list.Count > 0 ? list[0] : null : raw;
        return value is null ? fallback : Convert.ToInt64(value, System.Globalization.CultureInfo.InvariantCulture);
    }
}

/// <summary>
/// End Loop: returns what the iterations of its Start Loop gave output_value: all of them in order with accumulate (each iteration's list joined), else the
/// last iteration's. Measured with ComfyUI: with no iterations, or with output_value not connected, the result is a single None.
/// </summary>
[Node("EndLoop", Category = "utilities/looping", DisplayName = "End Loop", InputIsList = true, LoopBoundary = "end",
    OutputIsList = [true], OutputNames = ["outputs"], OutputMatchTypes = ["output_value"])]
public sealed class EndLoop
{
    /// <remarks>The run does not evaluate output_value, next_iteration_value or the terminations here: the iterations' copies of them do.</remarks>
    public (IReadOnlyList<object?> Outputs, NodeExpansion Expansion) Execute(
        NodeContext context,
        [Input(Default = false, Tooltip = "Return output_value from every iteration when enabled; otherwise return only the final iteration.")]
        IReadOnlyList<bool> accumulate,
        [Input(MatchType = "output_value", Optional = true, Tooltip = "Value returned by End Loop. It returns the final iteration or all iterations according to accumulate.")]
        IReadOnlyList<object?>? outputValue = null,
        [Input(MatchType = "carried_value", Optional = true, Tooltip = "Value sent from End Loop back to Start Loop for the next iteration.")]
        IReadOnlyList<object?>? nextIterationValue = null,
        [Autogrow(Prefix = "termination", Min = 0, Max = 50),
         Input(Optional = true, Tooltip = "Connect a preview or side-effect output that must execute on every iteration. Its value is not returned.")]
        IReadOnlyList<IReadOnlyList<object?>>? terminations = null)
    {
        var scope = context.Execution ?? throw new InvalidOperationException("End Loop runs only inside a prompt run.");
        if (scope.ExternalBlockResult(context.NodeId) is not LoopResult result)
            throw new InvalidOperationException($"End Loop {context.NodeId} has no result from its Start Loop.");
        bool all = accumulate is { Count: > 0 } && accumulate[0];
        var links = (all ? result.Outputs : result.Outputs.Count > 0 ? [result.Outputs[^1]] : []).ToList();
        // Links are resolved by the run (each brings its whole list); none connected, or no iterations, gives one None.
        var values = links.Count == 0 || links.Any(l => l is null) ? new List<object?> { null } : links.Select(l => (object?)l!.Value).ToList();
        return (values, new NodeExpansion());
    }
}

/// <summary>
/// One iteration's values (made by Start Loop, not offered in the editor): the index, whether it is the first and the last, the list item and the carried value
/// (passed on whole). It is served from the cache only when the loop asks to reuse iterations.
/// </summary>
[Node("LoopIteration", Category = "sd", DisplayName = "LoopIteration", InputIsList = true, DevOnly = true, CacheWhen = "reuse_cache",
    OutputIsList = [false, false, false, false, true], OutputNames = ["INT", "BOOLEAN", "BOOLEAN", "*", "*"])]
public sealed class LoopIteration
{
    public (long, bool, bool, object?, IReadOnlyList<object?>) Execute(
        IReadOnlyList<long> iterationIndex, IReadOnlyList<bool> isFirst, IReadOnlyList<bool> isLast, IReadOnlyList<bool> reuseCache,
        [Input(Optional = true)] IReadOnlyList<object?>? listItem = null,
        [Input(Optional = true)] IReadOnlyList<object?>? currentIterationValue = null)
    {
        static T At<T>(IReadOnlyList<T>? list) => list is { Count: > 0 } ? list[0] : default!;
        return (At(iterationIndex), At(isFirst), At(isLast), At(listItem), currentIterationValue ?? [null]);
    }
}

/// <summary>
/// Reports a loop's progress on its Start Loop ("Iteration 2 / 5", made by Start Loop, not offered in the editor). The Start Loop also links it to the
/// iteration's output value, carried value and terminations, so it runs after them.
/// </summary>
[Node("LoopProgress", Category = "sd", DisplayName = "LoopProgress", InputIsList = true, DevOnly = true, IsOutput = true, AlwaysExecute = true,
    AcceptAllInputs = true, OutputNames = ["INT"])]
public sealed class LoopProgress
{
    public long Execute(NodeContext context, IReadOnlyList<string> startId, IReadOnlyList<long> position, IReadOnlyList<long> total,
        [ExtraInputs] IReadOnlyDictionary<string, object?> results)
    {
        long at = position is { Count: > 0 } ? position[0] : 0, of = total is { Count: > 0 } ? total[0] : 0;
        context.Progress(at, of, $"Iteration {at} / {of}");
        return at;
    }
}
