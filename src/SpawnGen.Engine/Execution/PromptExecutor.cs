using System.Diagnostics;
using System.Globalization;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Engine.Execution;

/// <summary>
/// Runs a <see cref="PromptGraph"/> the way ComfyUI's executor does: validate, then evaluate each output node, pulling in what it needs.
/// A node whose inputs have not changed is served from the cache without looking further upstream. Runs are serialized; a second call waits its turn.
/// </summary>
/// <remarks>
/// Values on a socket are lists, as in ComfyUI. A node runs once per item of its longest list input (shorter lists repeat their last item) and its
/// results are gathered into lists again; a node marked <see cref="NodeAttribute.InputIsList"/> runs once and sees the whole lists, and an output
/// marked <see cref="NodeAttribute.OutputIsList"/> contributes its items one by one. Lazy inputs are only evaluated when the node's
/// <c>CheckLazyStatus</c> asks for them, and an <see cref="ExecutionBlocker"/> arriving on any input stops the node from running.
/// </remarks>
public sealed class PromptExecutor
{
    sealed class NoServices : IServiceProvider
    {
        public static readonly NoServices Instance = new();
        public object? GetService(Type serviceType) => null;
    }

    readonly NodeRegistry _registry;
    readonly IServiceProvider _services;
    readonly SemaphoreSlim _gate = new(1, 1);

    public PromptExecutor(NodeRegistry registry, IServiceProvider? services = null, ResultCache? cache = null)
    {
        _registry = registry;
        _services = services ?? NoServices.Instance;
        Cache = cache ?? new ResultCache();
    }

    public ResultCache Cache { get; }

    public ValidationResult Validate(PromptGraph graph, IReadOnlyList<string>? outputNodeIds = null) =>
        PromptValidator.Validate(_registry, graph, outputNodeIds);

    public async Task<ExecutionResult> ExecuteAsync(PromptGraph graph, ExecutionOptions? options = null,
        IExecutionObserver? observer = null, CancellationToken cancellationToken = default)
    {
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var validation = PromptValidator.Validate(_registry, graph, options?.OutputNodeIds);
            if (!validation.IsValid)
            {
                var ex = new PromptValidationException(validation);
                observer?.OnError(null, ex);
                throw ex;
            }

            var promptId = options?.PromptId ?? Guid.NewGuid().ToString("N");
            var stopwatch = Stopwatch.StartNew();
            observer?.OnStart(promptId, validation.Plan);

            var run = new Run(this, graph, options, observer, cancellationToken, validation.Loops);
            // An output node inside a loop body runs once per iteration, through the loop (ComfyUI inhibits the original).
            var inLoops = validation.Loops.Values.SelectMany(l => l.Body).ToHashSet(StringComparer.Ordinal);
            try
            {
                foreach (var id in validation.OutputNodeIds)
                    if (!inLoops.Contains(id)) await run.EvaluateAsync(id).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                observer?.OnError(ex is NodeExecutionException nee ? nee.NodeId : run.Current, ex);
                throw;
            }

            Cache.Trim();
            var outputs = validation.OutputNodeIds.Distinct(StringComparer.Ordinal).Where(run.Values.ContainsKey)
                .ToDictionary(id => id, id => run.Values[id], StringComparer.Ordinal);
            var finished = new ExecutionResult
            {
                PromptId = promptId,
                Outputs = outputs,
                Executed = run.Executed,
                Cached = run.CachedIds,
                Elapsed = stopwatch.Elapsed,
            };
            observer?.OnFinished(finished);
            return finished;
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <summary>
    /// The state of one run: what has been evaluated, every node's cache signature, and what expansions added (ComfyUI's dynprompt: the added nodes, the node each
    /// shows as, the loops among them, and the results handed to End Loops).
    /// </summary>
    sealed class Run(PromptExecutor owner, PromptGraph graph, ExecutionOptions? options, IExecutionObserver? observer, CancellationToken cancellationToken,
        IReadOnlyDictionary<string, LoopScope> loops) : IExecutionScope
    {
        readonly Lazy<string> _promptJson = new(() => PromptJson.ToJson(graph)); // only built if a node asks for it (Save Image does)
        readonly Dictionary<string, string> _signatures = new(StringComparer.Ordinal);
        readonly HashSet<string> _evaluating = new(StringComparer.Ordinal);
        readonly Dictionary<string, PromptNode> _added = new(StringComparer.Ordinal);
        readonly Dictionary<string, string> _display = new(StringComparer.Ordinal);
        readonly Dictionary<string, LoopScope> _loops = new(loops, StringComparer.Ordinal);
        readonly Dictionary<string, string> _endToStart = loops.ToDictionary(l => l.Value.End, l => l.Key, StringComparer.Ordinal);
        readonly Dictionary<string, object?> _external = new(StringComparer.Ordinal);

        public PromptNode Node(string id) => _added.TryGetValue(id, out var added) ? added : graph.Nodes[id];
        bool Exists(string id) => _added.ContainsKey(id) || graph.Nodes.ContainsKey(id);
        public LoopScope? Loop(string startId) => _loops.GetValueOrDefault(startId);
        public void ReleaseExternalBlock(string nodeId, object? value) => _external[nodeId] = value;
        public object? ExternalBlockResult(string nodeId) => _external.GetValueOrDefault(nodeId);

        /// <summary>The node the editor shows an added node as (ComfyUI's get_display_node_id: followed to the original).</summary>
        string Display(string id)
        {
            while (_display.TryGetValue(id, out var shown)) id = shown;
            return id;
        }

        public Dictionary<string, NodeOutputs> Values { get; } = new(StringComparer.Ordinal);
        public List<string> Executed { get; } = [];
        public List<string> CachedIds { get; } = [];
        public string? Current { get; private set; }

        public async Task<NodeOutputs> EvaluateAsync(string id)
        {
            if (Values.TryGetValue(id, out var done)) return done;
            if (!_evaluating.Add(id)) throw new InvalidOperationException($"Node {id} is part of a cycle.");
            cancellationToken.ThrowIfCancellationRequested();

            var node = Node(id);
            if (!owner._registry.TryGet(node.ClassType, out var registered))
                throw new NodeExecutionException(id, node.ClassType, new InvalidOperationException($"Unknown node type '{node.ClassType}'."));
            var schema = registered.Schema;
            var signature = Signature(id);

            if (!AlwaysRuns(node, schema) && owner.Cache.TryGet(signature, out var hit))
            {
                ReportCachedUpstream(node);
                Values[id] = hit;
                CachedIds.Add(id);
                observer?.OnNodeCached(Display(id));
                return hit;
            }

            // An End Loop takes its result from its loop: the Start Loop expands the body and the expansion's LoopResult hands the iterations' outputs over.
            // The body links themselves are not followed (ComfyUI overrides the End Loop's inputs without them).
            bool endLoop = schema.LoopBoundary == "end";
            if (endLoop)
            {
                if (!_endToStart.TryGetValue(id, out var startId))
                    throw new NodeExecutionException(id, node.ClassType, new InvalidOperationException("This End Loop has no Start Loop."));
                if (!_external.ContainsKey(id)) await EvaluateAsync(startId).ConfigureAwait(false);
            }

            // Every input as a list: a link brings the list its source made, a value is a list of one.
            var lists = new Dictionary<string, IReadOnlyList<object?>>(StringComparer.Ordinal);
            var unevaluated = new Dictionary<string, PromptInput>(StringComparer.Ordinal);
            foreach (var (name, input) in node.Inputs)
            {
                if (endLoop && (name is "output_value" or "next_iteration_value" || name.StartsWith("termination", StringComparison.Ordinal))) continue;
                var spec = schema.FindInput(name);
                if (spec is null && !schema.AcceptAllInputs) continue; // extra keys (e.g. editor-only) are ignored
                if (spec is null)
                {
                    lists[name] = input.IsLink ? (await EvaluateAsync(input.NodeId!).ConfigureAwait(false)).List(input.OutputIndex) : [input.Literal];
                    continue;
                }
                if (input.IsLink)
                {
                    if (spec.Lazy && registered.HasLazyCheck)
                    {
                        unevaluated[name] = input;
                        lists[name] = [null];
                    }
                    else lists[name] = (await EvaluateAsync(input.NodeId!).ConfigureAwait(false)).List(input.OutputIndex);
                }
                else if (input.Literal is { } literal) lists[name] = [Coerce(registered, name, literal)];
            }

            Current = id;
            // ComfyUI's _map_node_over_list: one call per item of the longest list; when every list is empty (or there are none) one call with no inputs.
            int Calls() => schema.InputIsList ? 1 : Math.Max(1, lists.Count == 0 ? 0 : lists.Values.Max(l => l.Count));
            int calls = Calls();
            if (!schema.InputIsList && lists.Values.Any(l => l.Count > 0) && lists.FirstOrDefault(kv => kv.Value.Count == 0) is { Key: { } emptyName })
                throw new NodeExecutionException(id, node.ClassType, new InvalidOperationException($"Input '{emptyName}' is an empty list, so there is nothing to run on."));

            // Lazy inputs: ask the node what it needs, evaluate that, and ask again until it needs nothing more.
            while (unevaluated.Count > 0)
            {
                var wanted = new HashSet<string>(StringComparer.Ordinal);
                for (int i = 0; i < calls; i++)
                {
                    var slice = Slice(schema, lists, i);
                    if (slice.Values.Any(v => v is ExecutionBlocker)) continue;
                    try
                    {
                        wanted.UnionWith(await registered.CheckLazyAsync(slice, Context(id, node, i)).ConfigureAwait(false));
                    }
                    catch (OperationCanceledException) { throw; }
                    catch (Exception ex) { throw new NodeExecutionException(id, node.ClassType, ex); }
                }
                var next = unevaluated.Keys.Where(wanted.Contains).ToList();
                if (next.Count == 0) break;
                foreach (var name in next)
                {
                    var input = unevaluated[name];
                    unevaluated.Remove(name);
                    lists[name] = (await EvaluateAsync(input.NodeId!).ConfigureAwait(false)).List(input.OutputIndex);
                }
                Current = id;
                calls = Calls();
            }

            observer?.OnNodeExecuting(Display(id));
            var results = new List<(NodeOutputs? Outputs, ExecutionBlocker? Blocked)>(calls);
            for (int i = 0; i < calls; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var slice = Slice(schema, lists, i);
                var blocker = slice.Values.SelectMany(v => schema.InputIsList && v is IReadOnlyList<object?> items ? items : [v]).OfType<ExecutionBlocker>().FirstOrDefault();
                if (blocker is not null)
                {
                    if (blocker.Message is not null) observer?.OnError(Display(id), new ExecutionBlockedException(Display(id), blocker.Message));
                    results.Add((null, new ExecutionBlocker())); // reported once here; what follows is blocked quietly
                    continue;
                }
                NodeOutputs one;
                try
                {
                    one = await registered.InvokeAsync(slice, Context(id, node, i)).ConfigureAwait(false);
                }
                catch (OperationCanceledException) { throw; }
                catch (Exception ex) { throw new NodeExecutionException(id, node.ClassType, ex); }
                if (one.Values.Length != schema.Outputs.Count)
                    throw new NodeExecutionException(id, node.ClassType,
                        new InvalidOperationException($"Produced {one.Values.Length} outputs; the schema declares {schema.Outputs.Count}."));
                results.Add((one, null));
            }

            var result = Merge(schema, results);
            var expansions = results.Select(r => r.Outputs?.Expansion).OfType<NodeExpansion>().ToList();
            if (expansions.Count > 0) result = await ExpandAsync(id, result, expansions).ConfigureAwait(false);
            Values[id] = result;
            _evaluating.Remove(id);
            if (!AlwaysRuns(node, schema) && expansions.Count == 0) owner.Cache.Set(signature, result);
            Executed.Add(id);
            observer?.OnNodeExecuted(Display(id), result);
            return result;
        }

        /// <summary>
        /// ComfyUI's expansion: the added nodes join the running prompt (each shown as its display node, or as the node that added it), their output nodes run
        /// (in the order they were added), and outputs that are links into them become the linked values.
        /// </summary>
        async Task<NodeOutputs> ExpandAsync(string id, NodeOutputs result, List<NodeExpansion> expansions)
        {
            foreach (var expansion in expansions)
            {
                foreach (var (addedId, added) in expansion.Nodes)
                {
                    if (Exists(addedId)) throw new NodeExecutionException(id, Node(id).ClassType, new InvalidOperationException($"Node {addedId} is added twice."));
                    _added[addedId] = added;
                    _display[addedId] = expansion.DisplayIds.GetValueOrDefault(addedId, id);
                }
                foreach (var (startId, scope) in expansion.Loops)
                {
                    _loops[startId] = scope;
                    _endToStart[scope.End] = startId;
                }
            }
            foreach (var expansion in expansions)
                foreach (var addedId in expansion.Nodes.Keys)
                    if (owner._registry.GetSchema(Node(addedId).ClassType)?.IsOutputNode == true) await EvaluateAsync(addedId).ConfigureAwait(false);

            int count = result.Values.Length;
            var lists = new List<object?>[count];
            bool linked = false;
            for (int o = 0; o < count; o++)
            {
                lists[o] = [];
                foreach (var value in result.List(o))
                {
                    if (value is PromptInput { IsLink: true } link)
                    {
                        linked = true;
                        lists[o].AddRange((await EvaluateAsync(link.NodeId!).ConfigureAwait(false)).List(link.OutputIndex));
                    }
                    else lists[o].Add(value);
                }
            }
            if (!linked) return result;
            return new NodeOutputs(lists.Select(l => l.Count > 0 ? l[0] : null).ToArray(), result.Ui) { Lists = lists.Any(l => l.Count != 1) ? lists : null };
        }

        /// <summary>Never served from the cache: an always-run node, or one whose cache switch (<see cref="NodeSchema.CacheWhen"/>) is not on.</summary>
        static bool AlwaysRuns(PromptNode node, NodeSchema schema) =>
            schema.AlwaysExecute || schema.CacheWhen is { } input && !(node.Inputs.TryGetValue(input, out var given) && given.Literal is true);

        /// <summary>
        /// A cached node needs nothing upstream, but ComfyUI still reports every node of the prompt that is in the cache: walk up and report
        /// (and hold on to) those whose results are cached too, in dependency order. Nothing runs.
        /// </summary>
        void ReportCachedUpstream(PromptNode node)
        {
            foreach (var input in node.Inputs.Values.Where(i => i.IsLink))
            {
                var source = input.NodeId!;
                if (Values.ContainsKey(source) || !Exists(source)) continue;
                var upstream = Node(source);
                if (!owner._registry.TryGet(upstream.ClassType, out var upReg) || AlwaysRuns(upstream, upReg.Schema)) continue;
                if (!owner.Cache.TryGet(Signature(source), out var cachedUp)) continue;
                ReportCachedUpstream(upstream);
                Values[source] = cachedUp;
                CachedIds.Add(source);
                observer?.OnNodeCached(Display(source));
            }
        }

        NodeContext Context(string id, PromptNode node, int index) =>
            new(id, node.ClassType, owner._services, cancellationToken,
                (value, max, message) => observer?.OnNodeProgress(Display(id), value, max, message), _promptJson, options?.ExtraPngInfo,
                observer is null ? null : png => observer.OnNodePreview(Display(id), png),
                input => node.Inputs.TryGetValue(input, out var given) && (given.IsLink || given.Literal is not null), index) { Execution = this, PreviewMethod = options?.PreviewMethod };

        /// <summary>The inputs of call <paramref name="index"/>: item index of each list, or its last item when it is shorter (ComfyUI's slice_dict).</summary>
        static Dictionary<string, object?> Slice(NodeSchema schema, Dictionary<string, IReadOnlyList<object?>> lists, int index)
        {
            var slice = new Dictionary<string, object?>(StringComparer.Ordinal);
            foreach (var (name, list) in lists)
                slice[name] = schema.InputIsList ? list : list.Count == 0 ? null : list[Math.Min(index, list.Count - 1)];
            return slice;
        }

        /// <summary>ComfyUI's merge_result_data: list outputs are concatenated, the others collect one value per call; previews are joined.</summary>
        static NodeOutputs Merge(NodeSchema schema, List<(NodeOutputs? Outputs, ExecutionBlocker? Blocked)> results)
        {
            int count = schema.Outputs.Count;
            var merged = new List<object?>[count];
            for (int o = 0; o < count; o++) merged[o] = [];
            UiOutput? ui = null;
            foreach (var (outputs, blocked) in results)
            {
                for (int o = 0; o < count; o++)
                {
                    if (outputs is null) { merged[o].Add(blocked); continue; }
                    var value = outputs.Values[o];
                    if (schema.Outputs[o].IsList && value is System.Collections.IEnumerable items and not string)
                        foreach (var item in items) merged[o].Add(item);
                    else merged[o].Add(value);
                }
                if (outputs?.Ui is { } part) ui = JoinUi(ui, part);
            }
            var first = merged.Select(l => l.Count > 0 ? l[0] : null).ToArray();
            return new NodeOutputs(first, ui) { Lists = merged.Any(l => l.Count != 1) ? merged : null };
        }

        /// <summary>Previews of several calls: lists under the same key are joined (ComfyUI concatenates its ui dicts the same way).</summary>
        static UiOutput JoinUi(UiOutput? joined, UiOutput part)
        {
            if (joined is null) return part;
            var result = new UiOutput();
            foreach (var (key, value) in joined) result[key] = value;
            foreach (var (key, value) in part)
                result[key] = result.TryGetValue(key, out var before) && before is System.Collections.IEnumerable a and not string && value is System.Collections.IEnumerable b and not string
                    ? a.Cast<object?>().Concat(b.Cast<object?>()).ToList()
                    : value;
            return result;
        }

        static object Coerce(RegisteredNode registered, string name, object literal)
        {
            var clr = registered.InputClrType(name);
            return clr is not null && ValueCoercion.TryCoerce(literal, clr, out var typed) ? typed! : literal;
        }

        /// <summary>
        /// What a node's result depends on: its type, its values and the signatures of what feeds it. Computed from the graph alone, so a node
        /// found in the cache never needs anything upstream of it to run.
        /// </summary>
        string Signature(string id)
        {
            if (_signatures.TryGetValue(id, out var known)) return known;
            var node = Node(id);
            owner._registry.TryGet(node.ClassType, out var registered);
            var sig = new System.Text.StringBuilder(registered.Schema.Id).Append('(');
            foreach (var (name, input) in node.Inputs.OrderBy(kv => kv.Key, StringComparer.Ordinal))
            {
                if (registered.Schema.FindInput(name) is null && !registered.Schema.AcceptAllInputs) continue;
                if (input.IsLink)
                    sig.Append(name).Append("=@").Append(owner.Cache.Intern(Signature(input.NodeId!))).Append(':').Append(input.OutputIndex).Append(';');
                else if (input.Literal is { } literal)
                    sig.Append(name).Append('=').Append(Canonical(Coerce(registered, name, literal))).Append(';');
            }
            if (AlwaysRuns(node, registered.Schema)) sig.Append('#').Append(owner.Cache.NextUnique());
            sig.Append(')');
            return _signatures[id] = sig.ToString();
        }
    }

    /// <summary>A value with no text form of its own (a picture handed to an added node): known by identity, so two different ones never share a cache entry.</summary>
    static readonly System.Runtime.CompilerServices.ConditionalWeakTable<object, string> Identities = new();
    static long _identityCounter;

    static string Canonical(object v) => v switch
    {
        string s => $"s{s.Length}:{s}",
        bool b => b ? "T" : "F",
        double d => "d" + d.ToString("R", CultureInfo.InvariantCulture),
        float f => "d" + ((double)f).ToString("R", CultureInfo.InvariantCulture),
        Enum e => "e" + e,
        IFormattable f => "n" + f.ToString(null, CultureInfo.InvariantCulture),
        _ => "o" + Identities.GetValue(v, _ => v.GetType().Name + "#" + Interlocked.Increment(ref _identityCounter)),
    };
}
