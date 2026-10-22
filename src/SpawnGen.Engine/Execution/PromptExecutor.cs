using System.Diagnostics;
using System.Globalization;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Engine.Execution;

/// <summary>
/// Runs a <see cref="PromptGraph"/>: validate, plan, then execute the needed nodes in dependency order, reusing
/// cached results for any node whose inputs have not changed. Runs are serialized; a second call waits its turn.
/// </summary>
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
            var promptJson = new Lazy<string>(() => PromptJson.ToJson(graph)); // only built if a node asks for it (Save Image does)
            var stopwatch = Stopwatch.StartNew();
            observer?.OnStart(promptId, validation.Plan);

            var values = new Dictionary<string, NodeOutputs>(StringComparer.Ordinal);
            var signatures = new Dictionary<string, string>(StringComparer.Ordinal);
            var executed = new List<string>();
            var cached = new List<string>();

            foreach (var id in validation.Plan)
            {
                string? current = id;
                try
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var node = graph.Nodes[id];
                    _registry.TryGet(node.ClassType, out var registered);

                    var bound = Bind(registered, node, values, signatures, out var signature);
                    signatures[id] = signature;

                    if (!registered.Schema.AlwaysExecute && Cache.TryGet(signature, out var hit))
                    {
                        values[id] = hit;
                        cached.Add(id);
                        observer?.OnNodeCached(id);
                        continue;
                    }

                    observer?.OnNodeExecuting(id);
                    var context = new NodeContext(id, node.ClassType, _services, cancellationToken,
                        (value, max, message) => observer?.OnNodeProgress(id, value, max, message), promptJson, options?.ExtraPngInfo);

                    NodeOutputs result;
                    try
                    {
                        result = await registered.InvokeAsync(bound, context).ConfigureAwait(false);
                    }
                    catch (OperationCanceledException) { throw; }
                    catch (Exception ex)
                    {
                        throw new NodeExecutionException(id, node.ClassType, ex);
                    }

                    if (result.Values.Length != registered.Schema.Outputs.Count)
                        throw new NodeExecutionException(id, node.ClassType,
                            new InvalidOperationException($"Produced {result.Values.Length} outputs; the schema declares {registered.Schema.Outputs.Count}."));

                    values[id] = result;
                    if (!registered.Schema.AlwaysExecute) Cache.Set(signature, result);
                    executed.Add(id);
                    observer?.OnNodeExecuted(id, result);
                }
                catch (Exception ex)
                {
                    observer?.OnError(ex is NodeExecutionException nee ? nee.NodeId : current, ex);
                    throw;
                }
            }

            Cache.Trim();
            var outputs = validation.OutputNodeIds.Distinct(StringComparer.Ordinal).ToDictionary(id => id, id => values[id], StringComparer.Ordinal);
            var finished = new ExecutionResult
            {
                PromptId = promptId,
                Outputs = outputs,
                Executed = executed,
                Cached = cached,
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

    /// <summary>Resolves a node's inputs to values and computes its cache signature.</summary>
    Dictionary<string, object?> Bind(RegisteredNode registered, PromptNode node, Dictionary<string, NodeOutputs> values,
        Dictionary<string, string> signatures, out string signature)
    {
        var bound = new Dictionary<string, object?>(StringComparer.Ordinal);
        var sig = new System.Text.StringBuilder(node.ClassType).Append('(');

        foreach (var (name, input) in node.Inputs.OrderBy(kv => kv.Key, StringComparer.Ordinal))
        {
            if (registered.Schema.FindInput(name) is null) continue; // extra keys (e.g. editor-only) are ignored

            if (input.IsLink)
            {
                bound[name] = values[input.NodeId!].Values[input.OutputIndex];
                sig.Append(name).Append("=@").Append(Cache.Intern(signatures[input.NodeId!])).Append(':').Append(input.OutputIndex).Append(';');
            }
            else if (input.Literal is { } literal)
            {
                var clr = registered.InputClrType(name);
                var typed = clr is not null && ValueCoercion.TryCoerce(literal, clr, out var t) ? t! : literal;
                bound[name] = typed;
                sig.Append(name).Append('=').Append(Canonical(typed)).Append(';');
            }
        }

        if (registered.Schema.AlwaysExecute) sig.Append('#').Append(Cache.NextUnique());
        sig.Append(')');
        signature = sig.ToString();
        return bound;
    }

    static string Canonical(object v) => v switch
    {
        string s => $"s{s.Length}:{s}",
        bool b => b ? "T" : "F",
        double d => "d" + d.ToString("R", CultureInfo.InvariantCulture),
        float f => "d" + ((double)f).ToString("R", CultureInfo.InvariantCulture),
        Enum e => "e" + e,
        IFormattable f => "n" + f.ToString(null, CultureInfo.InvariantCulture),
        _ => "o" + v,
    };
}
