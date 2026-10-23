using System.Reflection;
using System.Runtime.CompilerServices;
using Microsoft.Extensions.DependencyInjection;

namespace SpawnGen.Engine.Nodes;

/// <summary>The values a node produced: socket outputs in schema order plus optional editor preview data.</summary>
/// <remarks>
/// Like ComfyUI, every output is really a list: a node that ran once per item of a list input (or returned a list output) has several values
/// on one socket. <see cref="Values"/> holds the first value of each output; <see cref="Lists"/> holds them all (null when every list has one item).
/// </remarks>
public sealed record NodeOutputs(object?[] Values, UiOutput? Ui)
{
    public IReadOnlyList<object?>[]? Lists { get; init; }

    /// <summary>Every value of output <paramref name="index"/>.</summary>
    public IReadOnlyList<object?> List(int index) => Lists is not null ? Lists[index] : [Values[index]];
}

/// <summary>
/// ComfyUI's <c>ExecutionBlocker</c>: returned as an output value it stops every node that would use it from running (output nodes simply do
/// nothing). With a message the run reports it as the reason.
/// </summary>
public sealed class ExecutionBlocker(string? message = null)
{
    public string? Message { get; } = message;
}

/// <summary>A node type: its schema plus how to invoke it.</summary>
public sealed class RegisteredNode
{
    internal enum ParamKind { Input, Context, CancellationToken, Services }

    internal sealed record ParamBinding(ParameterInfo Info, ParamKind Kind, string? InputName)
    {
        /// <summary>For a list parameter of an <see cref="NodeAttribute.InputIsList"/> node: the item type.</summary>
        public Type? ElementType { get; init; }
        /// <summary>For an Autogrow parameter: its member inputs (full names) and the short names the dictionary form uses.</summary>
        public (string Input, string Key)[]? Members { get; init; }
        /// <summary>For an Autogrow parameter: the type each member's value is coerced to.</summary>
        public Type? MemberType { get; init; }
        /// <summary>For a dynamic combo parameter: each option's inputs (full names) and the names the choice reports them by.</summary>
        public Dictionary<string, (string Input, string Key)[]>? DynamicOptions { get; init; }
        /// <summary>For a dynamic combo parameter: the option chosen when the prompt names none.</summary>
        public string? DynamicDefault { get; init; }
    }

    internal Type Type { get; init; } = default!;
    internal MethodInfo Method { get; init; } = default!;
    internal ParamBinding[] Params { get; init; } = default!;
    internal Type[] OutputTypes { get; init; } = default!;
    internal bool ReturnsTuple { get; init; } = default!;
    internal int UiIndex { get; init; } = default!;
    internal Type ResultType { get; init; } = default!;
    internal MethodInfo? LazyCheck { get; init; }
    internal ParamBinding[] LazyParams { get; init; } = [];

    /// <summary>The node decides itself which of its lazy inputs it needs (it has a <c>CheckLazyStatus</c> method).</summary>
    public bool HasLazyCheck => LazyCheck is not null;

    public required NodeSchema Schema { get; init; }

    /// <summary>A node known only by its schema (it lives on a server): it can be edited and validated here but not run here.</summary>
    public bool IsRemote { get; init; }

    /// <summary>CLR type an input's value must be coerced to.</summary>
    internal Type? InputClrType(string inputName) =>
        Params?.FirstOrDefault(p => p.Kind == ParamKind.Input && p.InputName == inputName) is { } p ? p.ElementType ?? p.Info.ParameterType : null; // a remote node has no parameters

    /// <summary>Asks the node which lazy inputs it needs, given what has been evaluated so far (null for a linked input not evaluated yet).</summary>
    internal async ValueTask<IReadOnlyList<string>> CheckLazyAsync(IReadOnlyDictionary<string, object?> inputs, NodeContext context)
    {
        var instance = LazyCheck!.IsStatic ? null : ActivatorUtilities.CreateInstance(context.Services, Type);
        try
        {
            var ret = await AwaitResultAsync(Call(LazyCheck, instance, BuildArgs(LazyParams, inputs, context))).ConfigureAwait(false);
            return ret is IEnumerable<string> names ? names.ToList() : [];
        }
        finally
        {
            (instance as IDisposable)?.Dispose();
        }
    }

    static object? Call(MethodInfo method, object? instance, object?[] args)
    {
        try
        {
            return method.Invoke(instance, args);
        }
        catch (TargetInvocationException tie) when (tie.InnerException is not null)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(tie.InnerException).Throw();
            throw;
        }
    }

    static object?[] BuildArgs(ParamBinding[] ps, IReadOnlyDictionary<string, object?> inputs, NodeContext context)
    {
        var args = new object?[ps.Length];
        for (var i = 0; i < ps.Length; i++)
        {
            var p = ps[i];
            switch (p.Kind)
            {
                case ParamKind.Context: args[i] = context; break;
                case ParamKind.CancellationToken: args[i] = context.CancellationToken; break;
                case ParamKind.Services: args[i] = context.Services; break;
                default:
                    if (p.Members is { } members)
                    {
                        args[i] = BuildGroup(p, members, inputs);
                        break;
                    }
                    if (p.DynamicOptions is { } options)
                    {
                        var key = inputs.TryGetValue(p.InputName!, out var chosen) && chosen is string s ? s : p.DynamicDefault!;
                        var values = new Dictionary<string, object?>(StringComparer.Ordinal);
                        if (options.TryGetValue(key, out var optionInputs))
                            foreach (var (input, name) in optionInputs)
                                if (inputs.TryGetValue(input, out var value)) values[name] = value;
                        args[i] = new DynamicChoice(key, values);
                        break;
                    }
                    if (p.ElementType is { } element && inputs.TryGetValue(p.InputName!, out var rawList) && rawList is IReadOnlyList<object?> items)
                    {
                        var array = Array.CreateInstance(element, items.Count);
                        for (int k = 0; k < items.Count; k++)
                        {
                            if (items[k] is null) continue;
                            if (!ValueCoercion.TryCoerce(items[k]!, element, out var typedItem))
                                throw new InvalidCastException($"Input '{p.InputName}' expects a list of {element.Name} but item {k} is {items[k]!.GetType().Name}.");
                            array.SetValue(typedItem, k);
                        }
                        args[i] = p.Info.ParameterType.IsAssignableFrom(array.GetType()) ? array : Activator.CreateInstance(p.Info.ParameterType, array);
                    }
                    else if (inputs.TryGetValue(p.InputName!, out var raw) && raw is not null)
                    {
                        if (!ValueCoercion.TryCoerce(raw, p.Info.ParameterType, out var typed))
                            throw new InvalidCastException(
                                $"Input '{p.InputName}' expects {p.Info.ParameterType.Name} but got {raw.GetType().Name}.");
                        args[i] = typed;
                    }
                    else
                    {
                        args[i] = p.Info.HasDefaultValue && p.Info.DefaultValue is not DBNull ? p.Info.DefaultValue : null;
                        if (args[i] is null && p.Info.ParameterType.IsValueType && Nullable.GetUnderlyingType(p.Info.ParameterType) is null)
                            args[i] = Activator.CreateInstance(p.Info.ParameterType);
                    }
                    break;
            }
        }
        return args;
    }

    /// <summary>An Autogrow parameter: the members that have a value, in order, as a list or as a dictionary by short name.</summary>
    static object BuildGroup(ParamBinding p, (string Input, string Key)[] members, IReadOnlyDictionary<string, object?> inputs)
    {
        var type = p.MemberType!;
        var keys = new List<string>();
        var values = new List<object?>();
        foreach (var (input, key) in members)
        {
            if (!inputs.TryGetValue(input, out var raw) || raw is null) continue;
            if (!ValueCoercion.TryCoerce(raw, type, out var typed))
                throw new InvalidCastException($"Input '{input}' expects {type.Name} but got {raw.GetType().Name}.");
            keys.Add(key);
            values.Add(typed);
        }
        var param = p.Info.ParameterType;
        if (param.IsGenericType && param.GetGenericArguments().Length == 2)
        {
            var dict = (System.Collections.IDictionary)Activator.CreateInstance(typeof(Dictionary<,>).MakeGenericType(typeof(string), type))!;
            for (int k = 0; k < keys.Count; k++) dict[keys[k]] = values[k];
            return dict;
        }
        var array = Array.CreateInstance(type, values.Count);
        for (int k = 0; k < values.Count; k++) array.SetValue(values[k], k);
        return param.IsAssignableFrom(array.GetType()) ? array : Activator.CreateInstance(param, array)!;
    }

    internal async ValueTask<NodeOutputs> InvokeAsync(IReadOnlyDictionary<string, object?> inputs, NodeContext context)
    {
        if (IsRemote) throw new NotSupportedException($"'{Schema.DisplayName}' is a node of the SpawnGen server (a custom node): run this graph on the server engine.");
        var args = BuildArgs(Params, inputs, context);
        var instance = Method.IsStatic ? null : ActivatorUtilities.CreateInstance(context.Services, Type);
        object? ret;
        try
        {
            ret = Call(Method, instance, args);
        }
        finally
        {
            (instance as IDisposable)?.Dispose();
        }

        ret = await AwaitResultAsync(ret).ConfigureAwait(false);

        var elements = ReturnsTuple ? Enumerate((ITuple)ret!)
            : ResultType == typeof(void) ? new List<object?>()
            : new List<object?> { ret };
        var values = new List<object?>(elements.Count);
        UiOutput? ui = null;
        for (var i = 0; i < elements.Count; i++)
        {
            if (i == UiIndex) ui = elements[i] as UiOutput;
            else values.Add(elements[i]);
        }
        return new NodeOutputs([.. values], ui);
    }

    static List<object?> Enumerate(ITuple tuple)
    {
        var list = new List<object?>(tuple.Length);
        for (var i = 0; i < tuple.Length; i++) list.Add(tuple[i]);
        return list;
    }

    static async ValueTask<object?> AwaitResultAsync(object? ret)
    {
        switch (ret)
        {
            case null: return null;
            case Task task:
                await task.ConfigureAwait(false);
                var tt = task.GetType();
                return tt.IsGenericType ? tt.GetProperty("Result")!.GetValue(task) : null;
            case ValueTask vt:
                await vt.ConfigureAwait(false);
                return null;
        }
        var type = ret.GetType();
        if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(ValueTask<>))
        {
            var asTask = (Task)type.GetMethod(nameof(ValueTask<int>.AsTask))!.Invoke(ret, null)!;
            await asTask.ConfigureAwait(false);
            return asTask.GetType().GetProperty("Result")!.GetValue(asTask);
        }
        return ret;
    }
}

/// <summary>Holds the node types a host can run and builds their schemas from <c>Execute</c> signatures.</summary>
public sealed class NodeRegistry
{
    readonly Dictionary<string, RegisteredNode> _nodes = new(StringComparer.Ordinal);

    public IEnumerable<NodeSchema> Schemas => _nodes.Values.Select(n => n.Schema);

    public bool TryGet(string id, out RegisteredNode node) => _nodes.TryGetValue(id, out node!) || _aliases.TryGetValue(id, out var current) && _nodes.TryGetValue(current, out node!);

    /// <summary>Answers for node types that are not registered (the editor's subgraph instances: their schema comes from a definition). Null = unknown.</summary>
    public Func<string, NodeSchema?>? SchemaResolver { get; set; }

    public NodeSchema? GetSchema(string id) => TryGet(id, out var n) ? n.Schema : SchemaResolver?.Invoke(id);

    readonly Dictionary<string, string> _aliases = new(StringComparer.Ordinal);

    /// <summary>The current id for <paramref name="id"/>: itself, or the node that lists it as an earlier id.</summary>
    public string CanonicalId(string id) => !_nodes.ContainsKey(id) && _aliases.TryGetValue(id, out var current) ? current : id;

    public NodeRegistry Add<T>() => Add(typeof(T));

    public NodeRegistry Add(Type type)
    {
        var node = Build(type);
        _nodes[node.Schema.Id] = node;
        foreach (var alias in type.GetCustomAttribute<NodeAttribute>()?.Aliases ?? []) _aliases[alias] = node.Schema.Id;
        return this;
    }

    /// <summary>
    /// Gives an input of a registered node another widget (a combo with other choices: the models a server has, say) and returns the one it had, so it can be put back.
    /// Null when the node or input is unknown.
    /// </summary>
    public WidgetSpec? ReplaceWidget(string nodeId, string input, WidgetSpec widget)
    {
        if (!_nodes.TryGetValue(nodeId, out var node) || node.Schema.FindInput(input) is not { } old) return null;
        var schema = new NodeSchema
        {
            Id = node.Schema.Id, DisplayName = node.Schema.DisplayName, Category = node.Schema.Category, Description = node.Schema.Description,
            Inputs = node.Schema.Inputs.Select(i => i.Name == input ? i with { Widget = widget } : i).ToList(),
            Outputs = node.Schema.Outputs, IsOutputNode = node.Schema.IsOutputNode, AlwaysExecute = node.Schema.AlwaysExecute, FrontendOnly = node.Schema.FrontendOnly,
            InputIsList = node.Schema.InputIsList, DynamicCombos = node.Schema.DynamicCombos,
        };
        _nodes[nodeId] = new RegisteredNode
        {
            LazyCheck = node.LazyCheck, LazyParams = node.LazyParams,
            Schema = schema, IsRemote = node.IsRemote, Type = node.Type, Method = node.Method, Params = node.Params, OutputTypes = node.OutputTypes,
            ReturnsTuple = node.ReturnsTuple, UiIndex = node.UiIndex, ResultType = node.ResultType,
        };
        return old.Widget;
    }

    /// <summary>Registers a node known only by its schema (a server's custom node seen from the Studio); an existing node of the same id is left alone.</summary>
    public NodeRegistry AddRemote(NodeSchema schema)
    {
        _nodes.TryAdd(schema.Id, new RegisteredNode { Schema = schema, IsRemote = true });
        return this;
    }

    /// <summary>Registers every public class marked <see cref="NodeAttribute"/> in the assembly.</summary>
    public NodeRegistry AddAssembly(Assembly assembly)
    {
        foreach (var type in assembly.GetExportedTypes())
            if (type.IsClass && type.GetCustomAttribute<NodeAttribute>() is not null)
                Add(type);
        return this;
    }

    // --- schema derivation --------------------------------------------------------------------------------

    static RegisteredNode Build(Type type)
    {
        var attr = type.GetCustomAttribute<NodeAttribute>();
        var id = attr?.Id ?? type.Name;

        var methods = type.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly)
            .Where(m => m.Name is "Execute" or "ExecuteAsync").ToArray();
        if (methods.Length != 1)
            throw new InvalidOperationException(
                $"Node '{id}' ({type.FullName}) must declare exactly one public Execute/ExecuteAsync method; found {methods.Length}.");
        var method = methods[0];

        var inputIsList = attr?.InputIsList ?? false;
        var dynamicCombos = new Dictionary<string, IReadOnlyDictionary<string, IReadOnlyList<string>>>(StringComparer.Ordinal);
        var inputs = new List<InputSpec>();
        var bindings = new List<RegisteredNode.ParamBinding>();
        foreach (var p in method.GetParameters())
        {
            if (p.ParameterType == typeof(NodeContext)) { bindings.Add(new(p, RegisteredNode.ParamKind.Context, null)); continue; }
            if (p.ParameterType == typeof(CancellationToken)) { bindings.Add(new(p, RegisteredNode.ParamKind.CancellationToken, null)); continue; }
            if (p.ParameterType == typeof(IServiceProvider)) { bindings.Add(new(p, RegisteredNode.ParamKind.Services, null)); continue; }

            var ia = p.GetCustomAttribute<InputAttribute>();
            var name = ia?.Name ?? ToSnakeCase(p.Name!);
            if (p.GetCustomAttribute<DynamicComboAttribute>() is { } dynamic)
            {
                var provider = type.GetMethod(dynamic.OptionsMethod, BindingFlags.Public | BindingFlags.Static, Type.EmptyTypes)
                    ?? throw new InvalidOperationException($"Node '{id}': dynamic combo options method '{dynamic.OptionsMethod}' not found as a public static method.");
                var options = ((IEnumerable<DynamicOption>)provider.Invoke(null, null)!).ToList();
                var keys = options.Select(o => o.Key).ToList();
                var chosenByDefault = ia?.Default as string ?? (p.HasDefaultValue ? p.DefaultValue as string : null) ?? keys[0];
                inputs.Add(new InputSpec(name, SocketType.String, Required: true, new WidgetSpec(WidgetKind.Combo) { Options = keys, Default = chosenByDefault }, ia?.Tooltip));
                // Options may share an input name (two ways to scale both have "width"): one input, brought by every option that has it.
                var nested = new Dictionary<string, (DynamicInput Spec, List<string> Options)>(StringComparer.Ordinal);
                var order = new List<string>();
                foreach (var option in options)
                    foreach (var input in option.Inputs)
                    {
                        if (!nested.TryGetValue(input.Name, out var entry)) { nested[input.Name] = entry = (input, []); order.Add(input.Name); }
                        entry.Options.Add(option.Key);
                    }
                foreach (var n in order)
                    inputs.Add(new InputSpec($"{name}.{n}", nested[n].Spec.Type, Required: false, nested[n].Spec.Widget, nested[n].Spec.Tooltip)
                    {
                        DynamicParent = name, DynamicOptions = nested[n].Options, Optional = !nested[n].Spec.Required,
                    });
                dynamicCombos[name] = options.ToDictionary(o => o.Key, o => (IReadOnlyList<string>)o.Inputs.Select(i => $"{name}.{i.Name}").ToList(), StringComparer.Ordinal);
                bindings.Add(new(p, RegisteredNode.ParamKind.Input, name)
                {
                    DynamicOptions = options.ToDictionary(o => o.Key, o => o.Inputs.Select(i => ($"{name}.{i.Name}", i.Name)).ToArray(), StringComparer.Ordinal),
                    DynamicDefault = chosenByDefault,
                });
                continue;
            }
            if (p.GetCustomAttribute<AutogrowAttribute>() is { } grow)
            {
                var (members, memberType, socketType) = Autogrow(id, p, name, grow, inputIsList);
                for (int m = 0; m < members.Length; m++)
                    inputs.Add(new InputSpec(members[m].Input, SocketTypes.FromClr(socketType), Required: m < grow.Min && !(ia?.Optional ?? false), Tooltip: ia?.Tooltip)
                    {
                        Group = name, GroupMin = grow.Min, Optional = m >= grow.Min || (ia?.Optional ?? false), Lazy = ia?.Lazy ?? false,
                    });
                bindings.Add(new(p, RegisteredNode.ParamKind.Input, name) { Members = members, MemberType = memberType });
                continue;
            }
            // A list node's parameter is IReadOnlyList<T>: the socket and widget are T's.
            var element = inputIsList ? ListElementType(p.ParameterType)
                ?? throw new InvalidOperationException($"Node '{id}' takes whole lists (InputIsList): parameter '{p.Name}' must be a list such as IReadOnlyList<T>.") : null;
            var valueType = element ?? p.ParameterType;
            inputs.Add(new InputSpec(name, ia?.Type is { } declared ? new SocketType(declared) : SocketTypes.FromClr(valueType), Required: !p.HasDefaultValue,
                Widget: BuildWidget(type, p, ia, valueType), Tooltip: ia?.Tooltip) { Aliases = ia?.Aliases ?? [], Lazy = ia?.Lazy ?? false, Optional = ia?.Optional ?? false });
            bindings.Add(new(p, RegisteredNode.ParamKind.Input, name) { ElementType = element });
        }
        var dup = inputs.GroupBy(i => i.Name).FirstOrDefault(g => g.Count() > 1);
        if (dup is not null) throw new InvalidOperationException($"Node '{id}' declares input '{dup.Key}' twice.");

        // Return type -> outputs. Unwrap Task/ValueTask, then flatten a value-tuple; a UiOutput element is not a socket.
        var returnType = method.ReturnType;
        if (returnType == typeof(Task) || returnType == typeof(ValueTask)) returnType = typeof(void);
        else if (returnType.IsGenericType &&
                 (returnType.GetGenericTypeDefinition() == typeof(Task<>) || returnType.GetGenericTypeDefinition() == typeof(ValueTask<>)))
            returnType = returnType.GetGenericArguments()[0];

        var returnsTuple = typeof(ITuple).IsAssignableFrom(returnType) && returnType.IsValueType && returnType.IsGenericType;
        Type[] elementTypes = returnType == typeof(void) ? [] : returnsTuple ? returnType.GetGenericArguments() : [returnType];
        if (returnsTuple && elementTypes.Length > 7)
            throw new InvalidOperationException($"Node '{id}': return at most 7 values.");
        var tupleNames = returnsTuple
            ? method.ReturnParameter.GetCustomAttribute<TupleElementNamesAttribute>()?.TransformNames
            : null;

        var outputs = new List<OutputSpec>();
        var uiIndex = -1;
        for (var i = 0; i < elementTypes.Length; i++)
        {
            if (elementTypes[i] == typeof(UiOutput)) { uiIndex = i; continue; }
            var isList = attr?.OutputIsList is { } flags && outputs.Count < flags.Length && flags[outputs.Count];
            var clr = isList ? ListElementType(elementTypes[i])
                ?? throw new InvalidOperationException($"Node '{id}': output {outputs.Count} is a list output (OutputIsList) and must return a list such as IReadOnlyList<T>.") : elementTypes[i];
            var socket = SocketTypes.FromClr(clr);
            var given = tupleNames is not null && i < tupleNames.Count ? tupleNames[i] : null;
            var named = attr?.OutputNames is { } outputNames && outputs.Count < outputNames.Length ? outputNames[outputs.Count] : null;
            outputs.Add(new OutputSpec(named ?? (given is null ? socket.Name : ToSnakeCase(given)), socket) { IsList = isList });
        }

        // An optional CheckLazyStatus method (same parameter names as Execute) says which lazy inputs the node needs.
        var lazyCheck = type.GetMethod("CheckLazyStatus", BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly);
        RegisteredNode.ParamBinding[] lazyParams = [];
        if (lazyCheck is not null)
            lazyParams = lazyCheck.GetParameters().Select(lp =>
                lp.ParameterType == typeof(NodeContext) ? new RegisteredNode.ParamBinding(lp, RegisteredNode.ParamKind.Context, null)
                : lp.ParameterType == typeof(CancellationToken) ? new RegisteredNode.ParamBinding(lp, RegisteredNode.ParamKind.CancellationToken, null)
                : lp.ParameterType == typeof(IServiceProvider) ? new RegisteredNode.ParamBinding(lp, RegisteredNode.ParamKind.Services, null)
                : bindings.FirstOrDefault(b => b.Kind == RegisteredNode.ParamKind.Input && b.Info.Name == lp.Name) is { } match
                    ? new RegisteredNode.ParamBinding(lp, RegisteredNode.ParamKind.Input, match.InputName) { ElementType = match.ElementType }
                    : throw new InvalidOperationException($"Node '{id}': CheckLazyStatus parameter '{lp.Name}' is not a parameter of Execute.")).ToArray();

        var schema = new NodeSchema
        {
            Id = id,
            DisplayName = attr?.DisplayName ?? SplitWords(type.Name),
            Category = attr?.Category ?? "utils",
            Description = attr?.Description,
            Inputs = inputs,
            Outputs = outputs,
            IsOutputNode = attr?.IsOutput ?? false,
            AlwaysExecute = attr?.AlwaysExecute ?? false,
            FrontendOnly = attr?.FrontendOnly ?? false,
            InputIsList = inputIsList,
            DynamicCombos = dynamicCombos,
        };

        return new RegisteredNode
        {
            LazyCheck = lazyCheck,
            LazyParams = lazyParams,
            Schema = schema,
            Type = type,
            Method = method,
            Params = [.. bindings],
            OutputTypes = elementTypes,
            ReturnsTuple = returnsTuple,
            UiIndex = uiIndex,
            ResultType = returnType,
        };
    }

    /// <summary>
    /// The members of an Autogrow parameter, the type each member's value is bound as, and the socket type (for a list node each member brings a
    /// whole list, so the socket is the list's item type).
    /// </summary>
    static ((string Input, string Key)[] Members, Type MemberType, Type SocketType) Autogrow(string nodeId, ParameterInfo p, string group, AutogrowAttribute grow, bool inputIsList)
    {
        var t = p.ParameterType;
        Type? memberType = t.IsGenericType && t.GetGenericArguments().Length == 2 && t.GetGenericArguments()[0] == typeof(string)
            ? t.GetGenericArguments()[1] : ListElementType(t);
        if (memberType is null)
            throw new InvalidOperationException($"Node '{nodeId}': Autogrow parameter '{p.Name}' must be IReadOnlyList<T> or IReadOnlyDictionary<string, T>.");
        var socket = inputIsList ? ListElementType(memberType)
            ?? throw new InvalidOperationException($"Node '{nodeId}' takes whole lists: Autogrow parameter '{p.Name}' must hold lists (IReadOnlyList<IReadOnlyList<T>>).") : memberType;
        var keys = grow.Names ?? (grow.Prefix is { } prefix ? Enumerable.Range(0, grow.Max).Select(i => prefix + i).ToArray()
            : throw new InvalidOperationException($"Node '{nodeId}': Autogrow parameter '{p.Name}' needs a Prefix or Names."));
        return (keys.Select(k => ($"{group}.{k}", k)).ToArray(), memberType, socket);
    }

    /// <summary>T for T[], IReadOnlyList&lt;T&gt;, List&lt;T&gt; and other IEnumerable&lt;T&gt; (not string); null otherwise.</summary>
    internal static Type? ListElementType(Type t)
    {
        if (t == typeof(string)) return null;
        if (t.IsArray) return t.GetElementType();
        if (t.IsGenericType && t.GetGenericTypeDefinition() == typeof(IEnumerable<>)) return t.GetGenericArguments()[0];
        return t.GetInterfaces().FirstOrDefault(i => i.IsGenericType && i.GetGenericTypeDefinition() == typeof(IEnumerable<>))?.GetGenericArguments()[0];
    }

    static WidgetSpec? BuildWidget(Type nodeType, ParameterInfo p, InputAttribute? ia, Type valueType)
    {
        var t = Nullable.GetUnderlyingType(valueType) ?? valueType;
        var def = p.HasDefaultValue && p.DefaultValue is not DBNull ? p.DefaultValue : ia?.Default;
        double? Opt(double? v) => v is null || double.IsNaN(v.Value) ? null : v;

        if (t.IsEnum)
            return new WidgetSpec(WidgetKind.Combo)
            {
                Options = Enum.GetNames(t),
                Default = def is null ? null : def.ToString(),
            };

        if (t == typeof(string))
        {
            if (ia?.OptionsProvider is { } providerName)
            {
                var m = nodeType.GetMethod(providerName, BindingFlags.Public | BindingFlags.Static, Type.EmptyTypes)
                    ?? throw new InvalidOperationException($"Options provider '{providerName}' not found as a public static method on {nodeType.Name}.");
                return new WidgetSpec(WidgetKind.Combo)
                {
                    OptionsProvider = () => ((IEnumerable<string>)m.Invoke(null, null)!).ToList(),
                    Default = def,
                };
            }
            if (ia?.Options is { Length: > 0 } options)
                return new WidgetSpec(WidgetKind.Combo) { Options = options, Default = def ?? options[0] };
            return new WidgetSpec(WidgetKind.String) { Multiline = ia?.Multiline ?? false, Default = def ?? "" };
        }

        if (t == typeof(bool)) return new WidgetSpec(WidgetKind.Boolean) { Default = def ?? false };

        if (t == typeof(int) || t == typeof(long) || t == typeof(short) || t == typeof(byte) || t == typeof(uint) || t == typeof(ushort))
            return new WidgetSpec(WidgetKind.Int)
            {
                Default = def is null ? 0L : Convert.ToInt64(def),
                Min = Opt(ia?.Min), Max = Opt(ia?.Max), Step = Opt(ia?.Step) ?? 1,
            };

        if (t == typeof(float) || t == typeof(double))
            return new WidgetSpec(WidgetKind.Float)
            {
                // A float default (0.6f) is read as the decimal it was written as (0.6), not widened to 0.6000000238.
                Default = def switch { null => 0.0, float f => double.Parse(f.ToString("R", System.Globalization.CultureInfo.InvariantCulture), System.Globalization.CultureInfo.InvariantCulture), _ => Convert.ToDouble(def) },
                Min = Opt(ia?.Min), Max = Opt(ia?.Max), Step = Opt(ia?.Step) ?? 0.01,
            };

        return null; // socket-only (IMAGE, MODEL, ...)
    }

    internal static string ToSnakeCase(string s)
    {
        var sb = new System.Text.StringBuilder(s.Length + 4);
        for (var i = 0; i < s.Length; i++)
        {
            var c = s[i];
            if (char.IsUpper(c))
            {
                var prevLowerOrDigit = i > 0 && (char.IsLower(s[i - 1]) || char.IsDigit(s[i - 1]));
                var acronymEnd = i > 0 && char.IsUpper(s[i - 1]) && i + 1 < s.Length && char.IsLower(s[i + 1]);
                if (prevLowerOrDigit || acronymEnd) sb.Append('_');
                sb.Append(char.ToLowerInvariant(c));
            }
            else sb.Append(c);
        }
        return sb.ToString();
    }

    static string SplitWords(string s)
    {
        var sb = new System.Text.StringBuilder(s.Length + 4);
        for (var i = 0; i < s.Length; i++)
        {
            if (i > 0 && char.IsUpper(s[i]) && (char.IsLower(s[i - 1]) || (i + 1 < s.Length && char.IsLower(s[i + 1]))))
                sb.Append(' ');
            sb.Append(s[i]);
        }
        return sb.ToString();
    }
}

/// <summary>Maps CLR types to socket types.</summary>
public static class SocketTypes
{
    public static SocketType FromClr(Type type)
    {
        type = Nullable.GetUnderlyingType(type) ?? type;
        if (type == typeof(int) || type == typeof(long) || type == typeof(short) || type == typeof(byte) || type == typeof(uint) || type == typeof(ushort))
            return SocketType.Int;
        if (type == typeof(float) || type == typeof(double)) return SocketType.Float;
        if (type == typeof(string) || type.IsEnum) return SocketType.String;
        if (type == typeof(bool)) return SocketType.Boolean;
        if (type == typeof(object)) return SocketType.Any;
        var attr = type.GetCustomAttribute<SocketAttribute>();
        return new SocketType(attr?.Name ?? NodeRegistry.ToSnakeCase(type.Name).ToUpperInvariant());
    }
}

