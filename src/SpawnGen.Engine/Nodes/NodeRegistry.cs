using System.Reflection;
using System.Runtime.CompilerServices;
using Microsoft.Extensions.DependencyInjection;

namespace SpawnGen.Engine.Nodes;

/// <summary>The values a node produced: socket outputs in schema order plus optional editor preview data.</summary>
public sealed record NodeOutputs(object?[] Values, UiOutput? Ui);

/// <summary>A node type: its schema plus how to invoke it.</summary>
public sealed class RegisteredNode
{
    internal enum ParamKind { Input, Context, CancellationToken, Services }

    internal sealed record ParamBinding(ParameterInfo Info, ParamKind Kind, string? InputName);

    internal Type Type { get; init; } = default!;
    internal MethodInfo Method { get; init; } = default!;
    internal ParamBinding[] Params { get; init; } = default!;
    internal Type[] OutputTypes { get; init; } = default!;
    internal bool ReturnsTuple { get; init; } = default!;
    internal int UiIndex { get; init; } = default!;
    internal Type ResultType { get; init; } = default!;

    public required NodeSchema Schema { get; init; }

    /// <summary>A node known only by its schema (it lives on a server): it can be edited and validated here but not run here.</summary>
    public bool IsRemote { get; init; }

    /// <summary>CLR type an input's value must be coerced to.</summary>
    internal Type? InputClrType(string inputName) =>
        Params?.FirstOrDefault(p => p.Kind == ParamKind.Input && p.InputName == inputName)?.Info.ParameterType; // a remote node has no parameters

    internal async ValueTask<NodeOutputs> InvokeAsync(IReadOnlyDictionary<string, object?> inputs, NodeContext context)
    {
        if (IsRemote) throw new NotSupportedException($"'{Schema.DisplayName}' is a node of the SpawnGen server (a custom node): run this graph on the server engine.");
        var args = new object?[Params.Length];
        for (var i = 0; i < Params.Length; i++)
        {
            var p = Params[i];
            switch (p.Kind)
            {
                case ParamKind.Context: args[i] = context; break;
                case ParamKind.CancellationToken: args[i] = context.CancellationToken; break;
                case ParamKind.Services: args[i] = context.Services; break;
                default:
                    if (inputs.TryGetValue(p.InputName!, out var raw) && raw is not null)
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

        var instance = Method.IsStatic ? null : ActivatorUtilities.CreateInstance(context.Services, Type);
        object? ret;
        try
        {
            ret = Method.Invoke(instance, args);
        }
        catch (TargetInvocationException tie) when (tie.InnerException is not null)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(tie.InnerException).Throw();
            throw;
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

    public bool TryGet(string id, out RegisteredNode node) => _nodes.TryGetValue(id, out node!);

    public NodeSchema? GetSchema(string id) => _nodes.TryGetValue(id, out var n) ? n.Schema : null;

    public NodeRegistry Add<T>() => Add(typeof(T));

    public NodeRegistry Add(Type type)
    {
        var node = Build(type);
        _nodes[node.Schema.Id] = node;
        return this;
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

        var inputs = new List<InputSpec>();
        var bindings = new List<RegisteredNode.ParamBinding>();
        foreach (var p in method.GetParameters())
        {
            if (p.ParameterType == typeof(NodeContext)) { bindings.Add(new(p, RegisteredNode.ParamKind.Context, null)); continue; }
            if (p.ParameterType == typeof(CancellationToken)) { bindings.Add(new(p, RegisteredNode.ParamKind.CancellationToken, null)); continue; }
            if (p.ParameterType == typeof(IServiceProvider)) { bindings.Add(new(p, RegisteredNode.ParamKind.Services, null)); continue; }

            var ia = p.GetCustomAttribute<InputAttribute>();
            var name = ia?.Name ?? ToSnakeCase(p.Name!);
            inputs.Add(new InputSpec(name, SocketTypes.FromClr(p.ParameterType), Required: !p.HasDefaultValue,
                Widget: BuildWidget(type, p, ia), Tooltip: ia?.Tooltip));
            bindings.Add(new(p, RegisteredNode.ParamKind.Input, name));
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
            var socket = SocketTypes.FromClr(elementTypes[i]);
            var given = tupleNames is not null && i < tupleNames.Count ? tupleNames[i] : null;
            outputs.Add(new OutputSpec(given is null ? socket.Name : ToSnakeCase(given), socket));
        }

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
        };

        return new RegisteredNode
        {
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

    static WidgetSpec? BuildWidget(Type nodeType, ParameterInfo p, InputAttribute? ia)
    {
        var t = Nullable.GetUnderlyingType(p.ParameterType) ?? p.ParameterType;
        var def = p.HasDefaultValue && p.DefaultValue is not DBNull ? p.DefaultValue : null;
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
                Default = def is null ? 0.0 : Convert.ToDouble(def),
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

