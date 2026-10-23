using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Engine.Execution;

/// <summary>
/// Checks the part of a graph that feeds the output nodes and produces the run plan. Problems elsewhere in the
/// graph are ignored, as in ComfyUI, so a half-built side branch never blocks a run.
/// </summary>
public static class PromptValidator
{
    enum Visit { Visiting, Done }

    public static ValidationResult Validate(NodeRegistry registry, PromptGraph graph, IReadOnlyList<string>? outputNodeIds = null)
    {
        var errors = new List<ValidationError>();
        var plan = new List<string>();

        var outputs = new List<string>();
        if (outputNodeIds is not null)
        {
            foreach (var id in outputNodeIds)
            {
                if (graph.Nodes.ContainsKey(id)) outputs.Add(id);
                else errors.Add(new(id, null, "missing_node", $"Requested output node '{id}' does not exist."));
            }
        }
        else
        {
            foreach (var (id, node) in graph.Nodes)
                if (registry.GetSchema(node.ClassType)?.IsOutputNode == true) outputs.Add(id);
        }

        if (outputs.Count == 0 && errors.Count == 0)
            errors.Add(new(null, null, "no_outputs", "The prompt has no output nodes (such as Preview Image or Save Image)."));

        var state = new Dictionary<string, Visit>(StringComparer.Ordinal);
        foreach (var id in outputs) Walk(registry, graph, id, state, plan, errors);

        return new ValidationResult(errors, outputs, errors.Count == 0 ? plan : []);
    }

    static void Walk(NodeRegistry reg, PromptGraph graph, string id, Dictionary<string, Visit> state,
        List<string> plan, List<ValidationError> errors)
    {
        if (state.TryGetValue(id, out var s))
        {
            if (s == Visit.Visiting)
                errors.Add(new(id, null, "cycle", $"Node {id} is part of a cycle."));
            return;
        }
        state[id] = Visit.Visiting;

        var node = graph.Nodes[id];
        if (!reg.TryGet(node.ClassType, out var registered))
        {
            errors.Add(new(id, null, "unknown_node_type", $"Node {id}: unknown node type '{node.ClassType}'."));
            state[id] = Visit.Done;
            return;
        }

        foreach (var spec in registered.Schema.Inputs)
        {
            if (!node.Inputs.TryGetValue(spec.Name, out var input) || (!input.IsLink && input.Literal is null))
            {
                if (spec.Required)
                    errors.Add(new(id, spec.Name, "required_input_missing", $"Node {id} ({node.ClassType}): required input '{spec.Name}' is missing."));
                continue;
            }

            if (input.IsLink)
            {
                if (!graph.Nodes.TryGetValue(input.NodeId!, out var upstream))
                {
                    errors.Add(new(id, spec.Name, "missing_node", $"Node {id} input '{spec.Name}' links to missing node '{input.NodeId}'."));
                    continue;
                }
                if (!reg.TryGet(upstream.ClassType, out var upReg))
                {
                    errors.Add(new(input.NodeId, null, "unknown_node_type", $"Node {input.NodeId}: unknown node type '{upstream.ClassType}'."));
                    continue;
                }
                if (input.OutputIndex < 0 || input.OutputIndex >= upReg.Schema.Outputs.Count)
                {
                    errors.Add(new(id, spec.Name, "bad_link",
                        $"Node {id} input '{spec.Name}' links to output {input.OutputIndex} of node {input.NodeId}, which has {upReg.Schema.Outputs.Count}."));
                    continue;
                }
                var outType = upReg.Schema.Outputs[input.OutputIndex].Type;
                if (!spec.Type.Accepts(outType))
                {
                    errors.Add(new(id, spec.Name, "type_mismatch",
                        $"Node {id} input '{spec.Name}' expects {spec.Type} but node {input.NodeId} outputs {outType}."));
                    continue;
                }
                Walk(reg, graph, input.NodeId!, state, plan, errors);
            }
            else
            {
                var problem = CheckLiteral(registered, spec, input.Literal!);
                if (problem is not null)
                    errors.Add(new(id, spec.Name, "bad_value", $"Node {id} ({node.ClassType}) input '{spec.Name}': {problem}"));
            }
        }

        state[id] = Visit.Done;
        plan.Add(id);
    }

    static Type MemberClr(SocketType type) => type.Name switch { "INT" => typeof(long), "FLOAT" => typeof(double), "BOOLEAN" => typeof(bool), _ => typeof(string) };

    static string? CheckLiteral(RegisteredNode node, InputSpec spec, object literal)
    {
        var w = spec.Widget;
        if (w is null && spec.Group is not null)
        {
            // A member of an Autogrow group is a socket in the editor, but a prompt may give it a value (ComfyUI's templates can be widgets):
            // any value for an any-type member, one that fits a part of a union, or one of the member's own type.
            if (spec.Type.IsAny) return null;
            var parts = spec.Type.Name.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
            if (parts.All(p => p is "INT" or "FLOAT" or "BOOLEAN" or "STRING"))
                return parts.Any(p => ValueCoercion.TryCoerce(literal, MemberClr(new SocketType(p)), out _)) ? null : $"'{literal}' is not a valid {spec.Type}.";
        }
        if (w is null) return $"expects a {spec.Type} link, not a value.";

        switch (w.Kind)
        {
            case WidgetKind.Int:
            case WidgetKind.Float:
                var clr = node.InputClrType(spec.Name) ?? (w.Kind == WidgetKind.Int ? typeof(long) : typeof(double));
                if (!ValueCoercion.TryCoerce(literal, clr, out _))
                    return w.Kind == WidgetKind.Int ? $"'{literal}' is not a valid whole number." : $"'{literal}' is not a valid number.";
                var d = ValueCoercion.AsDouble(literal)!.Value;
                if (w.Min is { } min && d < min) return $"{d} is below the minimum {min}.";
                if (w.Max is { } max && d > max) return $"{d} is above the maximum {max}.";
                return null;
            case WidgetKind.Boolean:
                return literal is bool ? null : "expects true or false.";
            case WidgetKind.Color:
                return literal is string color && System.Text.RegularExpressions.Regex.IsMatch(color, "^#[0-9a-fA-F]{6}([0-9a-fA-F]{2})?$") ? null : $"'{literal}' is not a colour (#rrggbb).";
            case WidgetKind.String:
                return literal is string ? null : "expects text.";
            case WidgetKind.Combo:
                if (literal is not string str) return "expects one of the listed options.";
                var options = w.ResolveOptions();
                return options.Contains(str, StringComparer.OrdinalIgnoreCase)
                    ? null
                    : $"'{str}' is not one of: {string.Join(", ", options.Take(12))}{(options.Count > 12 ? ", ..." : "")}.";
            default:
                return null;
        }
    }
}
