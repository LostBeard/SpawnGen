using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Engine.Execution;

/// <summary>An output node left out of a run because what it needs does not validate.</summary>
public sealed record OutputFailure(string OutputId, ValidationResult Result);

/// <summary>Which output nodes of a prompt run, and which are left out with their errors.</summary>
public sealed record Admission(IReadOnlyList<string> Outputs, IReadOnlyList<OutputFailure> Failures)
{
    /// <summary>Nothing can run: no output nodes were asked for (or there are none), or none of them validates.</summary>
    public bool Refused => Outputs.Count == 0;
}

/// <summary>
/// Decides what a queued prompt runs, as ComfyUI's /prompt does (measured against its server): every output node is checked on its own; the ones
/// that validate run and the others are left out and reported, so a broken side branch does not stop the rest. <c>partial_execution_targets</c>
/// narrows the outputs to the listed ids that are output nodes of the prompt (others are ignored); with none left the prompt has no outputs.
/// </summary>
public static class PromptAdmission
{
    /// <summary>ComfyUI's error type when nothing is left to run.</summary>
    public const string NoOutputsCode = "prompt_no_outputs";

    public static Admission Check(NodeRegistry registry, PromptGraph graph, IReadOnlyList<string>? targets = null)
    {
        bool IsOutput(string id) => graph.Nodes.TryGetValue(id, out var node) && registry.GetSchema(node.ClassType)?.IsOutputNode == true;
        var candidates = targets is null
            ? graph.Nodes.Keys.Where(IsOutput).ToList()
            : targets.Distinct(StringComparer.Ordinal).Where(IsOutput).ToList();
        var outputs = new List<string>();
        var failures = new List<OutputFailure>();
        foreach (var id in candidates)
        {
            var result = PromptValidator.Validate(registry, graph, [id]);
            if (result.IsValid) outputs.Add(id);
            else failures.Add(new OutputFailure(id, result));
        }
        return new Admission(outputs, failures);
    }

    /// <summary>The exception for a refused prompt: "no outputs" when nothing was there to check, else every left-out output's errors.</summary>
    public static PromptValidationException Refusal(Admission admission)
    {
        if (admission.Failures.Count == 0)
            return new PromptValidationException(new ValidationResult([new ValidationError(null, null, NoOutputsCode, "Prompt has no outputs")], [], []));
        var errors = admission.Failures.SelectMany(f => f.Result.Errors).Distinct().ToList();
        return new PromptValidationException(new ValidationResult(errors, [], []), admission.Failures);
    }
}
