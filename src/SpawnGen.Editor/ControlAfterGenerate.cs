using SpawnGen.Engine.Nodes;

namespace SpawnGen.Editor;

/// <summary>
/// ComfyUI's "control after generate": a combo next to a seed widget that says what the seed does after each queue
/// (keep it, count up, count down, pick a new random one). The chosen mode lives in the node's widget values under <see cref="Name"/>.
/// </summary>
public static class ControlAfterGenerate
{
    public const string Name = "control_after_generate";
    public static readonly string[] Modes = ["fixed", "increment", "decrement", "randomize"];

    /// <summary>The widget a seed gets: a combo of <see cref="Modes"/>, "randomize" by default like ComfyUI.</summary>
    public static readonly WidgetSpec Spec = new(WidgetKind.Combo) { Default = "randomize", Options = Modes };

    /// <summary>True for the seed widgets that carry a control (ComfyUI adds it to <c>seed</c> and <c>noise_seed</c>).</summary>
    public static bool HasControl(InputSpec input) => input.Widget?.Kind == WidgetKind.Int && input.Name is "seed" or "noise_seed";

    public static string ModeOf(EditorNode node) =>
        node.Widgets.GetValueOrDefault(Name) is string s && Modes.Contains(s) ? s : "randomize";

    /// <summary>The seed after one queue under <paramref name="mode"/>. Seeds stay in [0, 2^53) like ComfyUI's (the largest exact JSON integer).</summary>
    public static long Next(long seed, string mode, Random random)
    {
        const long Max = 1L << 53;
        return mode switch
        {
            "increment" => seed + 1 >= Max ? 0 : seed + 1,
            "decrement" => seed <= 0 ? Max - 1 : seed - 1,
            "randomize" => random.NextInt64(Max),
            _ => seed,
        };
    }

    /// <summary>
    /// Applies every seed node's control once, as ComfyUI does right after queueing: the run used the seeds as they were, the graph now holds the next ones.
    /// One undo step. Seeds fed by a link are left alone (their source decides).
    /// </summary>
    public static void ApplyAfterQueue(EditorDocument doc, Func<string, NodeSchema?> schema, Random random)
    {
        doc.Group(() =>
        {
            foreach (var node in doc.Graph.Nodes.ToList())
            {
                if (node.Mode != NodeMode.Always) continue;
                foreach (var input in schema(node.ClassType)?.Inputs.Where(HasControl) ?? [])
                {
                    if (doc.Graph.LinkInto(node.Id, input.Name) is not null) continue;
                    var mode = ModeOf(node);
                    if (mode == "fixed") continue;
                    long current = node.Widgets.GetValueOrDefault(input.Name) switch { long l => l, int i => i, double d => (long)d, _ => 0L };
                    doc.SetWidget(node.Id, input.Name, Next(current, mode, random));
                }
            }
        });
        doc.EndGesture();
    }
}
