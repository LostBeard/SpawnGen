namespace SpawnGen.Engine.Execution;

/// <summary>
/// The prompt being run on this async flow, as a number that changes with every prompt (set by <see cref="PromptExecutor.ExecuteAsync"/>; 0
/// outside a run). A model cache uses it to tell the models the current prompt has used from those an earlier prompt left on the GPU, and to
/// free the latter before it loads something new (as ComfyUI unloads the models a new prompt does not need).
/// </summary>
public static class PromptScope
{
    static long _last;
    static readonly AsyncLocal<long> _current = new();

    public static long Current => _current.Value;

    /// <summary>Starts a prompt on this async flow (and the calls it awaits).</summary>
    internal static void Begin() => _current.Value = Interlocked.Increment(ref _last);
}
