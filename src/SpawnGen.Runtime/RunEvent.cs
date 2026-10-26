using System.Text.Json.Nodes;

namespace SpawnGen.Runtime;

/// <summary>An event in ComfyUI's websocket shape: <c>{"type": "...", "data": {...}}</c>. Every host reports a run with the same events.</summary>
public sealed record RunEvent(string Type, JsonObject Data)
{
    /// <summary>A preview's PNG bytes: a websocket sends them as ComfyUI's binary preview frame instead of the JSON event (whose data URL in-process listeners read).</summary>
    public byte[]? Image { get; init; }

    public string ToJson() => new JsonObject { ["type"] = Type, ["data"] = Data.DeepClone() }.ToJsonString();
}

/// <summary>Where a <see cref="PromptQueue"/> reports what it does: a websocket hub on the server, an in-process listener in the browser.</summary>
public interface IRunEvents
{
    void Publish(RunEvent e);
}

/// <summary>An <see cref="IRunEvents"/> that hands every event to subscribers in the same process.</summary>
public sealed class RunEventBus : IRunEvents
{
    public event Action<RunEvent>? Published;

    public void Publish(RunEvent e) => Published?.Invoke(e);
}
