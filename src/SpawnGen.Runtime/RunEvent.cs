using System.Text.Json.Nodes;

namespace SpawnGen.Runtime;

/// <summary>An event in ComfyUI's websocket shape: <c>{"type": "...", "data": {...}}</c>. Every host reports a run with the same events.</summary>
public sealed record RunEvent(string Type, JsonObject Data)
{
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
