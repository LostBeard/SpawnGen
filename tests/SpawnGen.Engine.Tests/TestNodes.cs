using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Engine.Tests;

/// <summary>Counts how many times each test node really ran, so tests can assert on caching.</summary>
public static class Calls
{
    static readonly Dictionary<string, int> _counts = new();
    public static void Hit(string name) { lock (_counts) _counts[name] = _counts.GetValueOrDefault(name) + 1; }
    public static int Of(string name) { lock (_counts) return _counts.GetValueOrDefault(name); }
    public static void Reset() { lock (_counts) _counts.Clear(); }
}

public enum Mode { Fast, Slow }

[Node(Category = "test")]
public sealed class IntSource
{
    public long Execute(long value = 1) { Calls.Hit(nameof(IntSource)); return value; }
}

[Node(Category = "test")]
public sealed class AddNumbers
{
    public long Execute(long a, long b = 10) { Calls.Hit(nameof(AddNumbers)); return a + b; }
}

[Node(Category = "test")]
public sealed class Widgets
{
    public string Execute(
        [Input(Min = 0, Max = 100, Step = 5)] int steps = 20,
        float cfgScale = 7.5f,
        [Input(Multiline = true)] string text = "hi",
        bool flag = true,
        Mode mode = Mode.Fast,
        [Input(Options = ["a", "b"])] string pick = "a") => $"{steps}/{cfgScale}/{text}/{flag}/{mode}/{pick}";
}

[Node(Category = "test")]
public sealed class Split
{
    public (long Low, long High, string Label) Execute(long value = 5) { Calls.Hit(nameof(Split)); return (value - 1, value + 1, "x"); }
}

[Node(Category = "test")]
public sealed class AsyncDouble
{
    public async Task<long> ExecuteAsync(long value, CancellationToken ct)
    {
        Calls.Hit(nameof(AsyncDouble));
        await Task.Delay(1, ct);
        return value * 2;
    }
}

[Node(Category = "test", IsOutput = true)]
public sealed class Sink
{
    public UiOutput Execute(long value)
    {
        Calls.Hit(nameof(Sink));
        return new UiOutput("values", new[] { value });
    }
}

[Node(Category = "test", IsOutput = true)]
public sealed class StringSink
{
    public UiOutput Execute(string text) { Calls.Hit(nameof(StringSink)); return new UiOutput("text", text); }
}

[Node(Category = "test", AlwaysExecute = true)]
public sealed class Counter
{
    static long _n;
    public long Execute() { Calls.Hit(nameof(Counter)); return Interlocked.Increment(ref _n); }
}

[Node(Category = "test")]
public sealed class Boom
{
    public long Execute(long value) => throw new InvalidOperationException("kaput");
}

[Node(Category = "test")]
public sealed class Slow
{
    public async Task<long> ExecuteAsync(long value, NodeContext ctx)
    {
        ctx.Progress(0, 2, "starting");
        await Task.Delay(Timeout.Infinite, ctx.CancellationToken);
        return value;
    }
}

[Node(Category = "test")]
public sealed class MakeImage
{
    public ImageBatch Execute(int size = 4) => new(1, size, size);
}

[Node(Category = "test")]
public sealed class ImageToLatent
{
    public LatentBatch Execute(ImageBatch image) => new(image.Batch, 4, image.Height / 8, image.Width / 8);
}

[Node(Category = "test", IsOutput = true)]
public sealed class LatentSink
{
    public UiOutput Execute(LatentBatch latent) => new("shape", latent.Shape);
}

[Node(Category = "test")]
public sealed class ModelPicker
{
    public static List<string> Models() => ["sd-turbo", "flux2"];
    public string Execute([Input(OptionsProvider = nameof(Models))] string model = "sd-turbo") => model;
}
