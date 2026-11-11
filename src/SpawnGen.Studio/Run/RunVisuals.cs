namespace SpawnGen.Studio.Run;

/// <summary>What the canvas shows about the run in progress: the node that is executing, how far along it is, and the nodes that failed.</summary>
public sealed class RunVisuals
{
    public int? ExecutingNode { get; set; }
    public double ProgressValue { get; set; }
    public double ProgressMax { get; set; }
    /// <summary>Node id to the message shown on it (a failed node, or a node the server refused to run).</summary>
    public Dictionary<int, string> Errors { get; } = [];

    /// <summary>App Builder, input step: widgets are outlined as pickable; the picked ones (in app order) carry their number.</summary>
    public bool ShowAppInputs { get; set; }
    public List<(int Node, string Widget)> AppInputs { get; } = [];

    /// <summary>App Builder, output step: output nodes are outlined as pickable; the picked ones (in app order) carry their number.</summary>
    public bool ShowAppOutputs { get; set; }
    public List<int> AppOutputs { get; } = [];
    public HashSet<int> AppOutputCandidates { get; } = [];

    public double Fraction => ProgressMax > 0 ? Math.Clamp(ProgressValue / ProgressMax, 0, 1) : 0;

    public void BeginRun()
    {
        ExecutingNode = null;
        ProgressValue = ProgressMax = 0;
        Errors.Clear();
    }

    public void EndRun()
    {
        ExecutingNode = null;
        ProgressValue = ProgressMax = 0;
    }
}
