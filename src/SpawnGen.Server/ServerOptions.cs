namespace SpawnGen.Server;

/// <summary>Where the server keeps models, inputs and results. Defaults live under <c>%LOCALAPPDATA%\SpawnGen</c>; each can be set in configuration (<c>SpawnGen:ModelsPath</c> ...) or the environment (<c>SpawnGen__ModelsPath</c>).</summary>
public sealed class ServerOptions
{
    static string Home => Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "SpawnGen");

    /// <summary>A folder with one sub-folder per checkpoint name (<c>sd_turbo/unet/model.onnx</c>).</summary>
    public string ModelsPath { get; set; } = Path.Combine(Home, "models");
    public string OutputPath { get; set; } = Path.Combine(Home, "output");
    public string InputPath { get; set; } = Path.Combine(Home, "input");
    public string TempPath { get; set; } = Path.Combine(Home, "temp");
    /// <summary>Folder of custom node assemblies (.dll with classes marked [Node]), loaded when the server starts. See docs/CUSTOM_NODES.md.</summary>
    public string CustomNodesPath { get; set; } = Path.Combine(Home, "custom_nodes");

    /// <summary>What the custom node loader found at start-up (set by the server, reported by /system_stats).</summary>
    public IReadOnlyList<CustomNodes.Loaded> LoadedCustomNodes { get; set; } = [];

    public static ServerOptions From(IConfiguration configuration)
    {
        var o = new ServerOptions();
        configuration.GetSection("SpawnGen").Bind(o);
        return o;
    }
}
