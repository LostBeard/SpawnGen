using SpawnGen.Nodes.ML;

namespace SpawnGen.Server.Tests;

/// <summary>
/// One GPU runtime for every test server in this process, as one server process has one in production: Load Image decodes and Save Image
/// encodes on the GPU (Codecs), and opening the accelerator and compiling those kernels per test server took each test from 0.7 s to 14 s
/// (the suite 9 s -> 152 s). Registered as an instance, so a test server's shutdown does not dispose it; kernels are cached per accelerator.
/// </summary>
static class SharedGpu
{
    static readonly Lazy<MlRuntime> Shared = new(() => new MlRuntime(new FolderModelSource(Path.Combine(Path.GetTempPath(), "spawngen-server-tests-models"))));

    public static MlRuntime Runtime => Shared.Value;
}
