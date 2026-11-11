using ILGPU.Runtime;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

internal sealed record SamplingSettings(long Seed, string Sampler, bool AddNoise = true, SamplerOptions? Options = null);

internal static class SamplingCore
{
    public static async Task<LatentBatch> RunAsync(Accelerator acc, GuiderChoice guider, LatentBatch latent, float[] sigmas, SamplingSettings settings, NodeContext context) =>
        (await RunWithDenoisedAsync(acc, guider, latent, sigmas, settings, context)).Output;

    /// <summary>ComfyUI's starting noise: torch.randn of the latent's shape from a generator seeded with the seed (CPU); with batch indexes, one draw per index in turn.</summary>
    internal static float[] InitialNoise(long seed, int index, LatentBatch latent)
    {
        int per = latent.Channels * latent.Height * latent.Width;
        if (latent.BatchIndex is null) return TorchRandom.Randn(seed, per * latent.Batch).AsSpan(index * per, per).ToArray();
        int want = NoiseIndex(latent, index);
        var g = new TorchRandom(seed);
        float[] draw = [];
        for (int i = 0; i <= want; i++) draw = g.Randn(per);
        return draw;
    }

    internal static int NoiseIndex(LatentBatch latent, int index) => latent.BatchIndex is { } bi && index < bi.Length ? bi[index] : index;

    public static Task<(LatentBatch Output, LatentBatch Denoised)> RunWithDenoisedAsync(Accelerator acc, GuiderChoice guider, LatentBatch latent,
        float[] sigmas, SamplingSettings settings, NodeContext context) => throw new NotImplementedException();

    internal static Task<float[]> InpaintedCall(float[] sample, double sigma, float[]? mask, Func<double, float[], double[], ModelSampling, float[]>? denoiseMask,
        double[] runSigmas, ModelSampling sampling, float[] original, float[] inpaintNoise, int plane, Func<float[], Task<float[]>> predict) => throw new NotImplementedException();

    internal static float[] Weight(CondEntry e, int h, int w) => throw new NotImplementedException();
    internal static bool IsActive(CondEntry e, double sigma, ModelSampling sampling) => throw new NotImplementedException();
}
