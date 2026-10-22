using SpawnDev.ILGPU.ML.Pipelines;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Diagnostic: runs the library's one-call ImageGenerationPipeline with the same prompt and seed as the node graph, so a wrong
/// node-graph image can be told apart from a backend/library problem. Needs SPAWNGEN_MODELS; writes PNGs to SPAWNGEN_DIAG_OUT.
/// </summary>
public class ReferencePipelineDiagnostic
{
    [Test, Explicit("Diagnostic; needs SPAWNGEN_MODELS and a GPU.")]
    public async Task Library_pipeline_reference_image()
    {
        var models = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS");
        if (string.IsNullOrEmpty(models)) Assert.Ignore("SPAWNGEN_MODELS is not set.");
        var outDir = Environment.GetEnvironmentVariable("SPAWNGEN_DIAG_OUT") ?? Path.GetTempPath();

        using var runtime = new MlRuntime(new FolderModelSource(models!));
        using var pipe = await ImageGenerationPipeline.CreateAsync(runtime.Accelerator, runtime.ModelSource);
        pipe.NumInferenceSteps = 1;
        pipe.GuidanceScale = 0f;
        var result = await pipe.RunAsync(new ImageGenerationInput { Prompt = "a photo of a red apple on a wooden table", Seed = 7 });

        var rgb = new float[result.Width * result.Height * 3];
        for (int i = 0; i < result.Width * result.Height; i++)
            for (int c = 0; c < 3; c++) rgb[i * 3 + c] = result.ImageRGBA[i * 4 + c] / 255f;
        var path = Path.Combine(outDir, "reference_pipeline.png");
        await File.WriteAllBytesAsync(path, PngWriter.Encode(new ImageBatch(1, result.Height, result.Width, 3, rgb)));
        TestContext.Out.WriteLine($"{path} ({result.InferenceTimeMs:F0} ms, {runtime.Accelerator.AcceleratorType})");
    }
}
