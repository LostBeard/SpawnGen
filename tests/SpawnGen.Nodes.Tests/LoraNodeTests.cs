using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>Load LoRA's handle semantics (copies, never changing what came in) and how the LoRAs of a model and a CLIP combine into one merge.</summary>
[NonParallelizable]
public class LoraNodeTests
{
    static readonly ModelHandle Model = new(null!, new ModelSampling(ModelSigmas.LinearSchedule()));
    static readonly ClipHandle Clip = new(null!) { Layer = -2 };

    [Test]
    public void Load_LoRA_adds_to_copies_and_leaves_the_inputs_alone()
    {
        var (model, clip) = new LoraLoader().Execute(Model, Clip, "a.safetensors", 0.8, -0.5);
        Assert.That(model.Loras, Is.EqualTo(new[] { new LoraUse(CheckpointCatalog.LocalLorasRepo, "a.safetensors", 0.8f) }));
        Assert.That(clip.Loras, Is.EqualTo(new[] { new LoraUse(CheckpointCatalog.LocalLorasRepo, "a.safetensors", -0.5f) }));
        Assert.That(clip.Layer, Is.EqualTo(-2), "CLIP Set Last Layer's choice is kept");
        Assert.That(model.Sampling, Is.SameAs(Model.Sampling));
        Assert.That(Model.Loras, Is.Empty);
        Assert.That(Clip.Loras, Is.Empty);

        var (twice, _) = new LoraLoader().Execute(model, clip, "b.safetensors", 1, 1);
        Assert.That(twice.Loras.Select(l => l.File), Is.EqualTo(new[] { "a.safetensors", "b.safetensors" }), "chained loaders stack");
        Assert.That(new CLIPSetLastLayer().Execute(clip, -1).Loras, Is.EqualTo(clip.Loras), "CLIP Set Last Layer keeps the LoRAs");
        Assert.That(model.WithPatches(p => p).Loras, Is.EqualTo(model.Loras), "guidance patches keep the LoRAs");
    }

    [Test]
    public void Zero_strengths_pass_the_inputs_through_as_in_ComfyUI()
    {
        var (model, clip) = new LoraLoader().Execute(Model, Clip, "a.safetensors", 0, 0);
        Assert.That(model, Is.SameAs(Model));
        Assert.That(clip, Is.SameAs(Clip));
        var (modelOnly, clipOnly) = new LoraLoader().Execute(Model, Clip, "a.safetensors", 1, 0);
        Assert.That(modelOnly.Loras, Has.Count.EqualTo(1));
        Assert.That(clipOnly, Is.SameAs(Clip), "a zero CLIP strength leaves the CLIP alone");
        Assert.That(new LoraLoaderModelOnly().Execute(Model, "a.safetensors", 0), Is.SameAs(Model));
        Assert.That(new LoraLoaderModelOnly().Execute(Model, "a.safetensors", 0.7).Loras.Single().Strength, Is.EqualTo(0.7f));
    }

    [Test]
    public void A_models_and_a_CLIPs_LoRAs_combine_into_one_merge_per_file()
    {
        var a = (string f, float s) => new LoraUse("r", f, s);
        var combined = Checkpoint.CombineLoras([a("x", 0.5f), a("y", 1f), a("x", 0.25f)], [a("x", 2f), a("z", -1f)]);
        Assert.That(combined, Is.EqualTo(new[]
        {
            ("r", "x", 0.75f, 2f),   // the same file twice on the model sums (the merges are linear)
            ("r", "y", 1f, 0f),
            ("r", "z", 0f, -1f),     // CLIP only
        }));
        Assert.That(Checkpoint.CombineLoras([], []), Is.Empty, "no LoRA: the merge removes every LoRA");
    }
}
