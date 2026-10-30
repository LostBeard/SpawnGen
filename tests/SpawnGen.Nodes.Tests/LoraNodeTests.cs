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
        Assert.That(model.Weights, Is.EqualTo(new[] { new LoraUse(CheckpointCatalog.LocalLorasRepo, "a.safetensors", 0.8f) }));
        Assert.That(clip.Weights, Is.EqualTo(new[] { new LoraUse(CheckpointCatalog.LocalLorasRepo, "a.safetensors", -0.5f) }));
        Assert.That(clip.Layer, Is.EqualTo(-2), "CLIP Set Last Layer's choice is kept");
        Assert.That(model.Sampling, Is.SameAs(Model.Sampling));
        Assert.That(Model.Weights, Is.Empty);
        Assert.That(Clip.Weights, Is.Empty);

        var (twice, _) = new LoraLoader().Execute(model, clip, "b.safetensors", 1, 1);
        Assert.That(twice.Weights.Cast<LoraUse>().Select(l => l.File), Is.EqualTo(new[] { "a.safetensors", "b.safetensors" }), "chained loaders stack");
        Assert.That(new CLIPSetLastLayer().Execute(clip, -1).Weights, Is.EqualTo(clip.Weights), "CLIP Set Last Layer keeps the LoRAs");
        Assert.That(model.WithPatches(p => p).Weights, Is.EqualTo(model.Weights), "guidance patches keep the LoRAs");
    }

    [Test]
    public void Zero_strengths_pass_the_inputs_through_as_in_ComfyUI()
    {
        var (model, clip) = new LoraLoader().Execute(Model, Clip, "a.safetensors", 0, 0);
        Assert.That(model, Is.SameAs(Model));
        Assert.That(clip, Is.SameAs(Clip));
        var (modelOnly, clipOnly) = new LoraLoader().Execute(Model, Clip, "a.safetensors", 1, 0);
        Assert.That(modelOnly.Weights, Has.Count.EqualTo(1));
        Assert.That(clipOnly, Is.SameAs(Clip), "a zero CLIP strength leaves the CLIP alone");
        Assert.That(new LoraLoaderModelOnly().Execute(Model, "a.safetensors", 0), Is.SameAs(Model));
        Assert.That(new LoraLoaderModelOnly().Execute(Model, "a.safetensors", 0.7).Weights.Cast<LoraUse>().Single().Strength, Is.EqualTo(0.7f));
    }

    [Test]
    public void Weight_steps_with_the_same_key_make_the_same_weights()
    {
        var a = (string f, float s) => new LoraUse("r", f, s);
        Assert.That(WeightStep.KeyOf([a("x", 0.5f), a("y", 1f)]), Is.EqualTo(WeightStep.KeyOf([a("x", 0.5f), a("y", 1f)])));
        Assert.That(WeightStep.KeyOf([a("x", 0.5f), a("y", 1f)]), Is.Not.EqualTo(WeightStep.KeyOf([a("y", 1f), a("x", 0.5f)])), "order counts once merges are in");
        Assert.That(WeightStep.KeyOf([a("x", 0.5f)]), Is.Not.EqualTo(WeightStep.KeyOf([a("x", 0.25f)])));
        Assert.That(WeightStep.KeyOf([]), Is.Empty, "no step: the file's weights");

        object one = new(), two = new();
        var merge = (object other, string name) => new WeightMerge(other, [a("x", 1f)], name, _ => null);
        Assert.That(merge(one, "simple 0.5").Key, Is.EqualTo(merge(one, "simple 0.5").Key), "the same model and settings");
        Assert.That(merge(one, "simple 0.5").Key, Is.Not.EqualTo(merge(two, "simple 0.5").Key), "another model");
        Assert.That(merge(one, "simple 0.5").Key, Is.Not.EqualTo(merge(one, "simple 0.6").Key), "other settings");
        Assert.That(merge(one, "simple 0.5").Key, Is.Not.EqualTo(new WeightMerge(one, [], "simple 0.5", _ => null).Key), "the other model's own steps count");
    }
}
