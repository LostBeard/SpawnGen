using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// InstructPixToPixConditioning against ComfyUI's behaviour (measured 2026-11-14 with ae.safetensors: _scratch/probe/ip2p): the concat latent is VAE
/// Encode's result exactly (its centred crop included), the same in both conditionings; the rest of each entry is kept; the latent is zeros the shape
/// of that encode, one per picture. The VAE here is a stand-in that records what it was given (VAE Encode itself is checked in AeVaeTests).
/// </summary>
public class InstructPixToPixTests
{
    /// <summary>A 16-channel, 8x "VAE" whose latent cell (c, y, x) is c + the picture's red at the cell's top-left pixel.</summary>
    sealed class StandInVae : IVaeModel
    {
        public List<(int H, int W, float First)> Encoded { get; } = [];
        public string Name => "stand-in";
        public SpawnDev.ILGPU.ML.StableDiffusion.SdVae? SdVae => null;
        public Task<float[]> DecodeAsync(float[] latent, int h, int w, CancellationToken cancellationToken = default) => throw new NotSupportedException();

        public Task<float[]> EncodeImageAsync(float[] image, int H, int W, CancellationToken cancellationToken = default)
        {
            // image is [3, H, W] in [-1, 1] or [H, W, 3] in [0, 1], whichever the helper hands over; only its first value is recorded.
            Encoded.Add((H, W, image[0]));
            int h = H / 8, w = W / 8;
            var latent = new float[16 * h * w];
            for (int c = 0; c < 16; c++)
                for (int i = 0; i < h * w; i++) latent[c * h * w + i] = c + 0.5f;
            return Task.FromResult(latent);
        }
    }

    sealed class Services(params object[] instances) : IServiceProvider
    {
        public object? GetService(Type serviceType) => instances.FirstOrDefault(serviceType.IsInstanceOfType);
    }

    static NodeContext Context(IServiceProvider services) => (NodeContext)Activator.CreateInstance(typeof(NodeContext),
        System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance, null,
        ["1", "InstructPixToPixConditioning", services, CancellationToken.None, null, new Lazy<string>(() => "{}"), new Dictionary<string, string>(), null, null, 0], null)!;

    [Test]
    public async Task Both_conditionings_carry_the_encoded_pictures_and_the_latent_is_zeros()
    {
        var vae = new StandInVae();
        var services = new Services();
        var pixels = new ImageBatch(2, 125, 205, 3);
        for (int i = 0; i < pixels.Data.Length; i++) pixels.Data[i] = (i % 251) / 250f;
        var positive = new Conditioning([new CondEntry([1, 2, 3, 4], [1, 2, 2]) { Pooled = [9f], Guidance = 3.5 }]);
        var negative = new Conditioning([new CondEntry([0, 0, 0, 0], [1, 2, 2]) { Strength = 0.5f }]);

        var (pos, neg, latent) = await new InstructPixToPixConditioning().ExecuteAsync(positive, negative, new VaeHandle(vae), pixels, Context(services), services);

        Assert.That(vae.Encoded.Select(e => (e.H, e.W)), Has.All.EqualTo((120, 200)), "VAE Encode's centred crop of 205 x 125");
        Assert.That(latent.Shape, Is.EqualTo(new[] { 2, 16, 15, 25 }));
        Assert.That(latent.Data, Has.All.EqualTo(0f));
        var concat = pos.Entries[0].ConcatLatent!;
        Assert.That(concat.Shape, Is.EqualTo(new[] { 2, 16, 15, 25 }));
        Assert.That(concat.Data[0], Is.EqualTo(0.5f));
        Assert.That(neg.Entries[0].ConcatLatent, Is.SameAs(concat), "one encode, in both");
        Assert.That(pos.Entries[0].Pooled, Is.EqualTo(new[] { 9f }));
        Assert.That(pos.Entries[0].Guidance, Is.EqualTo(3.5));
        Assert.That(neg.Entries[0].Strength, Is.EqualTo(0.5f));
        Assert.That(neg.Entries[0].ConcatMask, Is.Null);
    }
}
