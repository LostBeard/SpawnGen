using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.StableDiffusion;

namespace SpawnGen.Nodes.ML;

/// <summary>
/// A single-file Stable Diffusion checkpoint (<c>.safetensors</c> with the UNet, CLIP and VAE, as ComfyUI's models/checkpoints holds them: SD 1.x, SD 2.x, SDXL
/// and their fine-tunes), loaded by the library's <see cref="SdCheckpoint"/>. Weights live on the GPU as fp16 whatever the file stores.
/// </summary>
internal sealed class SafetensorsCheckpointModels : ICheckpointModels
{
    readonly SdCheckpoint _ck;
    readonly Accelerator _acc;
    readonly SpawnDev.ILGPU.ML.Hub.IModelSource _source;

    SafetensorsCheckpointModels(SdCheckpoint ck, Accelerator acc, SpawnDev.ILGPU.ML.Hub.IModelSource source)
    {
        _ck = ck;
        _acc = acc;
        _source = source;
        if (ck.UNet is null || ck.TextEncoder is null || ck.Vae is null)
            throw new InvalidDataException("The checkpoint file lacks a UNet, text encoder or VAE; Load Checkpoint needs all three.");
    }

    public SdFamily Family => _ck.Family;
    public PredictionType Prediction => _ck.Prediction == SdPrediction.V ? PredictionType.VPrediction : PredictionType.Eps;
    public int TextWidth => _ck.TextEncoder!.Width;

    public Task<TextEncoding> EncodeAsync(int[] ids, int[]? ids2, int? layer, CancellationToken cancellationToken) =>
        SdFamilies.EncodeAsync(_ck.TextEncoder!, Family, ids, ids2, layer, cancellationToken);

    public UNetRun BeginRun(Accelerator accelerator) => new SdUNetRun(_ck.UNet!, accelerator);

    public SdUNet? UNet => _ck.UNet;
    public SdTextEncoder? TextEncoder => _ck.TextEncoder;
    public SdVae? Vae => _ck.Vae;
    public SpawnDev.ILGPU.ML.Hub.IModelSource Source => _source;
    /// <summary>The loaded file (Save Checkpoint writes it back with its current weights).</summary>
    public SdCheckpoint File => _ck;

    public Task<float[]> DecodeAsync(float[] latent, int h, int w, CancellationToken cancellationToken) => _ck.Vae!.DecodeAsync(latent, h, w, cancellationToken);

    public Task<float[]> EncodeImageAsync(float[] image, int H, int W, CancellationToken cancellationToken) => _ck.Vae!.EncodeAsync(image, H, W, cancellationToken);

    internal static async Task<SafetensorsCheckpointModels> LoadAsync(MlRuntime runtime, string repoId, string file, Action<string, int>? onProgress)
    {
        int last = -1;
        var ck = await SdCheckpoint.LoadAsync(runtime.Accelerator, runtime.ModelSource, repoId, file, onProgress: (stage, fraction) =>
        {
            int pct = (int)Math.Round(fraction * 100);
            if (pct == last) return;
            last = pct;
            onProgress?.Invoke(stage, pct);
        });
        try { return new SafetensorsCheckpointModels(ck, runtime.Accelerator, runtime.ModelSource); }
        catch { ck.Dispose(); throw; }
    }

    public void Dispose() => _ck.Dispose();
}

/// <summary>A sampling run on a single-file or standalone UNet: each conditioning entry's context (and SDXL's label) goes to the GPU once per run.</summary>
internal sealed class SdUNetRun(SdUNet unet, Accelerator acc) : UNetRun
{
    readonly Dictionary<CondEntry, Uploaded> _uploaded = new(ReferenceEqualityComparer.Instance);
    MemoryBuffer1D<float, Stride1D.Dense>? _in, _out, _concat;

    public override int ConcatChannels => unet.ConcatChannels;

    sealed record Uploaded(MemoryBuffer1D<float, Stride1D.Dense> Context, int Rows, MemoryBuffer1D<float, Stride1D.Dense>? Label);

    public override async Task<float[]> PredictAsync(float[] input, int h, int w, float timestep, CondEntry entry, IReadOnlyList<ControlCall>? controls,
        UNetOptions options, float[]? concat, CancellationToken cancellationToken)
    {
        var cond = Upload(entry, h, w);
        int n = input.Length;
        if (_in is null || _in.Length < n) { _in?.Dispose(); _in = acc.Allocate1D<float>(n); }
        long outLength = (long)unet.OutChannels * h * w;
        if (_out is null || _out.Length < outLength) { _out?.Dispose(); _out = acc.Allocate1D<float>(outLength); }
        var inView = _in.View.SubView(0, n);
        var outView = _out.View.SubView(0, outLength);
        inView.CopyFromCPU(input);
        List<SdControlResiduals>? residuals = null;
        if (controls is { Count: > 0 })
        {
            // Each ControlNet sees what the UNet sees (the scaled input, the timestep, the entry's text and SDXL's label) plus its hint, into a residual set of its
            // own for the run, so the same loaded net can steer twice in one chain.
            residuals = [];
            foreach (var c in controls)
            {
                if (c.Net.ContextDim != unet.ContextDim)
                    throw new ArgumentException($"The ControlNet takes {c.Net.ContextDim}-wide text but this model's UNet takes {unet.ContextDim}: it was made for another model family.");
                if (!_residuals.TryGetValue(c.HintKey, out var set)) _residuals[c.HintKey] = set = c.Net.CreateResiduals();
                var res = await c.Net.RunAsync(inView, h, w, timestep, cond.Context.View, cond.Rows, Hint(c, 8 * h, 8 * w).View, cond.Label?.View, set, cancellationToken);
                res.Strength = c.Strength;
                residuals.Add(res);
            }
        }
        // The options live on the shared UNet; nodes run one at a time, so setting them per call is enough (and Dispose clears them).
        unet.FreeU = options.FreeU;
        unet.PerturbMiddleSelfAttention = options.PerturbMiddleSelfAttention;
        unet.RecordMiddleSelfAttention = options.RecordMiddleSelfAttention;
        unet.HyperTile = options.HyperTile;
        unet.DeepShrink = options.DeepShrink?.Shrink;
        if (unet.ConcatChannels > 0)
        {
            long concatLength = (long)unet.ConcatChannels * h * w;
            if (concat is null || concat.Length != concatLength)
                throw new InvalidOperationException($"This inpainting UNet takes {unet.ConcatChannels} extra channels; the call gave {concat?.Length ?? 0} values for {concatLength}.");
            if (_concat is null || _concat.Length < concatLength) { _concat?.Dispose(); _concat = acc.Allocate1D<float>(concatLength); }
            var concatView = _concat.View.SubView(0, concatLength);
            concatView.CopyFromCPU(concat);
            await unet.ForwardAsync(inView, concatView, h, w, timestep, cond.Context.View, cond.Rows, cond.Label?.View, outView, residuals, cancellationToken);
        }
        else await unet.ForwardAsync(inView, h, w, timestep, cond.Context.View, cond.Rows, cond.Label?.View, outView, residuals, cancellationToken);
        await acc.SynchronizeAsync();
        return await _out.CopyToHostAsync(0, outLength);
    }

    readonly Dictionary<object, MemoryBuffer1D<float, Stride1D.Dense>> _hints = new();
    readonly Dictionary<object, SdControlResiduals> _residuals = new();

    /// <summary>A control picture on the GPU, once per run and picture.</summary>
    MemoryBuffer1D<float, Stride1D.Dense> Hint(ControlCall c, int height, int width)
    {
        if (!_hints.TryGetValue(c.HintKey, out var buf))
        {
            var hint = c.Hint();
            if (hint.Length != 3L * height * width) throw new InvalidOperationException($"The control picture is {hint.Length} values, the run needs 3 x {height} x {width}.");
            _hints[c.HintKey] = buf = acc.Allocate1D(hint);
        }
        return buf;
    }

    /// <summary>The entry's context (and SDXL's label: pooled text plus the size numbers) on the GPU, once per run.</summary>
    Uploaded Upload(CondEntry entry, int h, int w)
    {
        if (_uploaded.TryGetValue(entry, out var known)) return known;
        int width = entry.Shape[^1];
        if (width != unet.ContextDim)
            throw new ArgumentException($"The conditioning is {width} wide but this model's UNet takes {unet.ContextDim}: it was encoded with another model's CLIP.");
        int rows = entry.Embeddings.Length / width;
        MemoryBuffer1D<float, Stride1D.Dense>? label = null;
        if (unet.IsXL)
        {
            if (entry.Pooled is null)
                throw new ArgumentException("An SDXL model needs the pooled text embedding in its conditioning; encode the prompt with this checkpoint's CLIP.");
            // ComfyUI's SDXL.encode_adm: the size numbers default to the latent's size in pixels, the crop to 0 and the target to the size.
            var size = entry.Size ?? new SdxlSize(w * 8, h * 8, 0, 0, w * 8, h * 8);
            var xl = new SdxlConditioning(entry.Pooled, size.Height, size.Width, size.CropH, size.CropW, size.TargetHeight, size.TargetWidth);
            label = acc.Allocate1D(SdUNet.SdxlLabel(xl, unet.LabelWidth));
        }
        var uploaded = new Uploaded(acc.Allocate1D(entry.Embeddings), rows, label);
        _uploaded[entry] = uploaded;
        return uploaded;
    }

    public override Task<float[]?> ReadMiddleAttentionAsync() => unet.ReadMiddleAttentionAsync();

    public override void Dispose()
    {
        unet.FreeU = null;
        unet.PerturbMiddleSelfAttention = false;
        unet.RecordMiddleSelfAttention = false;
        unet.HyperTile = null;
        unet.DeepShrink = null;
        foreach (var u in _uploaded.Values) { u.Context.Dispose(); u.Label?.Dispose(); }
        _uploaded.Clear();
        foreach (var b in _hints.Values) b.Dispose();
        _hints.Clear();
        foreach (var r in _residuals.Values) r.Dispose();
        _residuals.Clear();
        _in?.Dispose();
        _out?.Dispose();
        _concat?.Dispose();
    }
}
