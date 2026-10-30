using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnGen.Nodes.ML;

/// <summary>
/// A checkpoint as an Optimum-style ONNX export (text_encoder, unet, vae_decoder, vae_encoder folders), run by the library's ONNX graph executor.
/// The sub-models have fixed shapes (batch 1, 77 tokens, 64x64 latent); the UNet and VAE are recompiled for other sizes by the session on first run.
/// </summary>
internal sealed class OnnxCheckpointModels : ICheckpointModels
{
    readonly MlRuntime _runtime;
    readonly string _repoId;
    readonly object _encoderGate = new();
    Task<InferenceSession>? _vaeEncoder;

    OnnxCheckpointModels(MlRuntime runtime, string repoId, InferenceSession textEncoder, InferenceSession unet, InferenceSession vae, int textWidth)
    {
        _runtime = runtime;
        _repoId = repoId;
        TextEncoder = textEncoder;
        Unet = unet;
        Vae = vae;
        TextWidth = textWidth;
    }

    public InferenceSession TextEncoder { get; }
    public InferenceSession Unet { get; }
    public InferenceSession Vae { get; }
    public int TextWidth { get; }
    /// <summary>The exports SpawnGen knows are SD 1.x (CLIP-L, 768 wide) or SD 2.x (OpenCLIP-H, 1024: SD 2.1, SD-Turbo), all eps-prediction.</summary>
    public SdFamily Family => TextWidth == 1024 ? SdFamily.SD2 : SdFamily.SD1;
    public PredictionType Prediction => PredictionType.Eps;

    public async Task<TextEncoding> EncodeAsync(int[] ids, int[]? ids2, int? layer, CancellationToken cancellationToken)
    {
        // The export's text encoder ends at the layer the model was trained with (diffusers drops SD 2.x's last layer): it cannot read another one.
        int defaultLayer = Family == SdFamily.SD1 ? -1 : -2;
        if (layer is { } l && l != defaultLayer)
            throw new NotSupportedException($"This ONNX checkpoint's text encoder only gives CLIP layer {defaultLayer}; CLIP Set Last Layer {l} needs the model as a .safetensors checkpoint.");
        ids = ClipPadding.Pad(ids, ClipPadding.ForTextWidth(TextWidth));
        var acc = _runtime.Accelerator;
        var (buf, tokens) = GpuHelpers.Upload(acc, ids.Select(t => (float)t).ToArray(), [1, ids.Length]);
        using (buf)
        {
            var outputs = await TextEncoder.RunAsync(new Dictionary<string, Tensor> { [TextEncoder.InputNames[0]] = tokens });
            var embeddings = outputs[TextEncoder.OutputNames[0]];
            int count = embeddings.Shape.Aggregate(1, (x, y) => x * y);
            return new TextEncoding(await GpuHelpers.ReadAsync(acc, embeddings, count), embeddings.Shape.ToArray(), null);
        }
    }

    public UNetRun BeginRun(Accelerator accelerator) => new Run(Unet, accelerator);

    // An ONNX export has no weights SpawnGen can patch: LoRAs and merges need a .safetensors checkpoint.
    SpawnDev.ILGPU.ML.StableDiffusion.SdUNet? ICheckpointModels.UNet => null;
    SpawnDev.ILGPU.ML.StableDiffusion.SdTextEncoder? ICheckpointModels.TextEncoder => null;
    SpawnDev.ILGPU.ML.StableDiffusion.SdVae? ICheckpointModels.Vae => null;
    SpawnDev.ILGPU.ML.StableDiffusion.SdCheckpoint? ICheckpointModels.File => null;
    SpawnDev.ILGPU.ML.Hub.IModelSource ICheckpointModels.Source => _runtime.ModelSource;

    sealed class Run(InferenceSession unet, Accelerator acc) : UNetRun
    {
        readonly Dictionary<CondEntry, (MemoryBuffer1D<float, Stride1D.Dense> Buffer, Tensor Tensor)> _uploaded = new(ReferenceEqualityComparer.Instance);

        public override async Task<float[]> PredictAsync(float[] input, int h, int w, float timestep, CondEntry entry, IReadOnlyList<ControlCall>? controls,
            UNetOptions options, float[]? concat, CancellationToken cancellationToken)
        {
            if (controls is { Count: > 0 })
                throw new NotSupportedException("ControlNets steer single-file checkpoints; this is an ONNX export. Load the model as a .safetensors checkpoint.");
            if (!options.IsNone)
                throw new NotSupportedException("UNet patches (FreeU, PAG, SAG, HyperTile, deep shrink) work on single-file checkpoints; this is an ONNX export. Load the model as a .safetensors checkpoint.");
            if (!_uploaded.TryGetValue(entry, out var cond))
                _uploaded[entry] = cond = GpuHelpers.Upload(acc, entry.Embeddings, entry.Shape);
            var (sampleBuf, sampleTensor) = GpuHelpers.Upload(acc, input, [1, Checkpoint.LatentChannels, h, w]);
            var (tBuf, tTensor) = GpuHelpers.Upload(acc, [timestep], [1]);
            using (sampleBuf)
            using (tBuf)
            {
                var outputs = await unet.RunAsync(new Dictionary<string, Tensor>
                {
                    [unet.InputNames[0]] = sampleTensor,
                    [unet.InputNames[1]] = tTensor,
                    [unet.InputNames[2]] = cond.Tensor,
                });
                return await GpuHelpers.ReadAsync(acc, outputs[unet.OutputNames[0]], input.Length);
            }
        }

        public override void Dispose()
        {
            foreach (var (buf, _) in _uploaded.Values) buf.Dispose();
            _uploaded.Clear();
        }
    }

    public async Task<float[]> DecodeAsync(float[] latent, int h, int w, CancellationToken cancellationToken)
    {
        var acc = _runtime.Accelerator;
        var (buf, tensor) = GpuHelpers.Upload(acc, latent, [1, Checkpoint.LatentChannels, h, w]);
        using (buf)
        {
            var outputs = await Vae.RunAsync(new Dictionary<string, Tensor> { [Vae.InputNames[0]] = tensor });
            return await GpuHelpers.ReadAsync(acc, outputs[Vae.OutputNames[0]], 3 * h * 8 * w * 8);
        }
    }

    public async Task<float[]> EncodeImageAsync(float[] image, int H, int W, CancellationToken cancellationToken)
    {
        var acc = _runtime.Accelerator;
        var session = await GetVaeEncoderAsync();
        var (buf, tensor) = GpuHelpers.Upload(acc, image, [1, 3, H, W]);
        using (buf)
        {
            var outputs = await session.RunAsync(new Dictionary<string, Tensor> { [session.InputNames[0]] = tensor });
            return await GpuHelpers.ReadAsync(acc, outputs[session.OutputNames[0]], Checkpoint.LatentChannels * (H / 8) * (W / 8));
        }
    }

    /// <summary>The VAE encoder (pixels to latents), loaded on first use: text-to-image never needs it.</summary>
    Task<InferenceSession> GetVaeEncoderAsync()
    {
        lock (_encoderGate)
            return _vaeEncoder ??= LoadVaeEncoderAsync();
    }

    async Task<InferenceSession> LoadVaeEncoderAsync()
    {
        await using var m = await _runtime.ModelSource.OpenAsync(_repoId, "vae_encoder/model.onnx");
        var session = await InferenceSession.CreateFromOnnxStreamAsync(_runtime.Accelerator, m,
            inputShapes: new Dictionary<string, int[]> { ["sample"] = [1, 3, 512, 512] });
        session.CacheShapeReadbacks = false; // the encoder is run at whatever size the picture has
        return session;
    }

    internal static async Task<OnnxCheckpointModels> LoadAsync(MlRuntime runtime, string repoId, MlRuntime.CheckpointInfo info, Action<string, int>? onProgress)
    {
        var source = runtime.ModelSource;
        var acc = runtime.Accelerator;
        onProgress?.Invoke("text_encoder", 0);
        InferenceSession te, unet, vae;
        await using (var m = await source.OpenAsync(repoId, "text_encoder/model.onnx"))
            te = await InferenceSession.CreateFromOnnxStreamAsync(acc, m,
                onProgress: (s, p) => onProgress?.Invoke($"text_encoder:{s}", p),
                inputShapes: new Dictionary<string, int[]> { ["input_ids"] = [1, Checkpoint.TokenCount] });
        onProgress?.Invoke("unet", 0);
        await using var unetData = info.UnetExternalData is null ? null : await source.OpenAsync(repoId, info.UnetExternalData);
        await using (var m = await source.OpenAsync(repoId, "unet/model.onnx"))
            unet = await InferenceSession.CreateFromOnnxStreamAsync(acc, m, externalDataStream: unetData,
                onProgress: (s, p) => onProgress?.Invoke($"unet:{s}", p),
                inputShapes: new Dictionary<string, int[]>
                {
                    ["sample"] = [1, Checkpoint.LatentChannels, 64, 64],
                    ["timestep"] = [1],
                    ["encoder_hidden_states"] = [1, Checkpoint.TokenCount, info.TextWidth],
                });
        onProgress?.Invoke("vae_decoder", 0);
        await using (var m = await source.OpenAsync(repoId, "vae_decoder/model.onnx"))
            vae = await InferenceSession.CreateFromOnnxStreamAsync(acc, m,
                onProgress: (s, p) => onProgress?.Invoke($"vae_decoder:{s}", p),
                inputShapes: new Dictionary<string, int[]> { ["latent_sample"] = [1, Checkpoint.LatentChannels, 64, 64] });

        // Shapes are fixed across runs, so later runs can serve shape-constant readbacks from a CPU cache.
        te.CacheShapeReadbacks = true;
        unet.CacheShapeReadbacks = true;
        vae.CacheShapeReadbacks = true;
        onProgress?.Invoke("done", 100);
        return new OnnxCheckpointModels(runtime, repoId, te, unet, vae, info.TextWidth);
    }

    public void Dispose()
    {
        TextEncoder.Dispose();
        Unet.Dispose();
        Vae.Dispose();
        if (_vaeEncoder is { IsCompletedSuccessfully: true }) _vaeEncoder.Result.Dispose();
    }
}
