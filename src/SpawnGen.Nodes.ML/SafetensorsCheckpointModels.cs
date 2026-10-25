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

    SafetensorsCheckpointModels(SdCheckpoint ck, Accelerator acc)
    {
        _ck = ck;
        _acc = acc;
        if (ck.UNet is null || ck.TextEncoder is null || ck.Vae is null)
            throw new InvalidDataException("The checkpoint file lacks a UNet, text encoder or VAE; Load Checkpoint needs all three.");
    }

    public SdFamily Family => _ck.Family;
    public PredictionType Prediction => _ck.Prediction == SdPrediction.V ? PredictionType.VPrediction : PredictionType.Eps;
    public int TextWidth => _ck.TextEncoder!.Width;

    public async Task<TextEncoding> EncodeAsync(int[] ids, int[]? ids2, int? layer, CancellationToken cancellationToken)
    {
        // ComfyUI's padding per tower: CLIP-L (SD 1.x, SDXL's first) with the end token, OpenCLIP (SD 2.x, SDXL's second) with 0.
        var first = ClipPadding.Pad(ids, Family == SdFamily.SD2 ? 0 : ClipPadding.End);
        var second = Family == SdFamily.SDXL ? ClipPadding.Pad(ids2 ?? ids, 0) : null;
        var te = await _ck.TextEncoder!.EncodeAsync(first, second, layer, cancellationToken);
        return new TextEncoding(te.Context, [1, te.Rows, te.Width], te.Pooled);
    }

    public UNetRun BeginRun(Accelerator accelerator) => new Run(_ck.UNet!, accelerator);

    sealed class Run(SdUNet unet, Accelerator acc) : UNetRun
    {
        readonly Dictionary<CondEntry, Uploaded> _uploaded = new(ReferenceEqualityComparer.Instance);
        MemoryBuffer1D<float, Stride1D.Dense>? _in, _out;

        sealed record Uploaded(MemoryBuffer1D<float, Stride1D.Dense> Context, int Rows, MemoryBuffer1D<float, Stride1D.Dense>? Label);

        public override async Task<float[]> PredictAsync(float[] input, int h, int w, float timestep, CondEntry entry, CancellationToken cancellationToken)
        {
            var cond = Upload(entry, h, w);
            int n = input.Length;
            if (_in is null || _in.Length < n) { _in?.Dispose(); _in = acc.Allocate1D<float>(n); }
            long outLength = (long)unet.OutChannels * h * w;
            if (_out is null || _out.Length < outLength) { _out?.Dispose(); _out = acc.Allocate1D<float>(outLength); }
            var inView = _in.View.SubView(0, n);
            var outView = _out.View.SubView(0, outLength);
            inView.CopyFromCPU(input);
            await unet.ForwardAsync(inView, h, w, timestep, cond.Context.View, cond.Rows, cond.Label?.View, outView, cancellationToken);
            await acc.SynchronizeAsync();
            return await _out.CopyToHostAsync(0, outLength);
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
                // SDXL base: 1280 pooled + 6 x 256 = 2816 (the width of label_emb.0.0.weight).
                label = acc.Allocate1D(SdUNet.SdxlLabel(xl, entry.Pooled.Length + 6 * 256));
            }
            var uploaded = new Uploaded(acc.Allocate1D(entry.Embeddings), rows, label);
            _uploaded[entry] = uploaded;
            return uploaded;
        }

        public override void Dispose()
        {
            foreach (var u in _uploaded.Values) { u.Context.Dispose(); u.Label?.Dispose(); }
            _uploaded.Clear();
            _in?.Dispose();
            _out?.Dispose();
        }
    }

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
        try { return new SafetensorsCheckpointModels(ck, runtime.Accelerator); }
        catch { ck.Dispose(); throw; }
    }

    public void Dispose() => _ck.Dispose();
}
