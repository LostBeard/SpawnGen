using System.Text.Json.Nodes;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.ML;

/// <summary>A CLIP whose file is a Hugging Face Qwen3 causal language model (qwen_3_4b.safetensors): Generate Text runs it as one.</summary>
internal interface IQwen3File
{
    (string Repo, string File) Qwen3File { get; }
}

/// <summary>
/// A Qwen3 text-encoder file loaded with a CLIP type SpawnGen does not encode prompts with (ComfyUI's llm_qwen3_text_gen template loads qwen_3_4b
/// with type ltxv): Generate Text can run it; CLIP Text Encode cannot.
/// </summary>
internal sealed class Qwen3TextModel(string name, string type, string repo, string file) : ITextEncoderModel, IQwen3File
{
    public (string Repo, string File) Qwen3File => (repo, file);
    public string Name => name;
    public SdFamily Family => SdFamily.SD1;
    public BPETokenizer Tokenizer => throw new NotSupportedException(Refusal);
    public int FirstTowerWidth => 0;
    public SpawnDev.ILGPU.ML.StableDiffusion.SdTextEncoder? TextEncoder => null;
    string Refusal => $"'{name}' loaded as CLIP type '{type}' runs in Generate Text only; to encode prompts with it choose type flux2 (FLUX.2) or lumina2 (Z-Image).";
    public Task<TextEncoding> EncodeAsync(int[] ids, int[]? ids2 = null, int? layer = null, CancellationToken cancellationToken = default) => throw new NotSupportedException(Refusal);
    public Task<TextEncoding> EmptyPromptAsync(int? layer, IReadOnlyList<WeightStep> weights) => throw new NotSupportedException(Refusal);
    public Task UseWeightsAsync(IReadOnlyList<WeightStep> weights, CancellationToken cancellationToken) => Task.CompletedTask;

    /// <summary>Whether a file's tensors are a Hugging Face Qwen3 causal LM's (any prefix before model.).</summary>
    public static bool Is(IReadOnlyCollection<string> names) =>
        names.Any(n => n.EndsWith("model.embed_tokens.weight", StringComparison.Ordinal)) && names.Any(n => n.EndsWith("self_attn.q_norm.weight", StringComparison.Ordinal));
}

/// <summary>A Qwen3 file loaded as a language model on the device (kept between runs; it gives way to the image models' transformers).</summary>
internal sealed class Qwen3Generator(GgufGenerator generator, SentencePieceTokenizer tokenizer, Stream stream) : IDisposable
{
    public GgufGenerator Generator { get; } = generator;
    public SentencePieceTokenizer Tokenizer { get; } = tokenizer;
    public void Dispose() { Generator.Dispose(); stream.Dispose(); }
}

/// <summary>
/// Generate Text (ComfyUI's TextGenerate): the CLIP's language model answers the prompt. Measured with ComfyUI (2026-11-14, qwen_3_4b, black box:
/// the ids it feeds the model): the prompt in Qwen3's chat template with no system turn; thinking off adds the empty think block; with thinking on
/// the model opens it itself and max_length counts those tokens; the thinking text comes out on "thinking" without its tags, and the answer is
/// empty when the thinking used the budget. Greedy with sampling off. Runs on SpawnDev.ILGPU.ML's GGUF decoder over the safetensors file
/// (no conversion); on a card the 16-bit weights do not fit (8 GB) the library loads them 8-bit (Q8_0), and greedy output then follows ComfyUI's
/// for its first tokens (13 of 23 on the measured prompt) rather than all of them. Pictures, video and audio are not taken yet; min_p and
/// presence_penalty wait for the library.
/// </summary>
[Node("TextGenerate", Category = "text", DisplayName = "Generate Text", OutputNames = ["generated_text", "thinking"])]
public sealed class TextGenerate
{
    /// <summary>Qwen3's tokenizer and chat template (Apache-2.0): ComfyUI's text-encoder file carries neither.</summary>
    public const string TokenizerRepo = "Qwen/Qwen3-4B";

    public static IEnumerable<DynamicOption> SamplingModes() =>
    [
        new("on",
            DynamicInput.Float("temperature", 0.7, 0.01, 2.0, 1e-6),
            DynamicInput.Int("top_k", 64, 0, 1000),
            DynamicInput.Float("top_p", 0.95, 0.0, 1.0, 0.01),
            DynamicInput.Float("min_p", 0.05, 0.0, 1.0, 0.01),
            DynamicInput.Float("repetition_penalty", 1.05, 0.0, 5.0, 0.01),
            DynamicInput.Int("seed", 0, 0, 18446744073709551615d),
            DynamicInput.Float("presence_penalty", 0.0, 0.0, 5.0, 0.01) with { Required = false }),
        new("off"),
    ];

    public static string[] MtpModes() => ["auto", "off", "2", "3", "4", "5", "tree2s1", "tree2s2", "tree3s1", "tree4s1m8"];

    public async Task<(string GeneratedText, string Thinking)> ExecuteAsync(ClipHandle clip, NodeContext context, IServiceProvider services,
        [Input(Multiline = true)] string prompt = "",
        [Input(Min = 1, Max = 32768)] long maxLength = 512,
        [DynamicCombo(nameof(SamplingModes))] DynamicChoice? samplingMode = null,
        [Input(Optional = true)] ImageBatch? image = null,
        [Input(Optional = true, Type = "IMAGE", Tooltip = "Video frames as image batch. Assumed to be 24 FPS; subsampled to 1 FPS internally.")] ImageBatch? video = null,
        [Input(Optional = true)] AudioClip? audio = null,
        [Input(Optional = true, Tooltip = "Operate in thinking mode if the model supports it.")] bool thinking = false,
        [Input(Optional = true, Tooltip = "Use the built in system prompt/template if the model has one.")] bool useDefaultTemplate = true,
        [Input(Optional = true, OptionsProvider = nameof(MtpModes),
            Tooltip = "Speculative decoding with the checkpoint's multi-token-prediction head. No effect without MTP weights. auto adapts the draft depth and breadth to observed token matches and execution cost; 2-5 pins a constant draft depth; tree modes pin both breadth and depth. Sampled output stays correctly distributed but differs from non-MTP output for the same seed.")]
        string mtp = "auto",
        [Input(Optional = true, ForceInput = true, Tooltip = "Replaces the system prompt in the model's chat template. Ignored when the default template is not used.")]
        string? systemPrompt = null)
    {
        if (image is not null || video is not null || audio is not null)
            throw new NotSupportedException("Generate Text takes text only in SpawnGen so far (pictures, video and audio need a vision / audio language model in the library).");
        if (clip.Encoder is not IQwen3File qwen)
            throw new NotSupportedException($"Generate Text runs Qwen3 language models (qwen_3_4b.safetensors); '{clip.Encoder.Name}' is not one.");
        var runtime = await GpuHelpers.RuntimeAsync(services);
        var (repo, file) = qwen.Qwen3File;
        // The image models' transformers give way (an 8 GB card holds one big model at a time).
        Flux2Residency.Release();
        ZImageResidency.Release();
        Flux1Residency.Release();
        var model = await runtime.GetOrLoadAsync($"qwen3-lm:{repo}/{file}", async () =>
        {
            var tokenizerJson = System.Text.Encoding.UTF8.GetString(await runtime.ModelSource.FetchBytesAsync(TokenizerRepo, "tokenizer.json", context.CancellationToken));
            var config = JsonNode.Parse(System.Text.Encoding.UTF8.GetString(await runtime.ModelSource.FetchBytesAsync(TokenizerRepo, "tokenizer_config.json", context.CancellationToken)));
            var chatTemplate = (string?)config?["chat_template"] ?? "";
            var stream = await runtime.ModelSource.OpenAsync(repo, file, context.CancellationToken);
            try
            {
                var gguf = await SafeTensorsGguf.FromHfQwen3Async(stream, tokenizerJson, chatTemplate);
                var generator = await GgufGenerator.CreateAsync(runtime.Accelerator, stream, 4096, gguf);
                return new Qwen3Generator(generator, SentencePieceTokenizer.FromGGUF(gguf)!, stream);
            }
            catch { stream.Dispose(); throw; }
        });

        int[] ids;
        if (useDefaultTemplate)
        {
            var turns = new List<(string, string)>();
            if (!string.IsNullOrEmpty(systemPrompt)) turns.Add(("system", systemPrompt));
            turns.Add(("user", prompt));
            ids = ChatTemplates.BuildChatMLPromptTokens(model.Tokenizer, turns, emptyThinking: !thinking);
        }
        else ids = model.Tokenizer.Encode(prompt);

        var config = new GenerationConfig { MaxNewTokens = (int)Math.Clamp(maxLength, 1, 32768), Strategy = "greedy" };
        if (samplingMode?.Key == "on")
        {
            config.Strategy = "sample";
            config.Temperature = (float)samplingMode.Get("temperature", 0.7);
            config.TopK = (int)samplingMode.Get("top_k", 64L);
            config.TopP = (float)samplingMode.Get("top_p", 0.95);
            config.RepetitionPenalty = (float)samplingMode.Get("repetition_penalty", 1.05);
            config.Seed = (int)(samplingMode.Get("seed", 0L) & int.MaxValue);
        }
        int produced = 0;
        var result = await model.Generator.GenerateAsync(ids, config, null, null, delta =>
        {
            produced++;
            context.Progress(produced, (double)config.MaxNewTokens, "Generating text");
            return Task.CompletedTask;
        }, context.CancellationToken);
        return Split(result.Text, thinking);
    }

    /// <summary>
    /// The model's text as ComfyUI splits it (measured): with thinking, what precedes "&lt;/think&gt;" is the thinking (its tags taken off) and the
    /// rest the answer; a thinking run cut off before "&lt;/think&gt;" is all thinking and no answer.
    /// </summary>
    public static (string GeneratedText, string Thinking) Split(string text, bool thinking)
    {
        int end = text.IndexOf("</think>", StringComparison.Ordinal);
        if (end >= 0)
        {
            var think = text[..end];
            int start = think.IndexOf("<think>", StringComparison.Ordinal);
            if (start >= 0) think = think[(start + 7)..];
            return (text[(end + 8)..].Trim(), think.Trim());
        }
        if (thinking)
        {
            int start = text.IndexOf("<think>", StringComparison.Ordinal);
            return ("", (start >= 0 ? text[(start + 7)..] : text).Trim());
        }
        return (text.Trim(), "");
    }
}
