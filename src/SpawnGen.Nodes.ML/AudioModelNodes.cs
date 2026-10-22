using System.Collections.Concurrent;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.Phonemizer;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.ML;

[Node(Category = "audio", DisplayName = "Kokoro Text to Speech")]
public sealed class KokoroTextToSpeech
{
    const string Repo = KokoroVoicePack.VoiceRepoId;

    static readonly Lazy<EnglishPhonemizer> Phonemizer = new(EmbeddedData.CreatePhonemizer, isThreadSafe: true);
    static readonly ConcurrentDictionary<string, Task<KokoroVoicePack>> Voices_ = new(StringComparer.OrdinalIgnoreCase);

    public static string[] Voices() => [.. KokoroVoicePack.EnglishVoiceNames];

    /// <summary>
    /// Speaks the text with Kokoro-82M (English, the built-in voices; it cannot clone a voice). 24 kHz mono. The model (about 330 MB) downloads on first use.
    /// </summary>
    public async Task<AudioClip> ExecuteAsync(NodeContext context, IServiceProvider services,
        [Input(Multiline = true)] string text = "",
        [Input(OptionsProvider = nameof(Voices))] string voice = KokoroVoicePack.DefaultVoiceName,
        [Input(Min = 0.5, Max = 2, Step = 0.05)] float speed = 1f)
    {
        if (string.IsNullOrWhiteSpace(text)) throw new ArgumentException("There is nothing to say: the text is empty.", nameof(text));
        if (!KokoroVoicePack.EnglishVoiceNames.Contains(voice, StringComparer.OrdinalIgnoreCase))
            throw new ArgumentException($"Unknown voice '{voice}'. Known: {string.Join(", ", KokoroVoicePack.EnglishVoiceNames)}.", nameof(voice));
        var runtime = await GpuHelpers.RuntimeAsync(services);
        context.Status("Loading Kokoro");
        var pipeline = await runtime.GetOrLoadAsync($"kokoro:{Repo}", async () =>
        {
            await using var stream = await runtime.ModelSource.OpenAsync(Repo, "onnx/model.onnx", context.CancellationToken);
            return await KokoroPipeline.CreateFromStreamAsync(runtime.Accelerator, stream, context.CancellationToken);
        });
        var pack = await Voices_.GetOrAdd(voice, async name => KokoroVoicePack.FromBytes(name, await runtime.ModelSource.FetchBytesAsync(Repo, KokoroVoicePack.VoicePath(name), context.CancellationToken)));

        context.Status("Speaking");
        var audio = await pipeline.SpeakAsync(Phonemizer.Value.ToSymbols(text), pack, speed, context.CancellationToken);
        return AudioClip.Mono(audio.Samples, audio.SampleRate);
    }
}

[Node(Category = "audio", DisplayName = "Speech to Text (Parakeet)")]
public sealed class SpeechToText
{
    const string Repo = "istupakov/parakeet-tdt-0.6b-v3-onnx";

    /// <summary>
    /// Writes down what is said in the clip with NVIDIA Parakeet TDT 0.6B v3 (25 European languages, punctuation and capitals included). Stereo is mixed to mono and any sample rate
    /// is resampled to 16 kHz. Needs a desktop-class GPU (the encoder is 2.4 GB) and about 2.6 GB of download on first use.
    /// </summary>
    public async Task<string> ExecuteAsync(AudioClip audio, NodeContext context, IServiceProvider services)
    {
        var runtime = await GpuHelpers.RuntimeAsync(services);
        context.Status("Loading Parakeet");
        var pipeline = await runtime.GetOrLoadAsync($"parakeet:{Repo}", async () =>
        {
            async Task<SpawnDev.ILGPU.ML.InferenceSession> Open(string file, string? dataFile = null)
            {
                await using var graph = await runtime.ModelSource.OpenAsync(Repo, file, context.CancellationToken);
                if (dataFile is null) return await SpawnDev.ILGPU.ML.InferenceSession.CreateFromOnnxStreamAsync(runtime.Accelerator, graph, ct: context.CancellationToken);
                await using var data = await runtime.ModelSource.OpenAsync(Repo, dataFile, context.CancellationToken);
                return await SpawnDev.ILGPU.ML.InferenceSession.CreateFromOnnxStreamAsync(runtime.Accelerator, graph, externalDataStream: data, ct: context.CancellationToken);
            }
            var pre = await Open("nemo128.onnx");
            var enc = await Open("encoder-model.onnx", "encoder-model.onnx.data");
            var dec = await Open("decoder_joint-model.onnx");
            string vocab;
            await using (var v = await runtime.ModelSource.OpenAsync(Repo, "vocab.txt", context.CancellationToken))
            using (var reader = new StreamReader(v)) vocab = await reader.ReadToEndAsync(context.CancellationToken);
            return new ParakeetTdtPipeline(pre, enc, dec, runtime.Accelerator, ParakeetTdtPipeline.ParseVocabulary(vocab));
        });

        var mono = new float[audio.Length];
        foreach (var channel in audio.Channels)
            for (int i = 0; i < mono.Length; i++) mono[i] += channel[i] / audio.ChannelCount;
        context.Status("Listening");
        var result = await pipeline.TranscribeAsync(mono, audio.SampleRate);
        return result.Text;
    }
}

[Node(Category = "text", DisplayName = "Text Generation (LLM)")]
public sealed class TextGeneration
{
    /// <summary>The small chat models the node can run, by the name it shows: repo and file on Hugging Face (all GGUF, all open weights).</summary>
    public static readonly IReadOnlyDictionary<string, (string Repo, string File)> Models = new Dictionary<string, (string, string)>
    {
        ["qwen2.5-0.5b-instruct"] = ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf"),
        ["qwen2.5-1.5b-instruct"] = ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
        ["qwen3-0.6b"] = ("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf"),
        ["smollm2-360m-instruct"] = ("HuggingFaceTB/SmolLM2-360M-Instruct-GGUF", "smollm2-360m-instruct-q8_0.gguf"),
        ["gemma-3-270m-it"] = ("ggml-org/gemma-3-270m-it-GGUF", "gemma-3-270m-it-Q8_0.gguf"),
    };

    public static string[] ModelNames() => [.. Models.Keys];

    /// <summary>
    /// Answers the prompt with a small local language model (the model downloads on first use, 0.3 to 1 GB). Link the text into a CLIP Text Encode to let the model write the
    /// image prompt. With temperature 0 the answer is the model's most likely one and repeats exactly.
    /// </summary>
    public async Task<string> ExecuteAsync(NodeContext context, IServiceProvider services,
        [Input(Multiline = true)] string prompt = "",
        [Input(Multiline = true)] string systemPrompt = "You are a helpful assistant.",
        [Input(OptionsProvider = nameof(ModelNames))] string model = "qwen2.5-0.5b-instruct",
        [Input(Min = 1, Max = 2048)] int maxNewTokens = 128,
        [Input(Min = 0, Max = 2, Step = 0.05)] float temperature = 0f)
    {
        if (string.IsNullOrWhiteSpace(prompt)) throw new ArgumentException("The prompt is empty.", nameof(prompt));
        if (!Models.TryGetValue(model, out var m)) throw new ArgumentException($"Unknown model '{model}'. Known: {string.Join(", ", Models.Keys)}.", nameof(model));
        var runtime = await GpuHelpers.RuntimeAsync(services);
        context.Status($"Loading {model}");
        var pipeline = await runtime.GetOrLoadAsync($"llm:{m.Repo}/{m.File}", async () =>
        {
            await using var stream = await runtime.ModelSource.OpenAsync(m.Repo, m.File, context.CancellationToken);
            return await GgufTextGenerationPipeline.CreateFromStreamAsync(runtime.Accelerator, stream, ct: context.CancellationToken);
        });
        context.Status("Writing");
        var config = new GenerationConfig { MaxNewTokens = maxNewTokens, Strategy = temperature <= 0 ? "greedy" : "top_p", Temperature = Math.Max(temperature, 0.01f) };
        var messages = new List<(string, string)>();
        if (!string.IsNullOrWhiteSpace(systemPrompt)) messages.Add(("system", systemPrompt));
        messages.Add(("user", prompt));
        int tokens = 0;
        var text = await pipeline.GenerateAsync(messages, config, thinking: false, onToken: (_, _) => { context.Progress(++tokens, maxNewTokens); return Task.CompletedTask; }, ct: context.CancellationToken);
        return text.Trim();
    }
}
