using System.Collections.Concurrent;
using SpawnDev.ILGPU.ML.Pipelines;
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
