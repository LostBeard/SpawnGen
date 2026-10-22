using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

[Node(Category = "audio", DisplayName = "Load Audio")]
public sealed class LoadAudio
{
    public static string[] AudioNames() => [.. InputFiles.Store.List(n => n.EndsWith(".wav", StringComparison.OrdinalIgnoreCase))];

    public async Task<AudioClip> ExecuteAsync([Input(OptionsProvider = nameof(AudioNames))] string audio) =>
        WavFile.Decode(await InputFiles.Store.ReadAsync(audio));
}

[Node(Category = "audio", DisplayName = "Save Audio", IsOutput = true)]
public sealed class SaveAudio
{
    /// <summary>Writes the clip as a 16-bit WAV file next to the saved pictures (<c>prefix_00001_.wav</c>).</summary>
    public async Task<UiOutput> ExecuteAsync(AudioClip audio, IServiceProvider services, string filenamePrefix = "ComfyUI")
    {
        var file = await Results.For(services).SaveAsync("output", filenamePrefix, WavFile.Encode(audio), "wav");
        return new UiOutput("audio", new object[] { file });
    }
}

[Node(Category = "audio", DisplayName = "Preview Audio", IsOutput = true)]
public sealed class PreviewAudio
{
    public async Task<UiOutput> ExecuteAsync(AudioClip audio, IServiceProvider services)
    {
        var file = await Results.For(services).SaveAsync("temp", "preview", WavFile.Encode(audio), "wav");
        return new UiOutput("audio", new object[] { file });
    }
}

[Node(Category = "audio", DisplayName = "Audio Info")]
public sealed class AudioInfo
{
    public (double Seconds, long SampleRate, long Channels) Execute(AudioClip audio) => (audio.Seconds, audio.SampleRate, audio.ChannelCount);
}
