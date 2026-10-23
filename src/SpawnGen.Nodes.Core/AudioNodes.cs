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

[Node(Category = "audio", DisplayName = "Save Audio (FLAC) (DEPRECATED)", IsOutput = true)]
public sealed class SaveAudio
{
    /// <summary>
    /// Writes the clip as a 16-bit WAV file (<c>prefix_00001_.wav</c>) and passes it on. ComfyUI writes FLAC; the 16-bit samples are the same (x 32768,
    /// rounded half to even, clipped), only the container differs: there is no FLAC encoder here.
    /// </summary>
    public async Task<(AudioClip Audio, UiOutput Ui)> ExecuteAsync(AudioClip? audio, IServiceProvider services, string filenamePrefix = "audio/ComfyUI")
    {
        if (audio is null) throw new ArgumentException("SaveAudio: input audio is None (source video may have no audio track).", nameof(audio));
        var file = await Results.For(services).SaveAsync("output", filenamePrefix, WavFile.Encode(audio), "wav");
        return (audio, new UiOutput("audio", new object[] { file }));
    }
}

[Node(Category = "audio", DisplayName = "Preview Audio", IsOutput = true, Description = "Preview the audio without saving it to the ComfyUI output directory.")]
public sealed class PreviewAudio
{
    public async Task<(AudioClip Audio, UiOutput Ui)> ExecuteAsync(AudioClip? audio, IServiceProvider services)
    {
        if (audio is null) throw new ArgumentException("PreviewAudio: input audio is None (source video may have no audio track).", nameof(audio));
        var file = await Results.For(services).SaveAsync("temp", "preview", WavFile.Encode(audio), "wav");
        return (audio, new UiOutput("audio", new object[] { file }));
    }
}

[Node(Category = "audio", DisplayName = "Audio Info")]
public sealed class AudioInfo
{
    public (double Seconds, long SampleRate, long Channels) Execute(AudioClip audio) => (audio.Seconds, audio.SampleRate, audio.ChannelCount);
}
