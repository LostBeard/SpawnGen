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
    /// Writes the clip as a 16-bit FLAC file with the prompt and workflow as Vorbis comments, like ComfyUI (<see cref="AudioFiles"/>), and passes it on.
    /// Without SpawnDev.Codecs (a host with no accelerator) the same 16-bit samples go into a WAV file.
    /// </summary>
    public async Task<(AudioClip Audio, UiOutput Ui)> ExecuteAsync(AudioClip? audio, NodeContext context, IServiceProvider services, string filenamePrefix = "audio/ComfyUI")
    {
        if (audio is null) throw new ArgumentException("SaveAudio: input audio is None (source video may have no audio track).", nameof(audio));
        var (data, extension) = await AudioFiles.EncodeAsync(audio, Results.Metadata(context));
        var file = await Results.For(services).SaveAsync("output", filenamePrefix, data, extension, trailingUnderscore: false); // ComfyUI: prefix_00001.flac
        return (audio, new UiOutput("audio", new object[] { file }));
    }
}

[Node(Category = "audio", DisplayName = "Preview Audio", IsOutput = true, Description = "Preview the audio without saving it to the ComfyUI output directory.")]
public sealed class PreviewAudio
{
    /// <summary>A temporary FLAC file (ComfyUI's PreviewAudio saves like Save Audio, into temp), WAV without SpawnDev.Codecs.</summary>
    public async Task<(AudioClip Audio, UiOutput Ui)> ExecuteAsync(AudioClip? audio, NodeContext context, IServiceProvider services)
    {
        if (audio is null) throw new ArgumentException("PreviewAudio: input audio is None (source video may have no audio track).", nameof(audio));
        var (data, extension) = await AudioFiles.EncodeAsync(audio, Results.Metadata(context));
        var file = await Results.For(services).SaveAsync("temp", "preview", data, extension);
        return (audio, new UiOutput("audio", new object[] { file }));
    }
}

[Node(Category = "audio", DisplayName = "Audio Info")]
public sealed class AudioInfo
{
    public (double Seconds, long SampleRate, long Channels) Execute(AudioClip audio) => (audio.Seconds, audio.SampleRate, audio.ChannelCount);
}
