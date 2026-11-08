using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

[Node(Category = "audio", DisplayName = "Load Audio")]
public sealed class LoadAudio
{
    public static string[] AudioNames() =>
        [.. InputFiles.Store.List(n => AudioFiles.LoadableExtensions.Any(e => n.EndsWith(e, StringComparison.OrdinalIgnoreCase)))];

    /// <summary>A WAV or FLAC file from the inputs (FLAC through SpawnDev.Codecs), as floats the way ComfyUI's PyAV load gives them.</summary>
    public async Task<AudioClip> ExecuteAsync([Input(OptionsProvider = nameof(AudioNames))] string audio) =>
        await AudioFiles.LoadAsync(await InputFiles.Store.ReadAsync(audio));
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

[Node(Category = "audio", DisplayName = "Save Audio (Opus) (DEPRECATED)", IsOutput = true)]
public sealed class SaveAudioOpus
{
    public static string[] Qualities() => AudioFiles.OpusQualities;

    /// <summary>
    /// Writes the clip as an Ogg Opus file (prefix_00001.opus) at the chosen bit rate, resampled to an Opus rate first, with the prompt and workflow as
    /// comments, like ComfyUI. SpawnDev.Codecs' Opus encoder is a port of libopus' fixed-point build, so the packets are valid but not libopus' float
    /// build's bytes (ComfyUI's PyAV).
    /// </summary>
    public async Task<(AudioClip Audio, UiOutput Ui)> ExecuteAsync(AudioClip? audio, NodeContext context, IServiceProvider services, string filenamePrefix = "audio/ComfyUI",
        [Input(OptionsProvider = nameof(Qualities))] string quality = "128k")
    {
        if (audio is null) throw new ArgumentException("SaveAudioOpus: input audio is None (source video may have no audio track).", nameof(audio));
        var (data, extension) = await AudioFiles.EncodeAsync(audio, Results.Metadata(context), "opus", quality);
        var file = await Results.For(services).SaveAsync("output", filenamePrefix, data, extension, trailingUnderscore: false);
        return (audio, new UiOutput("audio", new object[] { file }));
    }
}

[Node(Category = "audio", DisplayName = "Save Audio (Advanced)", IsOutput = true, Description = "Saves the input audio to your ComfyUI output directory.")]
public sealed class SaveAudioAdvanced
{
    public static DynamicOption[] Formats() =>
    [
        new("flac"),
        new("mp3", DynamicInput.Combo("quality", AudioFiles.Mp3Qualities, "V0")),
        new("opus", DynamicInput.Combo("quality", AudioFiles.OpusQualities, "128k")),
    ];

    /// <summary>ComfyUI's Save Audio (Advanced): FLAC, or Opus at a bit rate (see <see cref="SaveAudioOpus"/>); MP3 waits for SpawnDev.Codecs' MP3 encoder.</summary>
    public async Task<(AudioClip Audio, UiOutput Ui)> ExecuteAsync(NodeContext context, IServiceProvider services,
        [Input(Tooltip = "The audio to save.")] AudioClip audio,
        [Input(Default = "audio/ComfyUI", Tooltip = "The prefix for the file to save. May include formatting tokens such as %date:yyyy-MM-dd%.")] string filenamePrefix,
        [DynamicCombo(nameof(Formats)), Input(Tooltip = "The file format in which to save the audio.")] DynamicChoice format)
    {
        var quality = format.Get<string?>("quality", null);
        var (data, extension) = await AudioFiles.EncodeAsync(audio, Results.Metadata(context), format.Key, quality);
        var file = await Results.For(services).SaveAsync("output", filenamePrefix, data, extension, trailingUnderscore: false);
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
