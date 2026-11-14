using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entries and SpawnGen's tests. WAV is read and written by WavFile; FLAC (and Opus) go
// through AudioFiles, which a host with SpawnDev.Codecs connects.

/// <summary>Load Audio: a WAV or FLAC file from the input folder (an " [input]" annotation is dropped).</summary>
[Node(Category = "audio", DisplayName = "Load Audio")]
public sealed class LoadAudio
{
    public static string[] AudioNames() =>
        [.. InputFiles.Store.List(n => AudioFiles.LoadableExtensions.Contains(Path.GetExtension(n).ToLowerInvariant()))];

    public async Task<AudioClip> ExecuteAsync([Input(OptionsProvider = nameof(AudioNames))] string audio) =>
        await AudioFiles.LoadAsync(await InputFiles.Store.ReadAsync(RecordAudio.StripAnnotation(audio)));
}

/// <summary>
/// Save Audio: the clip as a numbered FLAC with the run's metadata (16-bit WAV without SpawnDev.Codecs) in the output folder (prefix "audio/ComfyUI" gives
/// audio/ComfyUI_00001.flac), shown in the node.
/// </summary>
[Node(Category = "audio", DisplayName = "Save Audio (FLAC) (DEPRECATED)", IsOutput = true)]
public sealed class SaveAudio
{
    public async Task<(AudioClip Audio, UiOutput Ui)> ExecuteAsync(AudioClip? audio, NodeContext context, IServiceProvider services, string filenamePrefix = "audio/ComfyUI") =>
        await AudioResults.SaveAsync(audio, context, services, "output", filenamePrefix);
}

/// <summary>Save Audio (Opus): the clip as a numbered Ogg Opus at the chosen bit rate (resampled to an Opus rate first), needs SpawnDev.Codecs.</summary>
[Node(Category = "audio", DisplayName = "Save Audio (Opus) (DEPRECATED)", IsOutput = true)]
public sealed class SaveAudioOpus
{
    public async Task<(AudioClip Audio, UiOutput Ui)> ExecuteAsync(AudioClip? audio, NodeContext context, IServiceProvider services, string filenamePrefix = "audio/ComfyUI",
        [Input(Options = ["64k", "96k", "128k", "192k", "320k"])] string quality = "128k") =>
        await AudioResults.SaveAsync(audio, context, services, "output", filenamePrefix, "opus", quality);
}

/// <summary>Save Audio (MP3): the clip as MP3 at V0 (variable bit rate), 128k or 320k, encoded by SpawnDev.Codecs (see CodecsAudio.EncodeMp3Async).</summary>
[Node(Category = "audio", DisplayName = "Save Audio (MP3) (DEPRECATED)", IsOutput = true)]
public sealed class SaveAudioMP3
{
    public async Task<(AudioClip Audio, UiOutput Ui)> ExecuteAsync(AudioClip? audio, NodeContext context, IServiceProvider services, string filenamePrefix = "audio/ComfyUI",
        [Input(Options = ["V0", "128k", "320k"])] string quality = "V0") =>
        await AudioResults.SaveAsync(audio, context, services, "output", filenamePrefix, "mp3", quality);
}

/// <summary>Save Audio (Advanced): FLAC, MP3 or Opus, each with its own quality choices.</summary>
[Node(Category = "audio", DisplayName = "Save Audio (Advanced)", IsOutput = true, Description = "Saves the input audio to your ComfyUI output directory.")]
public sealed class SaveAudioAdvanced
{
    public static DynamicOption[] Formats() =>
    [
        new("flac"),
        new("mp3", DynamicInput.Combo("quality", ["V0", "128k", "320k"], "V0")),
        new("opus", DynamicInput.Combo("quality", ["64k", "96k", "128k", "192k", "320k"], "128k")),
    ];

    public async Task<(AudioClip Audio, UiOutput Ui)> ExecuteAsync(
        [Input(Tooltip = "The audio to save.")] AudioClip? audio, NodeContext context, IServiceProvider services,
        [Input(Tooltip = "The prefix for the file to save. May include formatting tokens such as %date:yyyy-MM-dd%.")] string filenamePrefix = "audio/ComfyUI",
        [DynamicCombo(nameof(Formats)), Input(Tooltip = "The file format in which to save the audio.")] DynamicChoice? format = null) =>
        await AudioResults.SaveAsync(audio, context, services, "output", filenamePrefix, format?.Key ?? "flac", format?.Get<string?>("quality", null));
}

/// <summary>Preview Audio: the clip as a temporary file (FLAC, or WAV without SpawnDev.Codecs), shown in the node.</summary>
[Node(Category = "audio", DisplayName = "Preview Audio", IsOutput = true, Description = "Preview the audio without saving it to the ComfyUI output directory.")]
public sealed class PreviewAudio
{
    public async Task<(AudioClip Audio, UiOutput Ui)> ExecuteAsync(AudioClip? audio, NodeContext context, IServiceProvider services) =>
        await AudioResults.SaveAsync(audio, context, services, "temp", "ComfyUI_temp");
}

/// <summary>Audio Info (SpawnGen): the clip's length in seconds, sample rate and channel count.</summary>
[Node(Category = "audio", DisplayName = "Audio Info")]
public sealed class AudioInfo
{
    public (double Seconds, long SampleRate, long Channels) Execute(AudioClip audio) => (audio.Seconds, audio.SampleRate, audio.ChannelCount);
}

static class AudioResults
{
    public static async Task<(AudioClip, UiOutput)> SaveAsync(AudioClip? audio, NodeContext? context, IServiceProvider services, string type, string prefix,
        string format = "flac", string? quality = null)
    {
        if (audio is null) throw new ArgumentException("There is no audio to save.", nameof(audio));
        var (data, extension) = await AudioFiles.EncodeAsync(audio, Results.Metadata(context), format, quality);
        var file = await Results.For(services).SaveAsync(type, prefix, data, extension, trailingUnderscore: false);
        return (audio, new UiOutput("audio", new object[] { file }));
    }
}
