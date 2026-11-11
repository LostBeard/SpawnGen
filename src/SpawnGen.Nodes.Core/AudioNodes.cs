using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entries and SpawnGen's tests. SpawnGen reads and writes WAV (WavFile); ComfyUI's
// Save Audio writes FLAC.

/// <summary>Load Audio: a WAV file from the input folder (an " [input]" annotation is dropped).</summary>
[Node(Category = "audio", DisplayName = "Load Audio")]
public sealed class LoadAudio
{
    public static string[] AudioNames() => [.. InputFiles.Store.List(n => n.EndsWith(".wav", StringComparison.OrdinalIgnoreCase))];

    public async Task<AudioClip> ExecuteAsync([Input(OptionsProvider = nameof(AudioNames))] string audio) =>
        WavFile.Decode(await InputFiles.Store.ReadAsync(RecordAudio.StripAnnotation(audio)));
}

/// <summary>Save Audio: the clip as a numbered 16-bit WAV in the output folder (prefix "audio/ComfyUI" gives audio/ComfyUI_00001.wav), shown in the node.</summary>
[Node(Category = "audio", DisplayName = "Save Audio (FLAC) (DEPRECATED)", IsOutput = true)]
public sealed class SaveAudio
{
    public async Task<(AudioClip Audio, UiOutput Ui)> ExecuteAsync(AudioClip? audio, IServiceProvider services, string filenamePrefix = "audio/ComfyUI") =>
        await AudioResults.SaveAsync(audio, services, "output", filenamePrefix);
}

/// <summary>Preview Audio: the clip as a temporary WAV, shown in the node.</summary>
[Node(Category = "audio", DisplayName = "Preview Audio", IsOutput = true, Description = "Preview the audio without saving it to the ComfyUI output directory.")]
public sealed class PreviewAudio
{
    public async Task<(AudioClip Audio, UiOutput Ui)> ExecuteAsync(AudioClip? audio, IServiceProvider services) =>
        await AudioResults.SaveAsync(audio, services, "temp", "ComfyUI_temp");
}

/// <summary>Audio Info (SpawnGen): the clip's length in seconds, sample rate and channel count.</summary>
[Node(Category = "audio", DisplayName = "Audio Info")]
public sealed class AudioInfo
{
    public (double Seconds, long SampleRate, long Channels) Execute(AudioClip audio) => (audio.Seconds, audio.SampleRate, audio.ChannelCount);
}

static class AudioResults
{
    public static async Task<(AudioClip, UiOutput)> SaveAsync(AudioClip? audio, IServiceProvider services, string type, string prefix)
    {
        if (audio is null) throw new ArgumentException("There is no audio to save.", nameof(audio));
        var file = await Results.For(services).SaveAsync(type, prefix, WavFile.Encode(audio), "wav", trailingUnderscore: false);
        return (audio, new UiOutput("audio", new object[] { file }));
    }
}
