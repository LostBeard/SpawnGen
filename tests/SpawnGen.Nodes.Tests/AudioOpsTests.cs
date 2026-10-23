using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// ComfyUI's model-free audio nodes and comfy/audio.py's resampler and biquads. Expected numbers come from a by-hand port of the Python with float32
/// rounding emulated step by step, plus filter responses that follow from the formulas (shelf gain at DC and Nyquist, peak gain at the centre).
/// </summary>
public class AudioOpsTests
{
    static readonly float[] Five = [0.5f, -0.25f, 1f, 0f, 0.75f];
    static readonly float[] Eight = [0.5f, -0.25f, 0.125f, 0f, 0.25f, -0.5f, 0f, 0f];

    static AudioClip Clip(int rate, params float[][] channels) => new(channels, rate);

    static NodeSchema Schema(string id) => new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).GetSchema(id)!;

    // --- comfy.audio.resample -----------------------------------------------------------------------------

    [Test]
    public void Resample_matches_the_torch_sinc_kernels_and_output_lengths()
    {
        // 8000 -> 12000 reduces to 2 -> 3: width 7, 16 taps, ceil(float32(3 * 5 / 2)) = 8 samples.
        Assert.That(AudioDsp.Resample(Five, 8000, 12000),
            Is.EqualTo(new[] { 0.4831861f, -0.2038168f, 0.1609523f, 0.9782393f, 0.3393545f, 0.0907070f, 0.7323998f, 0.3793453f }).Within(1e-6f));
        Assert.That(AudioDsp.Resample(Five, 12000, 8000), Is.EqualTo(new[] { 0.1754282f, 0.3408565f, 0.5009372f, 0.3033165f }).Within(1e-6f));
        Assert.That(AudioDsp.Resample(Five, 8000, 16000),
            Is.EqualTo(new[] { 0.4831861f, -0.0507637f, -0.2296843f, 0.4391955f, 0.9782393f, 0.5755001f, 0.0207684f, 0.2505942f, 0.7323998f, 0.5641246f }).Within(1e-6f));
        Assert.That(AudioDsp.Resample(Five, 8000, 8000), Is.SameAs(Five));
        Assert.That(AudioDsp.Resample(new float[44100], 44100, 48000).Length, Is.EqualTo(48000));
        Assert.That(AudioDsp.Resample([], 8000, 12000), Is.Empty);
    }

    [Test]
    public void A_resampled_tone_keeps_its_shape()
    {
        // 440 Hz at 22050 Hz up to 44100 Hz: away from the edges the samples lie on the same sine (the rolloff passes 440 Hz untouched).
        var tone = Enumerable.Range(0, 2205).Select(i => 0.5f * MathF.Sin(2 * MathF.PI * 440 * i / 22050f)).ToArray();
        var up = AudioDsp.Resample(tone, 22050, 44100);
        Assert.That(up.Length, Is.EqualTo(4410));
        for (int i = 200; i < 4200; i++) Assert.That(up[i], Is.EqualTo(0.5f * MathF.Sin(2 * MathF.PI * 440 * i / 44100f)).Within(2e-3f), $"sample {i}");
    }

    // --- biquads -----------------------------------------------------------------------------------------

    [Test]
    public void The_three_biquads_match_scipy_lfilter_in_float32()
    {
        Assert.That(AudioDsp.Bass(Eight, 8000, 6, 100),
            Is.EqualTo(new[] { 0.5097234f, -0.2352852f, 0.1374258f, 0.0148914f, 0.2697652f, -0.4850828f, 0.0050588f, 0.0047445f }).Within(1e-6f));
        Assert.That(AudioDsp.Equalizer(Eight, 8000, 1000, -3, 2),
            Is.EqualTo(new[] { 0.4746466f, -0.2669532f, 0.1407519f, 0.0167985f, 0.2530114f, -0.4820925f, 0.0316391f, -0.0000669f }).Within(1e-6f));
        // +12 dB of treble pushes several samples past full scale: _biquad clamps them to -1..1.
        Assert.That(AudioDsp.Treble(Eight, 8000, 12, 2000),
            Is.EqualTo(new[] { 0.9976312f, -1f, 0.9641981f, -0.3249282f, 0.4855766f, -1f, 0.9286180f, -0.3047722f }).Within(1e-6f));
    }

    [Test]
    public void Shelves_and_the_peak_reach_their_gain_where_they_should()
    {
        // Low shelf +6 dB: DC is multiplied by 10^(6/20).
        var dc = AudioDsp.Bass(Enumerable.Repeat(0.25f, 8000).ToArray(), 8000, 6, 100);
        Assert.That(dc[^1], Is.EqualTo(0.25 * Math.Pow(10, 6 / 20.0)).Within(1e-4));
        // High shelf +12 dB: the Nyquist frequency (+-0.1 alternating) is multiplied by 10^(12/20).
        var nyquist = AudioDsp.Treble(Enumerable.Range(0, 8000).Select(i => i % 2 == 0 ? 0.1f : -0.1f).ToArray(), 8000, 12, 2000);
        Assert.That(Math.Abs(nyquist[^1]), Is.EqualTo(0.1 * Math.Pow(10, 12 / 20.0)).Within(1e-4));
        // Peak -6 dB at 1 kHz: a 1 kHz tone comes out at 10^(-6/20) of its RMS.
        var tone = Enumerable.Range(0, 8000).Select(i => 0.5f * MathF.Sin(MathF.PI * i / 4)).ToArray();
        var cut = AudioDsp.Equalizer(tone, 8000, 1000, -6, 0.707);
        double Rms(float[] s) => Math.Sqrt(s[^800..].Average(v => (double)v * v));
        Assert.That(Rms(cut) / Rms(tone), Is.EqualTo(Math.Pow(10, -6 / 20.0)).Within(1e-3));
    }

    [Test]
    public void Equalizer_3_band_runs_bass_mids_treble_and_skips_zero_gains()
    {
        var clip = Clip(8000, Eight, [.. Eight.Select(v => -v)]);
        var eq = new AudioEqualizer3Band().Execute(clip, 6, 100, -3, 1000, 2, 12, 2000)!;
        Assert.That(eq.Channels[0], Is.EqualTo(new[] { 0.9654615f, -1f, 1f, -0.3091328f, 0.5003287f, -1f, 0.9634058f, -0.3330343f }).Within(1e-6f));
        Assert.That(eq.SampleRate, Is.EqualTo(8000));
        // All gains 0: an untouched copy.
        var flat = new AudioEqualizer3Band().Execute(clip)!;
        Assert.That(flat.Channels[1], Is.EqualTo(clip.Channels[1]));
        Assert.That(flat.Channels[1], Is.Not.SameAs(clip.Channels[1]));
        // Only the mids: the same as the peaking filter alone.
        Assert.That(new AudioEqualizer3Band().Execute(clip, 0, 100, -3, 1000, 2, 0, 2000)!.Channels[0], Is.EqualTo(AudioDsp.Equalizer(Eight, 8000, 1000, -3, 2)));
        Assert.That(new AudioEqualizer3Band().Execute(null), Is.Null);
    }

    // --- the nodes -----------------------------------------------------------------------------------------

    [Test]
    public void Trim_rounds_half_to_even_counts_negative_starts_from_the_end_and_rejects_empty_ranges()
    {
        var clip = Clip(10, [.. Enumerable.Range(0, 20).Select(i => (float)i)]);
        Assert.That(new TrimAudioDuration().Execute(clip, 0.5, 1.0)!.Channels[0], Is.EqualTo(Enumerable.Range(5, 10).Select(i => (float)i)));
        Assert.That(new TrimAudioDuration().Execute(clip, -0.5, 60)!.Channels[0], Is.EqualTo(new[] { 15f, 16f, 17f, 18f, 19f }));
        // 0.25 s * 10 = 2.5 rounds to 2, 0.35 * 10 = 3.5 to 4.
        Assert.That(new TrimAudioDuration().Execute(clip, 0.25, 0.35)!.Channels[0], Is.EqualTo(new[] { 2f, 3f, 4f, 5f }));
        // A start before the beginning is clamped to 0; past the end is an error, as is a zero duration.
        Assert.That(new TrimAudioDuration().Execute(clip, -100, 0.3)!.Channels[0], Is.EqualTo(new[] { 0f, 1f, 2f }));
        Assert.That(() => new TrimAudioDuration().Execute(clip, 5, 1), Throws.ArgumentException);
        Assert.That(() => new TrimAudioDuration().Execute(clip, 0, 0), Throws.ArgumentException);
        var empty = Clip(10, new float[0]);
        Assert.That(new TrimAudioDuration().Execute(empty, 1, 1), Is.SameAs(empty));
        Assert.That(new TrimAudioDuration().Execute(null), Is.Null);
    }

    [Test]
    public void Split_and_join_channels()
    {
        float[] l = [1, 2, 3], r = [4, 5, 6];
        var (left, right) = new SplitAudioChannels().Execute(Clip(8000, l, r));
        Assert.That((left!.ChannelCount, left.Channels[0], right!.Channels[0], right.SampleRate), Is.EqualTo((1, l, r, 8000)));
        Assert.That(() => new SplitAudioChannels().Execute(Clip(8000, l)), Throws.ArgumentException.With.Message.Contains("got 1 channel(s)"));
        Assert.That(new SplitAudioChannels().Execute(null), Is.EqualTo(((AudioClip?)null, (AudioClip?)null)));

        // Join trims to the shorter channel.
        var (joined, ui) = new JoinAudioChannels().Execute(Clip(8000, l), Clip(8000, [7, 8]));
        Assert.That((joined!.ChannelCount, joined.Channels[0], joined.Channels[1], ui), Is.EqualTo((2, new[] { 1f, 2f }, new[] { 7f, 8f }, (UiOutput?)null)));
        // Different rates: the lower one is resampled up first (3 samples at 8 kHz become 5 at 12 kHz, then both are cut to 5).
        var mixed = new JoinAudioChannels().Execute(Clip(8000, Five[..3]), Clip(12000, Five)).Audio!;
        Assert.That((mixed.SampleRate, mixed.Length), Is.EqualTo((12000, 5)));
        Assert.That(mixed.Channels[0], Is.EqualTo(AudioDsp.Resample(Five[..3], 8000, 12000)));
        Assert.That(() => new JoinAudioChannels().Execute(Clip(8000, l, r), Clip(8000, l)), Throws.ArgumentException);
        var only = Clip(8000, l);
        Assert.That(new JoinAudioChannels().Execute(null, only).Audio, Is.SameAs(only));
    }

    [Test]
    public void Concat_makes_mono_stereo_matches_rates_and_honours_the_direction()
    {
        var mono = Clip(8000, [1, 2]);
        var stereo = Clip(8000, [3], [4]);
        var after = new AudioConcat().Execute(mono, stereo, "after")!;
        Assert.That((after.Channels[0], after.Channels[1]), Is.EqualTo((new[] { 1f, 2f, 3f }, new[] { 1f, 2f, 4f })));
        var before = new AudioConcat().Execute(mono, stereo, "before")!;
        Assert.That((before.Channels[0], before.Channels[1]), Is.EqualTo((new[] { 3f, 1f, 2f }, new[] { 4f, 1f, 2f })));
        var rates = new AudioConcat().Execute(Clip(8000, Five), Clip(16000, [0.5f]))!;
        Assert.That((rates.SampleRate, rates.Length), Is.EqualTo((16000, 11)));
        Assert.That(rates.Channels[1][..10], Is.EqualTo(AudioDsp.Resample(Five, 8000, 16000)));
        Assert.That(new AudioConcat().Execute(null, mono), Is.SameAs(mono));
        Assert.That(() => new AudioConcat().Execute(Clip(8000, [1], [2], [3]), stereo), Throws.ArgumentException);
    }

    [Test]
    public void Merge_fits_audio2_to_audio1_combines_and_normalises_peaks_above_one()
    {
        var a = Clip(8000, [0.5f, 0.25f, -0.5f]);
        // audio2 shorter: padded with silence.
        Assert.That(new AudioMerge().Execute(a, Clip(8000, [0.25f]), "add")!.Channels[0], Is.EqualTo(new[] { 0.75f, 0.25f, -0.5f }));
        Assert.That(new AudioMerge().Execute(a, Clip(8000, [0.25f, 0.25f, 0.25f, 9f]), "subtract")!.Channels[0], Is.EqualTo(new[] { 0.25f, 0f, -0.75f }));
        Assert.That(new AudioMerge().Execute(a, Clip(8000, [0.5f, 0.5f, 0.5f]), "mean")!.Channels[0], Is.EqualTo(new[] { 0.5f, 0.375f, 0f }));
        Assert.That(new AudioMerge().Execute(a, Clip(8000, [2f, 2f]), "multiply")!.Channels[0], Is.EqualTo(new[] { 1f, 0.5f, 0f }), "a peak of exactly 1 is kept");
        // 0.5 + 1 = 1.5 is the loudest: everything is divided by 1.5.
        var loud = new AudioMerge().Execute(a, Clip(8000, [1f, 0.5f, -0.25f]), "add")!;
        Assert.That(loud.Channels[0], Is.EqualTo(new[] { 1f, 0.75f / 1.5f, -0.75f / 1.5f }).Within(1e-7f));
        // A mono clip goes with each channel of a stereo one.
        var both = new AudioMerge().Execute(Clip(8000, [0.25f, 0.25f]), Clip(8000, [0.5f, 0f], [0f, 0.5f]), "add")!;
        Assert.That((both.ChannelCount, both.Channels[0], both.Channels[1]), Is.EqualTo((2, new[] { 0.75f, 0.25f }, new[] { 0.25f, 0.75f })));
        // Rates: audio1 at the lower rate is resampled up, and its new length is the one audio2 is fitted to.
        var up = new AudioMerge().Execute(Clip(8000, Five), Clip(12000, [0f]), "add")!;
        Assert.That((up.SampleRate, up.Channels[0]), Is.EqualTo((12000, AudioDsp.Resample(Five, 8000, 12000))));
        var silent = Clip(8000, new float[0]);
        Assert.That(new AudioMerge().Execute(silent, a, "add"), Is.SameAs(silent));
        Assert.That(() => new AudioMerge().Execute(a, a, "divide"), Throws.ArgumentException);
    }

    [Test]
    public void Volume_is_in_decibels()
    {
        var clip = Clip(8000, [0.5f, -0.1f]);
        Assert.That(new AudioAdjustVolume().Execute(clip, 0), Is.SameAs(clip));
        Assert.That(new AudioAdjustVolume().Execute(clip, -20)!.Channels[0], Is.EqualTo(new[] { 0.05f, -0.01f }).Within(1e-8f));
        // The default +1 dB, and +6 dB is (almost) double; nothing is clipped.
        Assert.That(new AudioAdjustVolume().Execute(clip)!.Channels[0][0], Is.EqualTo(0.5f * (float)Math.Pow(10, 0.05)));
        Assert.That(new AudioAdjustVolume().Execute(clip, 6)!.Channels[0][0], Is.EqualTo(0.5f * 1.9952623f).Within(1e-7f));
        Assert.That(new AudioAdjustVolume().Execute(Clip(8000, [0.9f]), 100)!.Channels[0][0], Is.EqualTo(90000f).Within(0.01f));
    }

    [Test]
    public void Empty_audio_is_silence_of_the_rounded_length()
    {
        var clip = new EmptyAudio().Execute(0.5, 44100, 1);
        Assert.That((clip.ChannelCount, clip.Length, clip.SampleRate), Is.EqualTo((1, 22050, 44100)));
        Assert.That(clip.Channels[0].All(v => v == 0), Is.True);
        Assert.That(new EmptyAudio().Execute().Length, Is.EqualTo(60 * 44100));
        Assert.That(new EmptyAudio().Execute().ChannelCount, Is.EqualTo(2));
        Assert.That(new EmptyAudio().Execute(2.5, 1).Length, Is.EqualTo(2), "round half to even");
        Assert.That(new EmptyAudio().Execute(0, 8000).Length, Is.EqualTo(0));
    }

    [Test]
    public async Task Record_audio_reads_the_uploaded_recording_by_its_annotated_name()
    {
        var store = new MemoryInputStore();
        var saved = InputFiles.Store;
        InputFiles.Store = store;
        try
        {
            await store.SaveAsync("recording-1700000000000.wav", WavFile.Encode(Clip(48000, [0.5f, -0.5f])));
            var clip = await new RecordAudio().ExecuteAsync("audio/recording-1700000000000.wav [temp]");
            Assert.That((clip.SampleRate, clip.Length), Is.EqualTo((48000, 2)));
            Assert.That(clip.Channels[0], Is.EqualTo(new[] { 0.5f, -0.5f }));
            Assert.That((await new RecordAudio().ExecuteAsync("recording-1700000000000.wav")).Length, Is.EqualTo(2));
            Assert.That(() => new RecordAudio().ExecuteAsync(""), Throws.ArgumentException);
        }
        finally { InputFiles.Store = saved; }
    }

    [Test]
    public async Task Save_and_preview_pass_the_clip_on()
    {
        var clip = Clip(8000, [0.5f, 0.25f]);
        var store = new MemoryResultStore();
        var services = new Services(store);
        var (saved, ui) = await new SaveAudio().ExecuteAsync(clip, services);
        Assert.That(saved, Is.SameAs(clip));
        var file = (ResultFile)((object[])ui["audio"]!)[0];
        Assert.That((file.Subfolder, file.Filename, file.Type), Is.EqualTo(("audio", "ComfyUI_00001_.wav", "output")));
        Assert.That(WavFile.Decode(store.Get("output", file.Filename, "audio")!).Channels[0], Is.EqualTo(clip.Channels[0]));
        var (previewed, pui) = await new PreviewAudio().ExecuteAsync(clip, services);
        Assert.That((previewed, ((ResultFile)((object[])pui["audio"]!)[0]).Type), Is.EqualTo((clip, "temp")));
        Assert.That(() => new SaveAudio().ExecuteAsync(null, services), Throws.ArgumentException);
    }

    [Test]
    public void Schemas_match_ComfyUI()
    {
        string[] Inputs(string id) => [.. Schema(id).Inputs.Select(i => i.Name)];
        string[] Outputs(string id) => [.. Schema(id).Outputs.Select(o => o.Name + ":" + o.Type.Name)];

        Assert.That(Inputs("AudioEqualizer3Band"), Is.EqualTo(new[] { "audio", "low_gain_dB", "low_freq", "mid_gain_dB", "mid_freq", "mid_q", "high_gain_dB", "high_freq" }));
        Assert.That(Schema("AudioEqualizer3Band").Inputs.Skip(1).Select(i => i.Widget!.Default), Is.EqualTo(new object[] { 0.0, 100L, 0.0, 1000L, 0.707, 0.0, 5000L }));
        Assert.That(Inputs("TrimAudioDuration"), Is.EqualTo(new[] { "audio", "start_index", "duration" }));
        Assert.That(Inputs("EmptyAudio"), Is.EqualTo(new[] { "duration", "sample_rate", "channels" }));
        Assert.That(Schema("EmptyAudio").Inputs.Select(i => i.Widget!.Default), Is.EqualTo(new object[] { 60.0, 44100L, 2L }));
        Assert.That(Inputs("JoinAudioChannels"), Is.EqualTo(new[] { "audio_left", "audio_right" }));
        Assert.That(Inputs("AudioConcat"), Is.EqualTo(new[] { "audio1", "audio2", "direction" }));
        Assert.That(Schema("AudioConcat").FindInput("direction")!.Widget!.ResolveOptions(), Is.EqualTo(new[] { "after", "before" }));
        Assert.That(Inputs("AudioMerge"), Is.EqualTo(new[] { "audio1", "audio2", "merge_method" }));
        Assert.That(Schema("AudioMerge").FindInput("merge_method")!.Widget!.ResolveOptions(), Is.EqualTo(new[] { "add", "mean", "subtract", "multiply" }));
        Assert.That(Schema("AudioAdjustVolume").FindInput("volume")!.Widget, Has.Property("Default").EqualTo(1L).And.Property("Min").EqualTo(-100.0).And.Property("Max").EqualTo(100.0));
        Assert.That(Inputs("RecordAudio"), Is.EqualTo(new[] { "audio" }));
        Assert.That(Inputs("SaveAudio"), Is.EqualTo(new[] { "audio", "filename_prefix" }));
        Assert.That(Schema("SaveAudio").FindInput("filename_prefix")!.Widget!.Default, Is.EqualTo("audio/ComfyUI"));

        Assert.That(Outputs("SplitAudioChannels"), Is.EqualTo(new[] { "left:AUDIO", "right:AUDIO" }));
        Assert.That(Outputs("JoinAudioChannels"), Is.EqualTo(new[] { "audio:AUDIO" }));
        Assert.That(Outputs("SaveAudio"), Is.EqualTo(new[] { "audio:AUDIO" }));
        Assert.That(Outputs("PreviewAudio"), Is.EqualTo(new[] { "audio:AUDIO" }));
        foreach (var id in (string[])["AudioAdjustVolume", "AudioConcat", "AudioMerge", "EmptyAudio", "TrimAudioDuration", "AudioEqualizer3Band", "RecordAudio", "LoadAudio"])
        {
            Assert.That(Outputs(id), Is.EqualTo(new[] { "AUDIO:AUDIO" }), id);
            Assert.That(Schema(id).Category, Is.EqualTo("audio"), id);
        }
        Assert.That(Schema("AudioEqualizer3Band").DisplayName, Is.EqualTo("Audio Equalizer (3-Band)"));
        Assert.That(Schema("AudioConcat").DisplayName, Is.EqualTo("Concatenate Audio"));
        Assert.That(Schema("AudioMerge").DisplayName, Is.EqualTo("Merge Audio"));
        Assert.That(Schema("AudioAdjustVolume").DisplayName, Is.EqualTo("Adjust Audio Volume"));
    }

    sealed class Services(params object[] instances) : IServiceProvider
    {
        public object? GetService(Type serviceType) => instances.FirstOrDefault(serviceType.IsInstanceOfType);
    }
}
