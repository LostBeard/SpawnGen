using System.Text.Json;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;
using SpawnDev.ILGPU.ML.AceStep;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// ACE-Step 1.5 against ComfyUI (bef11fc, CPU float32; Fixtures/acestep, the dumps sent to SpawnDev.ILGPU.ML on 2026-11-17): ComfyUI's
/// audio_ace_step_1_5_split template (turbo DiT, DualCLIPLoader ace, ace_1.5_vae, ModelSamplingAuraFlow 3, KSampler euler / simple cfg 1) cut to
/// 10 s and 2 steps, seed 31, with the template's own prompt.
/// </summary>
public class AceStepTests
{
    static string Dir => Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "acestep");

    static float[] Raw(string name)
    {
        var b = File.ReadAllBytes(Path.Combine(Dir, name));
        var d = new float[b.Length / 4];
        Buffer.BlockCopy(b, 0, d, 0, b.Length);
        return d;
    }

    static float[] Sigmas(string name) =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, name))).RootElement.GetProperty("sigmas").EnumerateArray().Select(v => (float)v.GetDouble()).ToArray();

    static int[] Ids(string file, int call) =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, file))).RootElement.GetProperty("calls")[call].GetProperty("ids").EnumerateArray().Select(v => v.GetInt32()).ToArray();

    static JsonElement Prompt => JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "ace_template_prompt.json"))).RootElement;

    static int[] Codes => JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "ace_codes_seed31.json"))).RootElement.GetProperty("codes").EnumerateArray().Select(v => v.GetInt32()).ToArray();

    static double RelRms(float[] ours, float[] theirs)
    {
        double num = 0, den = 0;
        for (int i = 0; i < ours.Length; i++) { double d = ours[i] - theirs[i]; num += d * d; den += (double)theirs[i] * theirs[i]; }
        return Math.Sqrt(num / den);
    }

    sealed class Services(params object[] instances) : IServiceProvider
    {
        public object? GetService(Type serviceType) => instances.FirstOrDefault(serviceType.IsInstanceOfType);
    }

    static NodeContext Context(Services services, string classType) =>
        (NodeContext)Activator.CreateInstance(typeof(NodeContext), System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance, null,
            ["1", classType, services, CancellationToken.None, null, new Lazy<string>(() => "{}"), new Dictionary<string, string>(), null, null, 0], null)!;

    [Test]
    public void The_empty_latent_has_ComfyUIs_frame_count()
    {
        // Measured: Python's round (half to even) of seconds x 25.
        foreach (var (seconds, frames) in new[] { (1.0, 25), (10.0, 250), (10.01, 250), (10.02, 250), (10.03, 251), (10.05, 251), (10.06, 252), (7.77, 194), (1000.0, 25000) })
            Assert.That(EmptyAceStep15LatentAudio.Frames(seconds), Is.EqualTo(frames), $"{seconds} s");
        var latent = new EmptyAceStep15LatentAudio().Execute(10.0, 2);
        Assert.That(latent.Shape, Is.EqualTo(new[] { 2, 64, 250 }));
        Assert.That((latent.IsAudio, latent.Height, latent.Width, latent.ItemSize), Is.EqualTo((true, 1, 250, 64 * 250)));
        Assert.That(LatentBatch.Shaped(latent, 1).Shape, Is.EqualTo(new[] { 1, 64, 250 }));
    }

    [Test]
    public void The_models_own_schedule_and_AuraFlow_3_are_ComfyUIs()
    {
        var model = new ModelHandle(new AceStepDiffusionModel(null!, "acestep_v1.5_turbo.safetensors", "", ""));
        Assert.That(new BasicScheduler().Execute(model, "simple", 8, 1.0).Values, Is.EqualTo(Sigmas("ace_meas_sigmas_default_simple8.json")).Within(2e-6));
        var aura = new ModelSamplingAuraFlow().Execute(model, 3.0);
        Assert.That(new BasicScheduler().Execute(aura, "simple", 8, 1.0).Values, Is.EqualTo(Sigmas("ace_sigmas_simple8_auraflow3.json")).Within(2e-6));
    }

    [Test]
    public void Zero_out_clears_the_lyrics_and_keeps_the_codes()
    {
        // Measured: ConditioningZeroOut zeroes the caption and conditioning_lyrics, audio_codes unchanged.
        var entry = new CondEntry([1f, 2f], [1, 2, 1]) { Lyrics = new HostTensor([1, 1, 2], [3f, 4f]), AudioCodes = [5, 6] };
        var zero = new ConditioningZeroOut().Execute(new Conditioning([entry])).Entries[0];
        Assert.That(zero.Embeddings, Is.All.EqualTo(0f));
        Assert.That(zero.Lyrics!.Data, Is.All.EqualTo(0f));
        Assert.That(zero.Lyrics.Shape, Is.EqualTo(new[] { 1, 1, 2 }));
        Assert.That(zero.AudioCodes, Is.EqualTo(new[] { 5, 6 }));
    }

    [Test]
    public void The_loaders_recognise_the_ACE_Step_files_by_their_tensors()
    {
        Assert.That(AceStepDiffusionModel.Is(["decoder.condition_embedder.weight", "decoder.layers.0.x", "detokenizer.embed_tokens"]), Is.True);
        Assert.That(AceStepDiffusionModel.Is(["decoder.layers.0.x", "encoder.layers.0.x"]), Is.False);
        Assert.That(AceStepVaeModel.Is(["encoder.layers.0.weight", "decoder.layers.0.weight"]), Is.True);
        Assert.That(AceStepVaeModel.Is(["encoder.conv_in.weight", "decoder.conv_in.weight"]), Is.False, "an SD VAE");
    }

    [Test]
    public void A_picture_node_refuses_an_audio_latent()
    {
        var registry = new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(KSampler).Assembly);
        var executor = new SpawnGen.Engine.Execution.PromptExecutor(registry);
        var g = new SpawnGen.Engine.Graph.PromptGraph();
        g.Add("1", "EmptyAceStep1.5LatentAudio", ("seconds", 1.0), ("batch_size", 1L));
        g.Add("2", "LatentUpscaleBy", ("samples", SpawnGen.Engine.Graph.PromptInput.Link("1")), ("upscale_method", "nearest-exact"), ("scale_by", 2.0));
        g.Add("3", "PreviewAny", ("source", SpawnGen.Engine.Graph.PromptInput.Link("2")));
        var e = Assert.ThrowsAsync<SpawnGen.Engine.Execution.NodeExecutionException>(() => executor.ExecuteAsync(g));
        Assert.That(e!.InnerException, Is.InstanceOf<NotSupportedException>());
        Assert.That(e.InnerException!.Message, Does.Contain("audio latents"));
    }

    static string? Root()
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS_ROOT");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS_ROOT is not set.");
        ModelFolders.Scan(root!);
        return root;
    }

    const string Needs = "Needs SPAWNGEN_MODELS_ROOT with diffusion_models/acestep_v1.5_turbo.safetensors, text_encoders/qwen_0.6b_ace15.safetensors and "
        + "qwen_1.7b_ace15.safetensors, vae/ace_1.5_vae.safetensors, a GPU, and ACE-Step/Ace-Step1.5's tokenizers and silence latent (downloaded on first use).";

    static AceStepEncodeSettings Settings(double duration = 10, string? lyrics = null, string? language = null, bool codes = false)
    {
        var p = Prompt;
        return new(p.GetProperty("tags").GetString()!, lyrics ?? p.GetProperty("lyrics").GetString()!, 31, p.GetProperty("bpm").GetInt32(), duration,
            p.GetProperty("timesignature").GetString()!, language ?? p.GetProperty("language").GetString()!, p.GetProperty("keyscale").GetString()!,
            codes, p.GetProperty("cfg_scale").GetDouble(), p.GetProperty("temperature").GetDouble(), p.GetProperty("top_p").GetDouble(), 0, 0);
    }

    static async Task<(MlRuntime Runtime, Services Services, ClipHandle Clip)> ClipAsync()
    {
        var root = Root();
        var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var services = new Services(runtime);
        var clip = await new DualCLIPLoader().ExecuteAsync("qwen_0.6b_ace15.safetensors", "qwen_1.7b_ace15.safetensors", "ace", Context(services, "DualCLIPLoader"), services);
        return (runtime, services, clip);
    }

    /// <summary>The ids ComfyUI fed its embedding tables (caption, lyrics, LM prompt, LM negative) on four measured runs.</summary>
    [Test, Explicit(Needs)]
    public async Task The_prompt_ids_are_ComfyUIs()
    {
        var (runtime, _, clip) = await ClipAsync();
        using var _r = runtime;
        var model = (AceStepTextModel)clip.Encoder;
        foreach (var (file, settings) in new[]
        {
            ("ace_tmpl_d10_codes1_ids.json", Settings(10, codes: true)),
            ("ace_tmpl_d120_codes1_ids.json", Settings(120, codes: true)),
            ("ace_tmpl_short_codes1_ids.json", Settings(10, "[Verse]\nhello world", codes: true)),
            ("ace_tmpl_unknownlang_codes1_ids.json", Settings(10, language: "unknown", codes: true)),
        })
        {
            var (caption, lyrics, pos, neg) = await model.TokenizeAsync(settings, CancellationToken.None);
            Assert.That(caption, Is.EqualTo(Ids(file, 0)), file + " caption");
            Assert.That(lyrics, Is.EqualTo(Ids(file, 1)), file + " lyrics");
            Assert.That(pos, Is.EqualTo(Ids(file, 2)), file + " LM prompt");
            // ComfyUI left-pads the negative to the positive's length with <|endoftext|>; the code LM strips that padding (AceStepCodeLm.StripLeftPadding).
            Assert.That(AceStepCodeLm.StripLeftPadding(neg!), Is.EqualTo(AceStepCodeLm.StripLeftPadding(Ids(file, 3))), file + " LM negative");
        }
    }

    /// <summary>The node, codes off: the caption's hidden states and the lyric embeddings against ComfyUI's conditioning.</summary>
    [Test, Explicit(Needs)]
    public async Task The_conditioning_is_ComfyUIs()
    {
        var (runtime, services, clip) = await ClipAsync();
        using var _r = runtime;
        var s = Settings();
        var entry = (await new TextEncodeAceStepAudio15().ExecuteAsync(clip, Context(services, "TextEncodeAceStepAudio1.5"), s.Tags, s.Lyrics, 31, s.Bpm, 10.0,
            s.TimeSignature, s.Language, s.KeyScale, false, s.CfgScale, s.Temperature, s.TopP, 0, 0)).Entries.Single();
        Assert.That(entry.Shape, Is.EqualTo(new[] { 1, 127, 1024 }));
        Assert.That(entry.Lyrics!.Shape, Is.EqualTo(new[] { 1, 350, 1024 }));
        Assert.That(entry.AudioCodes, Is.Null);
        double caption = RelRms(entry.Embeddings, Raw("ace_te_codes0_cond_0_cond_f32_1x127x1024.raw")),
            lyrics = RelRms(entry.Lyrics.Data, Raw("ace_te_codes0_cond_0_conditioning_lyrics_f32_1x350x1024.raw"));
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: caption {caption:E2}, lyrics {lyrics:E2} relative RMS vs ComfyUI");
        Assert.That(caption, Is.LessThan(1e-3));
        Assert.That(lyrics, Is.LessThan(1e-5));
    }

    /// <summary>
    /// The model's calls as ComfyUI made them (codes off: calls 0 and 1; codes on: call 0), with ComfyUI's silence table as the reference (a test
    /// fixture from its run: the shipped path uses ACE-Step's MIT silence latent).
    /// </summary>
    [Test, Explicit(Needs)]
    public async Task The_model_calls_give_ComfyUIs_denoised_estimate()
    {
        var (runtime, services, clip) = await ClipAsync();
        using var _r = runtime;
        var s = Settings();
        var entry = (await new TextEncodeAceStepAudio15().ExecuteAsync(clip, Context(services, "TextEncodeAceStepAudio1.5"), s.Tags, s.Lyrics, 31, s.Bpm, 10.0,
            s.TimeSignature, s.Language, s.KeyScale, false, s.CfgScale, s.Temperature, s.TopP, 0, 0)).Entries.Single();
        var handle = await new UNETLoader().ExecuteAsync("acestep_v1.5_turbo.safetensors", "default", Context(services, "UNETLoader"), services);
        var model = (AceStepDiffusionModel)handle.Model;
        model.SilenceOverride = (Raw("ace_g0_call_0_c_refer_audio_f32_1x64x250.raw"), 250);
        using var run = model.BeginRun(runtime.Accelerator);
        var report = new List<string>();
        foreach (var (name, call, codes) in new[] { ("ace_g0", 0, false), ("ace_g0", 1, false), ("ace_g1", 0, true) })
        {
            var e = codes ? entry with { AudioCodes = Codes } : entry;
            var x = Raw($"{name}_call_{call}_input_f32_1x64x250.raw");
            float sigma = call == 0 ? 1f : Raw($"ace_g0_call_{call}_timestep_f32_1.raw")[0];
            var v = await run.PredictAsync(x, 1, 250, sigma, e, null, UNetOptions.None, null, CancellationToken.None);
            double err = RelRms([.. x.Select((xi, i) => xi - sigma * v[i])], Raw($"{name}_call_{call}_output_f32_1x64x250.raw"));
            report.Add($"{name} call {call} (sigma {sigma}{(codes ? ", codes" : "")}): {err:E2}");
            Assert.That(err, Is.LessThan(1e-3), string.Join("; ", report));
        }
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: " + string.Join("; ", report));
    }

    /// <summary>
    /// The template's graph (10 s, 2 steps) with ComfyUI's silence table as the reference: codes off and codes on (ComfyUI's 50 codes put on the
    /// conditioning), the final latent and, codes off, the decoded audio against ComfyUI's.
    /// </summary>
    [Test, Explicit(Needs)]
    public async Task The_template_graph_matches_ComfyUI_with_its_silence_table()
    {
        var (runtime, services, clip) = await ClipAsync();
        using var _r = runtime;
        NodeContext Ctx(string type) => Context(services, type);
        var s = Settings();
        var positive = await new TextEncodeAceStepAudio15().ExecuteAsync(clip, Ctx("TextEncodeAceStepAudio1.5"), s.Tags, s.Lyrics, 31, s.Bpm, 10.0,
            s.TimeSignature, s.Language, s.KeyScale, false, s.CfgScale, s.Temperature, s.TopP, 0, 0);
        var negative = new ConditioningZeroOut().Execute(positive);
        var unet = await new UNETLoader().ExecuteAsync("acestep_v1.5_turbo.safetensors", "default", Ctx("UNETLoader"), services);
        ((AceStepDiffusionModel)unet.Model).SilenceOverride = (Raw("ace_g0_call_0_c_refer_audio_f32_1x64x250.raw"), 250);
        var model = new ModelSamplingAuraFlow().Execute(unet, 3.0);
        var empty = new EmptyAceStep15LatentAudio().Execute(10.0, 1);
        var report = new List<string>();
        var latent = await new KSampler().ExecuteAsync(model, positive, negative, empty, Ctx("KSampler"), services, 31, 2, 1.0, "euler", "simple", 1.0);
        Assert.That(latent.Shape, Is.EqualTo(new[] { 1, 64, 250 }));
        double off = RelRms(latent.Data, Raw("ace_g0_out_samples_f32_1x64x250.raw"));
        var withCodes = positive.Map(e => e with { AudioCodes = Codes });
        var latentCodes = await new KSampler().ExecuteAsync(model, withCodes, new ConditioningZeroOut().Execute(withCodes), empty, Ctx("KSampler"), services, 31, 2, 1.0, "euler", "simple", 1.0);
        double on = RelRms(latentCodes.Data, Raw("ace_g1_out_samples_f32_1x64x250.raw"));
        var vae = await new VAELoader().ExecuteAsync("ace_1.5_vae.safetensors", Ctx("VAELoader"), services);
        var audio = await new VAEDecodeAudio().ExecuteAsync(latent, vae, Ctx("VAEDecodeAudio"));
        Assert.That((audio.ChannelCount, audio.Length, audio.SampleRate), Is.EqualTo((2, 480000, 48000)));
        var theirs = Raw("ace_g0_audio_waveform_f32_1x2x480000.raw");
        double audioErr = RelRms([.. audio.Channels[0], .. audio.Channels[1]], theirs);
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: final latent codes off {off:E2}, codes on {on:E2}; audio {audioErr:E2} relative RMS vs ComfyUI");
        // Measured on the RX 590 (OpenCL): 8.4e-5, 5.0e-5, audio 2.1e-4.
        Assert.That(off, Is.LessThan(1e-3));
        Assert.That(on, Is.LessThan(1e-3));
        Assert.That(audioErr, Is.LessThan(2e-3));
    }

    /// <summary>
    /// The shipped path: ACE-Step's MIT silence latent, not ComfyUI's table. The two differ by ~2e-3 and the run amplifies that about tenfold: the
    /// library measured 1.6e-2 (latent) / 3.4e-2 (audio) on CUDA for the whole codes-off run; this bound says so.
    /// </summary>
    [Test, Explicit(Needs)]
    public async Task The_template_graph_with_the_MIT_silence_latent_stays_near_ComfyUI()
    {
        var (runtime, services, clip) = await ClipAsync();
        using var _r = runtime;
        NodeContext Ctx(string type) => Context(services, type);
        var s = Settings();
        var positive = await new TextEncodeAceStepAudio15().ExecuteAsync(clip, Ctx("TextEncodeAceStepAudio1.5"), s.Tags, s.Lyrics, 31, s.Bpm, 10.0,
            s.TimeSignature, s.Language, s.KeyScale, false, s.CfgScale, s.Temperature, s.TopP, 0, 0);
        var unet = await new UNETLoader().ExecuteAsync("acestep_v1.5_turbo.safetensors", "default", Ctx("UNETLoader"), services);
        var model = new ModelSamplingAuraFlow().Execute(unet, 3.0);
        var latent = await new KSampler().ExecuteAsync(model, positive, new ConditioningZeroOut().Execute(positive), new EmptyAceStep15LatentAudio().Execute(10.0, 1),
            Ctx("KSampler"), services, 31, 2, 1.0, "euler", "simple", 1.0);
        double err = RelRms(latent.Data, Raw("ace_g0_out_samples_f32_1x64x250.raw"));
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: final latent with the MIT silence latent {err:E2} relative RMS vs ComfyUI (its own table: see the test above)");
        // Measured on the RX 590: 1.63e-2 (the library's CUDA figure too).
        Assert.That(err, Is.LessThan(3e-2));
    }

    /// <summary>
    /// The code LM through the node: 5 codes a second, each in 0 .. 63999, the same codes for the same seed. Its sampling draws from .NET's Random,
    /// not torch's, so the codes are not ComfyUI's (the library checked the LM itself teacher-forced against ComfyUI).
    /// </summary>
    [Test, Explicit(Needs)]
    public async Task The_code_LM_makes_five_codes_a_second_from_the_seed()
    {
        var (runtime, services, clip) = await ClipAsync();
        using var _r = runtime;
        var s = Settings();
        async Task<int[]> CodesAsync(long seed) => (await new TextEncodeAceStepAudio15().ExecuteAsync(clip, Context(services, "TextEncodeAceStepAudio1.5"), s.Tags, s.Lyrics,
            seed, s.Bpm, 4.0, s.TimeSignature, s.Language, s.KeyScale, true, s.CfgScale, s.Temperature, s.TopP, 0, 0)).Entries.Single().AudioCodes!;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var a = await CodesAsync(31);
        var seconds = sw.Elapsed.TotalSeconds;
        var b = await CodesAsync(31);
        var c = await CodesAsync(32);
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: 20 codes in {seconds:0.0} s (first, with the load); seed 31 [{string.Join(", ", a.Take(8))} ...]");
        Assert.That(a, Has.Length.EqualTo(20));
        Assert.That(a, Is.All.InRange(0, 63999));
        Assert.That(b, Is.EqualTo(a), "same seed");
        Assert.That(c, Is.Not.EqualTo(a), "another seed");
    }
}
