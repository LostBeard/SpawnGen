using ILGPU;
using ILGPU.Runtime.CPU;
using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>ComfyUI's VIDEO: Create Video / Get Video Components, a combo whose list holds numbers, and Save Animated WEBP's encoding.</summary>
public class VideoNodeTests
{
    static ImageBatch Frames(int n, int w, int h)
    {
        var b = new ImageBatch(n, h, w, 3);
        for (int i = 0; i < b.Data.Length; i++) b.Data[i] = (i * 37 % 256) / 255f;
        return b;
    }

    [Test]
    public void Create_Video_and_Get_Video_Components_round_trip()
    {
        var frames = Frames(5, 8, 6);
        var tone = AudioClip.Mono(new float[4800], 48000);
        var video = new CreateVideo().Execute(frames, 24, tone, "10", "HDR");
        Assert.That((video.Fps, video.Seconds), Is.EqualTo((24.0, 5 / 24.0)));
        var (images, audio, fps, bitDepth, colorSpace) = new GetVideoComponents().Execute(video);
        Assert.That(images, Is.SameAs(frames));
        Assert.That(audio, Is.SameAs(tone));
        Assert.That((fps, bitDepth, colorSpace), Is.EqualTo((24.0, "10", "HDR")));
        Assert.That(new GetVideoComponents().Execute(new CreateVideo().Execute(frames)).Audio, Is.Null, "no sound: none");
        Assert.Throws<ArgumentOutOfRangeException>(() => new VideoClip(frames, 0));
    }

    [Test]
    public async Task A_combo_whose_list_holds_numbers_takes_a_number_as_ComfyUI_does()
    {
        // ComfyUI's Create Video lists bit_depth as ["auto", 8, 10]: a workflow sends the number 8.
        var registry = new NodeRegistry().AddAssembly(typeof(CreateVideo).Assembly);
        var graph = PromptJson.Parse("""
            { "1": { "class_type": "EmptyImage", "inputs": { "width": 8, "height": 8, "batch_size": 2, "color": 0 } },
              "2": { "class_type": "CreateVideo", "inputs": { "images": ["1", 0], "fps": 12.0, "bit_depth": 8 } },
              "3": { "class_type": "GetVideoComponents", "inputs": { "video": ["2", 0] } },
              "4": { "class_type": "PreviewAny", "inputs": { "source": ["3", 3] } } }
            """);
        Assert.That(PromptValidator.Validate(registry, graph).Errors, Is.Empty);
        var result = await new PromptExecutor(registry).ExecuteAsync(graph);
        Assert.That(result.Outputs["4"].Ui!["text"], Is.EqualTo(new object[] { "8" }));
        var wrong = PromptJson.Parse("""
            { "1": { "class_type": "EmptyImage", "inputs": { "width": 8, "height": 8, "batch_size": 2, "color": 0 } },
              "2": { "class_type": "CreateVideo", "inputs": { "images": ["1", 0], "fps": 12.0, "bit_depth": 12 } },
              "3": { "class_type": "GetVideoComponents", "inputs": { "video": ["2", 0] } },
              "4": { "class_type": "PreviewAny", "inputs": { "source": ["3", 3] } } }
            """);
        Assert.That(PromptValidator.Validate(registry, wrong).Errors.Single().Message, Does.Contain("'12' is not one of"));
    }

    [Test]
    public async Task Save_Animated_WEBP_writes_every_frame_losslessly()
    {
        using var context = Context.Create(b => b.CPU());
        using var accelerator = context.CreateCPUAccelerator(0);
        var frames = Frames(3, 16, 12);
        var webp = await SaveAnimatedWEBP.EncodeAsync(frames, 24, 80, accelerator);
        Assert.That(System.Text.Encoding.ASCII.GetString(webp, 8, 4), Is.EqualTo("WEBP"));
        var decoded = (await CodecsImages.DecodeAsync(accelerator, webp))!;
        Assert.That((decoded.Width, decoded.Height, decoded.Rgba.Count), Is.EqualTo((16, 12, 3)));
        for (int f = 0; f < 3; f++)
            for (int i = 0; i < 16 * 12; i++)
                for (int c = 0; c < 3; c++)
                    Assert.That(decoded.Rgba[f][i * 4 + c], Is.EqualTo((byte)Math.Round(frames.Data[(f * 16 * 12 + i) * 3 + c] * 255)), $"frame {f} pixel {i}");
    }
}
