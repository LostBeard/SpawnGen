using System.Buffers.Binary;
using System.Text;
using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

/// <summary>Latent files (ComfyUI's safetensors-based .latent) and the latent batch nodes, against hand-computed values.</summary>
public class LatentFileTests
{
    static LatentBatch Lat(int b, int c, int h, int w, Func<int, float> f) => new(b, c, h, w, Enumerable.Range(0, b * c * h * w).Select(f).ToArray());

    [Test]
    public void A_saved_latent_reads_back_exactly()
    {
        var latent = Lat(2, 4, 3, 5, i => i * 0.37f - 8f);
        var back = LoadLatent.Decode(SaveLatent.Encode(latent));
        Assert.That((back.Batch, back.Channels, back.Height, back.Width), Is.EqualTo((2, 4, 3, 5)));
        // The file holds the VAE's own values (ComfyUI's LATENT); reading it scales them back, so the latent returns to float rounding.
        Assert.That(back.Data, Is.EqualTo(latent.Data).Within(1e-5f));
    }

    [Test]
    public void The_file_has_the_layout_ComfyUI_and_the_safetensors_library_write()
    {
        var file = SaveLatent.Encode(Lat(1, 4, 2, 2, i => i));
        ulong headerLength = BinaryPrimitives.ReadUInt64LittleEndian(file);
        Assert.That(headerLength % 8, Is.EqualTo(0UL), "the header is padded to a multiple of 8");
        var header = Encoding.UTF8.GetString(file, 8, (int)headerLength).TrimEnd();
        Assert.That(header, Is.EqualTo("{\"latent_tensor\":{\"dtype\":\"F32\",\"shape\":[1,4,2,2],\"data_offsets\":[0,64]},\"latent_format_version_0\":{\"dtype\":\"F32\",\"shape\":[0],\"data_offsets\":[64,64]}}"));
        Assert.That(file.Length, Is.EqualTo(8 + (int)headerLength + 64));
        Assert.That(BinaryPrimitives.ReadSingleLittleEndian(file.AsSpan(8 + (int)headerLength + 4)), Is.EqualTo(1f / LoadLatent.SdScale), "stored without the model's scale, as ComfyUI stores it");
    }

    [Test]
    public void Half_precision_and_bfloat16_latents_are_widened_and_old_unversioned_files_are_rescaled()
    {
        // Hand-built files: F16 1.5 and -2.0 (0x3E00, 0xC000); BF16 1.5 and -2.0 (0x3FC0, 0xC000).
        static byte[] File(string dtype, ushort[] values, bool versioned)
        {
            var data = new byte[values.Length * 2];
            for (int i = 0; i < values.Length; i++) BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(i * 2), values[i]);
            var json = $"{{\"latent_tensor\":{{\"dtype\":\"{dtype}\",\"shape\":[1,1,1,{values.Length}],\"data_offsets\":[0,{data.Length}]}}" +
                       (versioned ? $",\"latent_format_version_0\":{{\"dtype\":\"F32\",\"shape\":[0],\"data_offsets\":[{data.Length},{data.Length}]}}" : "") + "}";
            var head = Encoding.UTF8.GetBytes(json.PadRight((json.Length + 7) / 8 * 8));
            var file = new byte[8 + head.Length + data.Length];
            BinaryPrimitives.WriteUInt64LittleEndian(file, (ulong)head.Length);
            head.CopyTo(file, 8);
            data.CopyTo(file, 8 + head.Length);
            return file;
        }
        // A versioned file holds the VAE's values: SpawnGen's LATENT is in model space, scaled by 0.18215 as ComfyUI's model scales on sampling.
        Assert.That(LoadLatent.Decode(File("F16", [0x3E00, 0xC000], true)).Data, Is.EqualTo(new[] { 1.5f * LoadLatent.SdScale, -2f * LoadLatent.SdScale }));
        Assert.That(LoadLatent.Decode(File("BF16", [0x3FC0, 0xC000], true)).Data, Is.EqualTo(new[] { 1.5f * LoadLatent.SdScale, -2f * LoadLatent.SdScale }));
        // An old unversioned file was saved scaled: ComfyUI divides by 0.18215 on load and its model multiplies again, so it reads back as written.
        var old = LoadLatent.Decode(File("F16", [0x3E00, 0xC000], false)).Data;
        Assert.That(old[0], Is.EqualTo(1.5f).Within(1e-6), "files from before latent scaling keep their values, as in ComfyUI");
    }

    [Test]
    public void Files_that_are_not_latents_say_so()
    {
        Assert.That(() => LoadLatent.Decode([1, 2, 3]), Throws.TypeOf<InvalidDataException>());
        var other = Safetensors.Write([("weights", [2], [1f, 2f])]);
        Assert.That(() => LoadLatent.Decode(other), Throws.TypeOf<InvalidDataException>().With.Message.Contains("latent_tensor"));
        var flat = Safetensors.Write([("latent_tensor", [4], [1f, 2f, 3f, 4f])]);
        Assert.That(() => LoadLatent.Decode(flat), Throws.TypeOf<InvalidDataException>().With.Message.Contains("has 4"));
        Assert.That(() => Safetensors.ToFloats(new Safetensors.Tensor("I64", [1], new byte[8])), Throws.TypeOf<NotSupportedException>());
    }

    [Test]
    public async Task Save_Latent_writes_a_numbered_file_and_Load_Latent_finds_it_in_the_input_folder()
    {
        var dir = Path.Combine(Path.GetTempPath(), "spawngen-latent-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(dir, "input"));
        InputFiles.Directory = Path.Combine(dir, "input");
        try
        {
            var registry = new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly);
            var options = new OutputOptions { Directory = Path.Combine(dir, "output"), TempDirectory = Path.Combine(dir, "temp") };
            var g = new PromptGraph();
            g.Add("1", "EmptyLatentImage", ("width", 64), ("height", 32), ("batch_size", 2));
            g.Add("2", "SaveLatent", ("samples", PromptInput.Link("1")), ("filename_prefix", "latents/test"));
            var result = await new PromptExecutor(registry, new Services(options)).ExecuteAsync(g);
            var file = (ResultFile)((object[])result.Outputs["2"].Ui!["latents"]!)[0];
            Assert.That(file.Filename, Is.EqualTo("test_00001_.latent"));
            Assert.That(file.Subfolder, Is.EqualTo("latents"));
            Assert.That(File.Exists(Path.Combine(dir, "output", "latents", "test_00001_.latent")), Is.True);

            File.Copy(Path.Combine(dir, "output", "latents", "test_00001_.latent"), Path.Combine(dir, "input", "mine.latent"));
            Assert.That(LoadLatent.LatentNames(), Is.EqualTo(new[] { "mine.latent" }));
            var loaded = await new LoadLatent().ExecuteAsync("mine.latent");
            Assert.That((loaded.Batch, loaded.Channels, loaded.Height, loaded.Width), Is.EqualTo((2, 4, 4, 8)));
        }
        finally { Directory.Delete(dir, true); }
    }

    [Test]
    public void Latent_Batch_stacks_two_latents_and_resizes_the_second_to_the_first()
    {
        var a = Lat(1, 2, 2, 2, i => i);
        var b = Lat(2, 2, 2, 2, i => 100 + i);
        var stacked = new LatentBatchNode().Execute(a, b);
        Assert.That(stacked.Batch, Is.EqualTo(3));
        Assert.That(stacked.Data.Take(8), Is.EqualTo(a.Data));
        Assert.That(stacked.Data.Skip(8), Is.EqualTo(b.Data));

        var big = Lat(1, 2, 4, 4, i => 7f);
        var mixed = new LatentBatchNode().Execute(a, big);
        Assert.That((mixed.Batch, mixed.Height, mixed.Width), Is.EqualTo((2, 2, 2)));
        Assert.That(mixed.Data.Skip(8).All(v => Math.Abs(v - 7f) < 1e-5), Is.True, "a flat latent stays flat when resized");
        Assert.That(() => new LatentBatchNode().Execute(a, Lat(1, 4, 2, 2, i => i)), Throws.ArgumentException);
    }

    [Test]
    public void Seed_behavior_fixed_marks_the_batch_to_share_one_noise_and_keeps_the_rest()
    {
        var mask = MaskBatch.Filled(2, 2, 2, 1f);
        var source = new LatentBatch(2, 4, 2, 2) { NoiseMask = mask };
        var fixedNoise = new LatentBatchSeedBehavior().Execute(source, "fixed");
        Assert.That(fixedNoise.BatchIndex, Is.EqualTo(new[] { 0, 0 }), "ComfyUI: batch_index = [0] * batch");
        Assert.That(fixedNoise.NoiseMask, Is.SameAs(mask));
        Assert.That(new LatentBatchSeedBehavior().Execute(fixedNoise, "random").BatchIndex, Is.Null, "random drops batch_index");
        // A picture taken from a batch keeps its index, and "fixed" spreads that one: [batch_index[0]] * batch.
        var picked = new LatentFromBatch().Execute(new LatentBatch(4, 4, 2, 2), 2, 2);
        Assert.That(picked.BatchIndex, Is.EqualTo(new[] { 2, 3 }));
        Assert.That(new LatentBatchSeedBehavior().Execute(picked, "fixed").BatchIndex, Is.EqualTo(new[] { 2, 2 }));
    }

    [Test]
    public void Composite_masked_pastes_through_the_mask_at_the_offset()
    {
        var dest = Lat(1, 1, 4, 4, _ => 0f);
        var src = Lat(1, 1, 2, 2, _ => 10f);
        // Without a mask: the source lands at latent (1,1) (pixels 8,8), the rest stays.
        var plain = new LatentCompositeMasked().Execute(dest, src, 8, 8);
        Assert.That(plain.Data, Is.EqualTo(new float[] { 0, 0, 0, 0, 0, 10, 10, 0, 0, 10, 10, 0, 0, 0, 0, 0 }));
        Assert.That(dest.Data, Is.All.EqualTo(0f), "the destination is not changed");

        // A half-strength mask blends.
        var half = new LatentCompositeMasked().Execute(dest, src, 0, 0, false, MaskBatch.Filled(1, 2, 2, 0.5f));
        Assert.That(half.Data[0], Is.EqualTo(5f).Within(1e-5));
        Assert.That(half.Data[2], Is.EqualTo(0f));

        // The source hanging over the edge is cut off; a bigger offset than the destination does nothing.
        var edge = new LatentCompositeMasked().Execute(dest, src, 24, 24);
        Assert.That(edge.Data[15], Is.EqualTo(10f));
        Assert.That(edge.Data.Count(v => v == 10f), Is.EqualTo(1));
        Assert.That(new LatentCompositeMasked().Execute(dest, src, 4000, 4000).Data, Is.All.EqualTo(0f));

        // resize_source stretches the source over the whole destination first.
        var stretched = new LatentCompositeMasked().Execute(dest, src, 0, 0, true);
        Assert.That(stretched.Data.All(v => Math.Abs(v - 10f) < 1e-5), Is.True);
    }

    [TestCase("a", "", "a")]
    [TestCase("folder/a", "folder", "a")]
    [TestCase("x\\y/z/a", "x/y/z", "a")]
    [TestCase("/a", null, null)]
    [TestCase("../a", null, null)]
    [TestCase("ok/../../a", null, null)]
    [TestCase("c:/a", null, null)]
    public void Save_prefixes_split_into_a_subfolder_and_a_name_and_cannot_leave_the_output_folder(string prefix, string? subfolder, string? name)
    {
        if (subfolder is null) { Assert.That(() => SavePrefix.Split(prefix), Throws.ArgumentException); return; }
        Assert.That(SavePrefix.Split(prefix), Is.EqualTo((subfolder, name)));
    }

    [Test]
    public void Save_prefixes_fill_ComfyUIs_size_and_time_variables()
    {
        var at = new DateTime(2026, 3, 7, 9, 5, 2);
        Assert.That(SavePrefix.ComputeVars("%year%-%month%-%day%/%width%x%height%_%hour%%minute%%second%_%date:yy%", 768, 512, at),
            Is.EqualTo("2026-03-07/768x512_090502_%date:yy%"), "%date:..% is the Studio's job, done before the prompt is sent");
        Assert.That(SavePrefix.ComputeVars("plain", 1, 1, at), Is.EqualTo("plain"));
        Assert.That(SavePrefix.Split("%year%/%width%_x").Subfolder, Is.EqualTo(DateTime.Now.Year.ToString()), "a store fills the time and 0 for sizes it does not know");
        Assert.That(SavePrefix.Split("%year%/%width%_x").Name, Is.EqualTo("0_x"));
    }

    [Test]
    public async Task Results_with_a_subfolder_prefix_are_kept_in_that_subfolder_on_disk_and_in_memory()
    {
        var dir = Path.Combine(Path.GetTempPath(), "spawngen-sub-" + Guid.NewGuid().ToString("N"));
        try
        {
            var disk = new DiskResultStore(new OutputOptions { Directory = Path.Combine(dir, "out"), TempDirectory = Path.Combine(dir, "tmp") });
            var a = await disk.SaveAsync("output", "pics/cat", [1], "png");
            var b = await disk.SaveAsync("output", "pics/cat", [2], "png");
            Assert.That((a.Filename, a.Subfolder, b.Filename), Is.EqualTo(("cat_00001_.png", "pics", "cat_00002_.png")));
            Assert.That(File.ReadAllBytes(Path.Combine(dir, "out", "pics", "cat_00002_.png")), Is.EqualTo(new byte[] { 2 }));
            Assert.That((await disk.SaveAsync("output", "cat", [3], "png")).Filename, Is.EqualTo("cat_00001_.png"), "numbering is per folder");

            var memory = new MemoryResultStore();
            var m = await memory.SaveAsync("output", "pics/cat", [9], "png");
            Assert.That((m.Filename, m.Subfolder), Is.EqualTo(("cat_00001_.png", "pics")));
            Assert.That(memory.Get("output", "cat_00001_.png", "pics"), Is.EqualTo(new byte[] { 9 }));
            Assert.That(memory.Get("output", "cat_00001_.png"), Is.Null, "the root folder has no such file");
        }
        finally { if (Directory.Exists(dir)) Directory.Delete(dir, true); }
    }

    sealed class Services(params object[] instances) : IServiceProvider
    {
        public object? GetService(Type serviceType) => instances.FirstOrDefault(serviceType.IsInstanceOfType);
    }
}
