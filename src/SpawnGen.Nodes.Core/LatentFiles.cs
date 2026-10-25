using System.Buffers.Binary;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// The safetensors container ComfyUI's <c>.latent</c> files use: 8 bytes of header length, a JSON header (tensor name to dtype, shape and byte range, plus
/// <c>__metadata__</c>), then the raw tensor bytes. Only what latents need: reading F32 / F16 / BF16 tensors and writing F32 ones.
/// </summary>
public static class Safetensors
{
    public sealed record Tensor(string Dtype, int[] Shape, byte[] Bytes);

    public static Dictionary<string, Tensor> Read(byte[] file, out Dictionary<string, string> metadata)
    {
        if (file.Length < 8) throw new InvalidDataException("Not a safetensors file (too short).");
        ulong headerLength = BinaryPrimitives.ReadUInt64LittleEndian(file);
        if (headerLength > (ulong)(file.Length - 8)) throw new InvalidDataException("Not a safetensors file (the header is longer than the file).");
        var header = JsonNode.Parse(Encoding.UTF8.GetString(file, 8, (int)headerLength)) as JsonObject ?? throw new InvalidDataException("The safetensors header is not a JSON object.");
        long dataStart = 8 + (long)headerLength;
        metadata = [];
        var tensors = new Dictionary<string, Tensor>();
        foreach (var (name, value) in header)
        {
            if (value is not JsonObject o) continue;
            if (name == "__metadata__") { foreach (var (k, v) in o) metadata[k] = v?.GetValue<string>() ?? ""; continue; }
            var shape = o["shape"]!.AsArray().Select(d => (int)d!).ToArray();
            var offsets = o["data_offsets"]!.AsArray();
            long begin = (long)offsets[0]!, end = (long)offsets[1]!;
            if (begin < 0 || end < begin || dataStart + end > file.Length) throw new InvalidDataException($"Tensor '{name}' points outside the file.");
            tensors[name] = new Tensor((string)o["dtype"]!, shape, file.AsSpan((int)(dataStart + begin), (int)(end - begin)).ToArray());
        }
        return tensors;
    }

    /// <summary>The tensor's values as floats, whatever its stored type.</summary>
    public static float[] ToFloats(Tensor t)
    {
        int size = t.Dtype switch { "F32" => 4, "F16" or "BF16" => 2, _ => throw new NotSupportedException($"Tensor type {t.Dtype} is not supported (F32, F16 and BF16 are).") };
        var result = new float[t.Bytes.Length / size];
        for (int i = 0; i < result.Length; i++)
            result[i] = t.Dtype switch
            {
                "F32" => BinaryPrimitives.ReadSingleLittleEndian(t.Bytes.AsSpan(i * 4)),
                "F16" => (float)BinaryPrimitives.ReadHalfLittleEndian(t.Bytes.AsSpan(i * 2)),
                _ => BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadUInt16LittleEndian(t.Bytes.AsSpan(i * 2)) << 16),
            };
        return result;
    }

    /// <summary>A file holding the given F32 tensors, in the order given, with optional string metadata.</summary>
    public static byte[] Write(IReadOnlyList<(string Name, int[] Shape, float[] Data)> tensors, IReadOnlyDictionary<string, string>? metadata = null)
    {
        var header = new JsonObject();
        if (metadata is { Count: > 0 })
        {
            var meta = new JsonObject();
            foreach (var (k, v) in metadata) meta[k] = v;
            header["__metadata__"] = meta;
        }
        long offset = 0;
        foreach (var (name, shape, data) in tensors)
        {
            long end = offset + (long)data.Length * 4;
            header[name] = new JsonObject
            {
                ["dtype"] = "F32", ["shape"] = new JsonArray(shape.Select(d => (JsonNode)d).ToArray()), ["data_offsets"] = new JsonArray(offset, end),
            };
            offset = end;
        }
        var json = Encoding.UTF8.GetBytes(header.ToJsonString(new JsonSerializerOptions { Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping }));
        int pad = (8 - json.Length % 8) % 8; // the header is padded with spaces to a multiple of 8, as the reference writer does
        var file = new byte[8 + json.Length + pad + offset];
        BinaryPrimitives.WriteUInt64LittleEndian(file, (ulong)(json.Length + pad));
        json.CopyTo(file, 8);
        for (int i = 0; i < pad; i++) file[8 + json.Length + i] = (byte)' ';
        long at = 8 + json.Length + pad;
        foreach (var (_, _, data) in tensors)
            foreach (var v in data) { BinaryPrimitives.WriteSingleLittleEndian(file.AsSpan((int)at), v); at += 4; }
        return file;
    }
}

[Node(Category = "model/latent", DisplayName = "Load Latent")]
public sealed class LoadLatent
{
    public static string[] LatentNames() => [.. InputFiles.Store.List(n => n.EndsWith(".latent", StringComparison.OrdinalIgnoreCase))];

    /// <summary>Reads a <c>.latent</c> file ComfyUI (or Save Latent) wrote. Files from before ComfyUI scaled latents are rescaled the way it does.</summary>
    public async Task<LatentBatch> ExecuteAsync([Input(OptionsProvider = nameof(LatentNames))] string latent) => Decode(await InputFiles.Store.ReadAsync(latent));

    public static LatentBatch Decode(byte[] file)
    {
        var tensors = Safetensors.Read(file, out _);
        if (!tensors.TryGetValue("latent_tensor", out var t)) throw new InvalidDataException("This file has no 'latent_tensor': it is not a latent file.");
        if (t.Shape.Length != 4) throw new InvalidDataException($"The latent has {t.Shape.Length} dimensions; a picture latent has 4 (batch, channels, height, width).");
        var data = Safetensors.ToFloats(t);
        // A file holds the VAE's own values, as a LATENT does (the model scales them when it samples). The earliest files (no version tag) were saved
        // scaled by SD's 0.18215: ComfyUI multiplies them by 1 / 0.18215 (in float32) on load.
        if (!tensors.ContainsKey("latent_format_version_0"))
        {
            float unscale = (float)(1.0 / LegacyScale);
            for (int i = 0; i < data.Length; i++) data[i] *= unscale;
        }
        return new LatentBatch(t.Shape[0], t.Shape[1], t.Shape[2], t.Shape[3], data);
    }

    /// <summary>The scale the unversioned latent files of early ComfyUI were saved with (SD 1.x's latent scale factor).</summary>
    public const float LegacyScale = 0.18215f;
}

[Node(Category = "model/latent", DisplayName = "Save Latent", IsOutput = true)]
public sealed class SaveLatent
{
    /// <summary>
    /// Writes the latent as a <c>.latent</c> file (safetensors, ComfyUI's layout, with the prompt and the workflow as metadata) in the output folder:
    /// <c>prefix_00001_.latent</c>. The latent also passes through to the output, as in ComfyUI.
    /// </summary>
    public async Task<(LatentBatch Samples, UiOutput Ui)> ExecuteAsync(LatentBatch samples, NodeContext context, IServiceProvider services, string filenamePrefix = "latents/ComfyUI")
    {
        var file = await Results.For(services).SaveAsync("output", filenamePrefix, Encode(samples, Results.Metadata(context)), "latent");
        return (samples, new UiOutput("latents", new object[] { file }));
    }

    public static byte[] Encode(LatentBatch samples, IReadOnlyDictionary<string, string>? metadata = null) => Safetensors.Write(
    [
        // The VAE's values, as the LATENT holds them (see LoadLatent.Decode).
        ("latent_tensor", [samples.Batch, samples.Channels, samples.Height, samples.Width], samples.Data),
        ("latent_format_version_0", [0], []),
    ], metadata);
}
