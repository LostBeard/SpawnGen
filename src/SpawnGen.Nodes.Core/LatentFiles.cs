using System.Buffers.Binary;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// The safetensors file format (Hugging Face's specification): an unsigned 64-bit little-endian header length N, N bytes of UTF-8 JSON mapping each
/// tensor name to its dtype, shape and [begin, end) data offsets (relative to the end of the header), an optional "__metadata__" map of strings, then
/// the tensor bytes, little-endian.
/// </summary>
public static class Safetensors
{
    /// <summary>One tensor as stored: its dtype name ("F32", "F16", ...), shape, and raw little-endian bytes.</summary>
    public sealed record Tensor(string Dtype, int[] Shape, byte[] Data);

    static readonly JsonWriterOptions Compact = new() { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping };

    /// <summary>
    /// A file holding the float32 tensors in the given order, with <paramref name="metadata"/> first when there is any. The header is padded with spaces
    /// to a multiple of 8 bytes, as the format recommends.
    /// </summary>
    public static byte[] Write(IReadOnlyList<(string Name, int[] Shape, float[] Data)> tensors, IReadOnlyDictionary<string, string>? metadata = null)
    {
        using var json = new MemoryStream();
        using (var w = new Utf8JsonWriter(json, Compact))
        {
            w.WriteStartObject();
            if (metadata is { Count: > 0 })
            {
                w.WriteStartObject("__metadata__");
                foreach (var (key, value) in metadata) w.WriteString(key, value);
                w.WriteEndObject();
            }
            long offset = 0;
            foreach (var (name, shape, data) in tensors)
            {
                w.WriteStartObject(name);
                w.WriteString("dtype", "F32");
                w.WriteStartArray("shape");
                foreach (var d in shape) w.WriteNumberValue(d);
                w.WriteEndArray();
                w.WriteStartArray("data_offsets");
                w.WriteNumberValue(offset);
                w.WriteNumberValue(offset += 4L * data.Length);
                w.WriteEndArray();
                w.WriteEndObject();
            }
            w.WriteEndObject();
        }
        var header = json.ToArray();
        int padded = (header.Length + 7) / 8 * 8;
        long total = 8 + padded + tensors.Sum(t => 4L * t.Data.Length);
        var file = new byte[total];
        BinaryPrimitives.WriteUInt64LittleEndian(file, (ulong)padded);
        header.CopyTo(file, 8);
        file.AsSpan(8 + header.Length, padded - header.Length).Fill((byte)' ');
        int at = 8 + padded;
        foreach (var (_, _, data) in tensors)
        {
            for (int i = 0; i < data.Length; i++) BinaryPrimitives.WriteSingleLittleEndian(file.AsSpan(at + 4 * i), data[i]);
            at += 4 * data.Length;
        }
        return file;
    }

    /// <summary>The tensors of a file by name, and its "__metadata__" strings (empty when there are none).</summary>
    public static Dictionary<string, Tensor> Read(byte[] file, out Dictionary<string, string> metadata)
    {
        if (file.Length < 8) throw new InvalidDataException("Not a safetensors file: it is shorter than the 8-byte header length.");
        ulong length = BinaryPrimitives.ReadUInt64LittleEndian(file);
        if (length > (ulong)(file.Length - 8)) throw new InvalidDataException("Not a safetensors file: the header length runs past the end of the file.");
        int dataStart = 8 + (int)length;
        JsonDocument doc;
        try { doc = JsonDocument.Parse(file.AsMemory(8, (int)length)); }
        catch (JsonException e) { throw new InvalidDataException("Not a safetensors file: the header is not JSON (" + e.Message + ")."); }
        using (doc)
        {
            if (doc.RootElement.ValueKind != JsonValueKind.Object) throw new InvalidDataException("Not a safetensors file: the header is not a JSON object.");
            metadata = [];
            var tensors = new Dictionary<string, Tensor>();
            foreach (var entry in doc.RootElement.EnumerateObject())
            {
                try
                {
                    if (entry.Name == "__metadata__")
                    {
                        foreach (var m in entry.Value.EnumerateObject()) metadata[m.Name] = m.Value.GetString() ?? "";
                        continue;
                    }
                    var dtype = entry.Value.GetProperty("dtype").GetString() ?? "";
                    var shape = entry.Value.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray();
                    var offsets = entry.Value.GetProperty("data_offsets").EnumerateArray().Select(e => e.GetInt64()).ToArray();
                    if (offsets.Length != 2 || offsets[0] < 0 || offsets[1] < offsets[0] || dataStart + offsets[1] > file.Length)
                        throw new InvalidDataException($"The tensor '{entry.Name}' has data offsets outside the file.");
                    tensors[entry.Name] = new Tensor(dtype, shape, file.AsSpan(dataStart + (int)offsets[0], (int)(offsets[1] - offsets[0])).ToArray());
                }
                catch (Exception e) when (e is KeyNotFoundException or InvalidOperationException or FormatException)
                {
                    throw new InvalidDataException($"The safetensors entry '{entry.Name}' is malformed: {e.Message}");
                }
            }
            return tensors;
        }
    }

    /// <summary>The tensor's values widened to float32: F32, F16, BF16 and F64 (rounded to float32). Other dtypes throw <see cref="NotSupportedException"/>.</summary>
    public static float[] ToFloats(Tensor tensor)
    {
        var d = tensor.Data;
        switch (tensor.Dtype)
        {
            case "F32":
            {
                var r = new float[d.Length / 4];
                for (int i = 0; i < r.Length; i++) r[i] = BinaryPrimitives.ReadSingleLittleEndian(d.AsSpan(4 * i));
                return r;
            }
            case "F16":
            {
                var r = new float[d.Length / 2];
                for (int i = 0; i < r.Length; i++) r[i] = (float)BinaryPrimitives.ReadHalfLittleEndian(d.AsSpan(2 * i));
                return r;
            }
            case "BF16":
            {
                // bfloat16 is the upper half of a float32.
                var r = new float[d.Length / 2];
                for (int i = 0; i < r.Length; i++) r[i] = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadUInt16LittleEndian(d.AsSpan(2 * i)) << 16);
                return r;
            }
            case "F64":
            {
                var r = new float[d.Length / 8];
                for (int i = 0; i < r.Length; i++) r[i] = (float)BinaryPrimitives.ReadDoubleLittleEndian(d.AsSpan(8 * i));
                return r;
            }
            default:
                throw new NotSupportedException($"Tensors of dtype {tensor.Dtype} cannot be read as floats here (F32, F16, BF16 and F64 can).");
        }
    }
}

[Node(Category = "model/latent", DisplayName = "Load Latent")]
public sealed class LoadLatent
{
    /// <summary>The .latent files in the input folder (measured: ComfyUI lists names ending in ".latent", lower case only, sorted).</summary>
    public static string[] LatentNames() => [.. InputFiles.Store.List(n => n.EndsWith(".latent", StringComparison.Ordinal))];

    public async Task<LatentBatch> ExecuteAsync([Input(OptionsProvider = nameof(LatentNames))] string latent) =>
        Decode(await InputFiles.Store.ReadAsync(latent));

    /// <summary>
    /// The latent in a .latent file: a safetensors file whose "latent_tensor" is (batch, channels, height, width). A file that also has a
    /// "latent_format_version_0" entry holds the values as they are; an older file without it was stored multiplied by 0.18215 and is multiplied by
    /// float32(1 / 0.18215) here.
    /// </summary>
    public static LatentBatch Decode(byte[] file)
    {
        var tensors = Safetensors.Read(file, out _);
        if (!tensors.TryGetValue("latent_tensor", out var tensor))
            throw new InvalidDataException("This file has no latent_tensor, so it is not a latent file (Save Latent writes those).");
        var values = Safetensors.ToFloats(tensor);
        if (tensor.Shape.Length != 4)
            throw new InvalidDataException($"The latent_tensor has {values.Length} values in shape [{string.Join(", ", tensor.Shape)}]; a picture latent has 4 dimensions (batch, channels, height, width).");
        if (!tensors.ContainsKey("latent_format_version_0"))
        {
            float scale = (float)(1.0 / 0.18215);
            for (int i = 0; i < values.Length; i++) values[i] *= scale;
        }
        return new LatentBatch(tensor.Shape[0], tensor.Shape[1], tensor.Shape[2], tensor.Shape[3], values);
    }
}

[Node(Category = "model/latent", DisplayName = "Save Latent", IsOutput = true, OutputNames = ["samples"])]
public sealed class SaveLatent
{
    /// <summary>
    /// Writes the latent as <c>prefix_00001_.latent</c> in the output folder (a "folder/name" prefix makes a subfolder; measured: ComfyUI names them the
    /// same way) with the prompt and the editor's extras as safetensors metadata, and passes the latent on.
    /// </summary>
    public async Task<(LatentBatch Samples, UiOutput Ui)> ExecuteAsync(LatentBatch samples, NodeContext context, IServiceProvider services,
        string filenamePrefix = "latents/ComfyUI")
    {
        var file = await Results.For(services).SaveAsync("output", filenamePrefix, Encode(samples, Results.Metadata(context)), "latent");
        return (samples, new UiOutput("latents", new object[] { file }));
    }

    /// <summary>
    /// The .latent file: "latent_tensor" (float32, the latent's own values) and an empty "latent_format_version_0" tensor that marks the values as
    /// unscaled, with <paramref name="metadata"/> as the safetensors metadata.
    /// </summary>
    public static byte[] Encode(LatentBatch latent, IReadOnlyDictionary<string, string>? metadata = null) =>
        Safetensors.Write([("latent_tensor", latent.Shape, latent.Data), ("latent_format_version_0", [0], [])], metadata);
}
