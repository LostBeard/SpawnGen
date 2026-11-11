using System.Buffers.Binary;
using System.Text;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// The EXIF block ComfyUI puts into saved WebP files, written byte for byte as Pillow's <c>Image.Exif.tobytes()</c> writes it: <c>Exif\0\0</c>, a big-endian
/// TIFF header, one IFD with the entries sorted by tag (ASCII strings, NUL-terminated, inline when they fit in 4 bytes), then the longer strings, each padded
/// to an even length.
/// </summary>
public static class ExifWriter
{
    /// <summary>
    /// The tags ComfyUI's saved WebP files carry (as read from its output files): the prompt as <c>"prompt:{json}"</c> in 0x0110 (Model), then each extra_pnginfo entry as
    /// <c>"key:{json}"</c> in 0x010F (Make) and downwards, in order. <paramref name="metadata"/> is <see cref="Results.Metadata"/>: "prompt" first.
    /// </summary>
    public static byte[] ComfyMetadata(IReadOnlyDictionary<string, string> metadata)
    {
        var tags = new SortedDictionary<int, string>();
        int next = 0x010F;
        foreach (var (key, value) in metadata)
        {
            if (key == "prompt") tags[0x0110] = "prompt:" + AsciiJson.Escape(value);
            else tags[next--] = key + ":" + AsciiJson.Escape(value);
        }
        return Encode(tags);
    }

    /// <summary>One IFD of ASCII entries (tag -> text), with the <c>Exif\0\0</c> prefix.</summary>
    public static byte[] Encode(IReadOnlyDictionary<int, string> asciiTags)
    {
        var entries = asciiTags.OrderBy(t => t.Key).Select(t => (Tag: t.Key, Data: Ascii(t.Value))).ToList();
        int dataStart = 8 + 2 + 12 * entries.Count + 4;
        using var ifd = new MemoryStream();
        using var data = new MemoryStream();
        static void U16(Stream s, int v) { Span<byte> b = stackalloc byte[2]; BinaryPrimitives.WriteUInt16BigEndian(b, (ushort)v); s.Write(b); }
        static void U32(Stream s, long v) { Span<byte> b = stackalloc byte[4]; BinaryPrimitives.WriteUInt32BigEndian(b, (uint)v); s.Write(b); }
        U16(ifd, entries.Count);
        foreach (var (tag, bytes) in entries)
        {
            U16(ifd, tag);
            U16(ifd, 2); // ASCII
            U32(ifd, bytes.Length);
            if (bytes.Length <= 4)
            {
                ifd.Write(bytes);
                for (int i = bytes.Length; i < 4; i++) ifd.WriteByte(0);
            }
            else
            {
                U32(ifd, dataStart + data.Length);
                data.Write(bytes);
                if (bytes.Length % 2 == 1) data.WriteByte(0);
            }
        }
        U32(ifd, 0); // no next IFD
        return [.. "Exif\0\0"u8, .. "MM\0*"u8, 0, 0, 0, 8, .. ifd.ToArray(), .. data.ToArray()];
    }

    /// <summary>Pillow's string value: ASCII (anything else becomes '?') plus the terminating NUL.</summary>
    static byte[] Ascii(string s) => [.. Encoding.ASCII.GetBytes(s), 0];
}
