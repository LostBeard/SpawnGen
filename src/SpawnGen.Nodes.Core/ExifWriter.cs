using System.Text;

namespace SpawnGen.Nodes.Core;

// Written clean-room (docs/CLEANROOM.md) from the TIFF 6.0 / Exif 2.3 specifications, with the layout and the tags read off Fixtures/codecs/save_webp
// (the bytes Pillow writes for ComfyUI's animated WEBP): "Exif\0\0", a big-endian TIFF header, one IFD with ASCII entries in tag order, values longer
// than four bytes after the IFD on even offsets; the prompt in tag 0x0110 and the workflow's keys from 0x010F down, each as "key:value".

/// <summary>EXIF blocks of ASCII tags, as ComfyUI's WEBP files carry the prompt and workflow.</summary>
public static class ExifWriter
{
    /// <summary>The run's metadata as EXIF: "prompt:..." in 0x0110, then each other key as "key:..." in 0x010F, 0x010E, ... (the values JSON, ASCII-escaped).</summary>
    public static byte[] ComfyMetadata(IReadOnlyDictionary<string, string> metadata)
    {
        var tags = new Dictionary<int, string>();
        if (metadata.TryGetValue("prompt", out var prompt)) tags[0x0110] = "prompt:" + AsciiJson.Escape(prompt);
        int tag = 0x010F;
        foreach (var (key, value) in metadata)
        {
            if (key == "prompt") continue;
            tags[tag--] = key + ":" + AsciiJson.Escape(value);
        }
        return Encode(tags);
    }

    /// <summary>"Exif\0\0" and a big-endian TIFF with one IFD of ASCII entries (tag order); each value NUL-terminated.</summary>
    public static byte[] Encode(IReadOnlyDictionary<int, string> asciiTags)
    {
        var entries = asciiTags.OrderBy(t => t.Key).Select(t => (Tag: t.Key, Bytes: (byte[])[.. Encoding.ASCII.GetBytes(t.Value), (byte)0])).ToList();
        int ifdSize = 2 + 12 * entries.Count + 4, dataStart = 8 + ifdSize;
        var data = new List<byte>();
        var tiff = new List<byte>();
        void U16(int v) { tiff.Add((byte)(v >> 8)); tiff.Add((byte)v); }
        void U32(int v) { tiff.Add((byte)(v >> 24)); tiff.Add((byte)(v >> 16)); tiff.Add((byte)(v >> 8)); tiff.Add((byte)v); }
        tiff.AddRange("MM"u8.ToArray());
        U16(42);
        U32(8);
        U16(entries.Count);
        foreach (var (t, bytes) in entries)
        {
            U16(t);
            U16(2); // ASCII
            U32(bytes.Length);
            if (bytes.Length <= 4)
            {
                for (int i = 0; i < 4; i++) tiff.Add(i < bytes.Length ? bytes[i] : (byte)0);
            }
            else
            {
                if (data.Count % 2 == 1) data.Add(0);
                U32(dataStart + data.Count);
                data.AddRange(bytes);
            }
        }
        U32(0); // no next IFD
        tiff.AddRange(data);
        if (tiff.Count % 2 == 1) tiff.Add(0);
        return [.. "Exif\0\0"u8.ToArray(), .. tiff];
    }
}
