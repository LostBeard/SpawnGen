using System.Text;
using System.Text.Json.Nodes;

namespace SpawnGen.Editor;

/// <summary>The ComfyUI protocol's binary preview frames, read the same way in the browser and in tests (no browser API here).</summary>
public static class PreviewFrames
{
    /// <summary>ComfyUI's PREVIEW_IMAGE_WITH_METADATA frame (event 4: header length, JSON header with the node, the image) as the Studio's preview event.</summary>
    public static JsonObject? ToEvent(ReadOnlySpan<byte> frame)
    {
        if (frame.Length < 8 || System.Buffers.Binary.BinaryPrimitives.ReadUInt32BigEndian(frame) != 4) return null;
        int length = (int)System.Buffers.Binary.BinaryPrimitives.ReadUInt32BigEndian(frame[4..]);
        if (length < 0 || 8 + length > frame.Length) return null;
        if (JsonNode.Parse(Encoding.UTF8.GetString(frame.Slice(8, length))) is not JsonObject meta) return null;
        var image = frame[(8 + length)..];
        return new JsonObject
        {
            ["type"] = "preview",
            ["data"] = new JsonObject
            {
                ["prompt_id"] = (string?)meta["prompt_id"], ["node"] = (string?)meta["display_node_id"] ?? (string?)meta["node_id"],
                ["image"] = $"data:{(string?)meta["image_type"] ?? "image/png"};base64,{Convert.ToBase64String(image)}",
            },
        };
    }
}
