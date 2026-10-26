using System.Globalization;
using System.Numerics;
using System.Text.Json;
using System.Text.Json.Nodes;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Engine.Types;

/// <summary>
/// ComfyUI's DICT: a JSON-like object (Python <c>dict</c>) moving between nodes. Keys keep the order they were added in; values are null, bool,
/// whole numbers (long or BigInteger), double, string, <see cref="ArrayValue"/> or another <see cref="DictValue"/>.
/// </summary>
[Socket("DICT")]
public sealed class DictValue : OrderedDictionary<string, object?>
{
    public DictValue() : base(StringComparer.Ordinal) { }
}

/// <summary>ComfyUI's ARRAY: a JSON-like list (Python <c>list</c>) moving between nodes, with the same kinds of items as <see cref="DictValue"/>.</summary>
[Socket("ARRAY")]
public sealed class ArrayValue : List<object?>
{
    public ArrayValue() { }

    public ArrayValue(IEnumerable<object?> items) : base(items) { }
}

/// <summary>
/// JSON text to and from the values a <see cref="DictValue"/> holds, as Python's <c>json</c> reads and writes them: objects become <see cref="DictValue"/>,
/// arrays <see cref="ArrayValue"/>, whole numbers long (BigInteger when they do not fit), other numbers double. Object-valued widgets (ComfyUI's
/// BOUNDING_BOX, BOUNDING_BOXES and COMPOSITOR) hold such values in prompts and workflows.
/// </summary>
public static class JsonLiteral
{
    public static object? FromJson(JsonElement e)
    {
        switch (e.ValueKind)
        {
            case JsonValueKind.Object:
                var d = new DictValue();
                foreach (var p in e.EnumerateObject()) d[p.Name] = FromJson(p.Value); // the last value of a repeated key wins, as in Python
                return d;
            case JsonValueKind.Array:
                return new ArrayValue(e.EnumerateArray().Select(FromJson));
            case JsonValueKind.String: return e.GetString();
            case JsonValueKind.True: return true;
            case JsonValueKind.False: return false;
            case JsonValueKind.Number:
                if (e.TryGetInt64(out var l)) return l;
                var raw = e.GetRawText();
                if (raw.IndexOfAny(['.', 'e', 'E']) < 0) return BigInteger.Parse(raw, CultureInfo.InvariantCulture);
                return e.GetDouble();
            default: return null;
        }
    }

    public static object? FromJson(JsonNode? node)
    {
        if (node is null) return null;
        using var doc = JsonDocument.Parse(node.ToJsonString());
        return FromJson(doc.RootElement);
    }

    /// <summary>The value of a JSON text; throws <see cref="JsonException"/> when it is not JSON.</summary>
    public static object? Parse(string json)
    {
        using var doc = JsonDocument.Parse(json);
        return FromJson(doc.RootElement);
    }

    /// <summary>The value of a JSON text; false when it is not JSON.</summary>
    public static bool TryParse(string json, out object? value)
    {
        try
        {
            value = Parse(json);
            return true;
        }
        catch (JsonException)
        {
            value = null;
            return false;
        }
    }

    /// <summary>Compact JSON text of the value (a whole double keeps its ".0", as Python writes it).</summary>
    public static string ToJson(object? value)
    {
        using var ms = new MemoryStream();
        using (var w = new Utf8JsonWriter(ms)) Write(w, value);
        return System.Text.Encoding.UTF8.GetString(ms.ToArray());
    }

    public static JsonNode? ToNode(object? value) => JsonNode.Parse(ToJson(value));

    public static void Write(Utf8JsonWriter w, object? value)
    {
        switch (value)
        {
            case null: w.WriteNullValue(); break;
            case bool b: w.WriteBooleanValue(b); break;
            case string s: w.WriteStringValue(s); break;
            case long l: w.WriteNumberValue(l); break;
            case int i: w.WriteNumberValue(i); break;
            case BigInteger big: w.WriteRawValue(big.ToString(CultureInfo.InvariantCulture), skipInputValidation: true); break;
            case float f: WriteDouble(w, f); break;
            case double dv: WriteDouble(w, dv); break;
            case DictValue dict:
                w.WriteStartObject();
                foreach (var (k, v) in dict)
                {
                    w.WritePropertyName(k);
                    Write(w, v);
                }
                w.WriteEndObject();
                break;
            case ArrayValue array:
                w.WriteStartArray();
                foreach (var v in array) Write(w, v);
                w.WriteEndArray();
                break;
            default: throw new NotSupportedException($"A {value.GetType().Name} has no JSON form.");
        }
    }

    static void WriteDouble(Utf8JsonWriter w, double d)
    {
        // Python's json writes NaN / Infinity / -Infinity (not strict JSON, but what it reads back too).
        string text = double.IsNaN(d) ? "NaN" : double.IsPositiveInfinity(d) ? "Infinity" : double.IsNegativeInfinity(d) ? "-Infinity"
            : d.ToString("R", CultureInfo.InvariantCulture);
        if (double.IsFinite(d) && text.IndexOfAny(['.', 'E', 'e']) < 0) text += ".0";
        w.WriteRawValue(text, skipInputValidation: true);
    }
}
