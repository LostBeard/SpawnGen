using SpawnGen.Engine.Nodes;

namespace SpawnGen.Engine.Execution;

/// <summary>
/// Remembers node outputs by signature so an edit only re-runs the nodes downstream of it. Least-recently-used
/// entries are dropped past <c>capacity</c>. The cache never disposes values: GPU residency of large model
/// handles belongs to the ML layer's model manager.
/// </summary>
public sealed class ResultCache(int capacity = 512)
{
    sealed class Entry(NodeOutputs outputs)
    {
        public NodeOutputs Outputs { get; } = outputs;
        public long LastUsed;
    }

    readonly Dictionary<string, Entry> _entries = new(StringComparer.Ordinal);
    readonly Dictionary<string, int> _ids = new(StringComparer.Ordinal);
    long _tick;
    long _unique;

    public int Count => _entries.Count;

    /// <summary>Short stable id for a signature, so downstream signatures stay small however deep the graph is.</summary>
    internal int Intern(string signature)
    {
        if (!_ids.TryGetValue(signature, out var id)) _ids[signature] = id = _ids.Count;
        return id;
    }

    internal long NextUnique() => ++_unique;

    internal bool TryGet(string signature, out NodeOutputs outputs)
    {
        if (_entries.TryGetValue(signature, out var e))
        {
            e.LastUsed = ++_tick;
            outputs = e.Outputs;
            return true;
        }
        outputs = null!;
        return false;
    }

    internal void Set(string signature, NodeOutputs outputs) =>
        _entries[signature] = new Entry(outputs) { LastUsed = ++_tick };

    internal void Trim()
    {
        if (_entries.Count <= capacity) return;
        foreach (var key in _entries.OrderBy(kv => kv.Value.LastUsed).Take(_entries.Count - capacity).Select(kv => kv.Key).ToList())
            _entries.Remove(key);
    }

    public void Clear()
    {
        _entries.Clear();
        _ids.Clear();
    }
}
