using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Engine.Execution;

/// <summary>
/// Remembers node outputs by signature so an edit only re-runs the nodes downstream of it. Least-recently-used
/// entries are dropped past <c>capacity</c>. Tensors (IMAGE / LATENT / MASK) are counted across entries (a Save node passes
/// its input on, so one tensor can be in two): when the last entry holding one goes, its device buffer is freed. Other
/// values are never disposed: GPU residency of large model handles belongs to the ML layer's model manager.
/// </summary>
public sealed class ResultCache(int capacity = 512)
{
    sealed class Entry(NodeOutputs outputs)
    {
        public NodeOutputs Outputs { get; } = outputs;
        public long LastUsed;
        /// <summary>Whether an output is something other than plain data (a number, text, a tensor, a list of those): a model handle or anything that may hold one.</summary>
        public bool MayHoldModels { get; } = outputs.Values.Any(v => !IsPlainData(v));
    }

    readonly Dictionary<string, Entry> _entries = new(StringComparer.Ordinal);
    readonly Dictionary<HostTensor, int> _held = new(ReferenceEqualityComparer.Instance);
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

    internal void Set(string signature, NodeOutputs outputs)
    {
        foreach (var t in Tensors.In(outputs.Values)) _held[t] = _held.GetValueOrDefault(t) + 1;
        if (_entries.Remove(signature, out var old)) Release(old);
        _entries[signature] = new Entry(outputs) { LastUsed = ++_tick };
    }

    internal void Trim()
    {
        if (_entries.Count <= capacity) return;
        foreach (var key in _entries.OrderBy(kv => kv.Value.LastUsed).Take(_entries.Count - capacity).Select(kv => kv.Key).ToList())
            if (_entries.Remove(key, out var e)) Release(e);
    }

    /// <summary>
    /// Plain data keeps no model alive: null, text, numbers, booleans, enums, tensors and lists of those. Everything else (MODEL / CLIP / VAE
    /// handles, conditioning, audio or 3D objects the engine does not know) counts as possibly holding a model: the cautious side.
    /// </summary>
    static bool IsPlainData(object? value) => value switch
    {
        null or string or HostTensor => true,
        IReadOnlyList<object?> list => list.All(IsPlainData),
        _ => value.GetType().IsPrimitive || value is decimal || value.GetType().IsEnum,
    };

    /// <summary>Whether the entry under <paramref name="signature"/> exists and may hold a model (see <see cref="DropModelEntries"/>).</summary>
    internal bool MayHoldModels(string signature) => _entries.TryGetValue(signature, out var e) && e.MayHoldModels;

    /// <summary>
    /// Drops every entry that may hold a model; plain-data entries stay (a loaded picture is still reused). Called when a prompt reuses none
    /// of them, so the ML layer can free the models of earlier workflows before this one loads its own. Returns how many were dropped.
    /// </summary>
    public int DropModelEntries()
    {
        var keys = _entries.Where(kv => kv.Value.MayHoldModels).Select(kv => kv.Key).ToList();
        foreach (var key in keys)
            if (_entries.Remove(key, out var e)) Release(e);
        return keys.Count;
    }

    /// <summary>Drops every entry; device buffers of the tensors they held are freed (after a lost device, the buffers are gone anyway).</summary>
    public void Clear()
    {
        foreach (var e in _entries.Values) Release(e);
        _entries.Clear();
        _ids.Clear();
    }

    void Release(Entry e)
    {
        foreach (var t in Tensors.In(e.Outputs.Values))
        {
            if (!_held.TryGetValue(t, out var n)) continue;
            if (n > 1) { _held[t] = n - 1; continue; }
            _held.Remove(t);
            t.Dispose();
        }
    }

    /// <summary>Whether a cache entry holds <paramref name="tensor"/>.</summary>
    public bool Holds(HostTensor tensor) => _held.ContainsKey(tensor);

    /// <summary>Frees the device buffers of a finished run's tensors that no entry holds, except those handed to the caller.</summary>
    internal void ReleaseUnheld(IEnumerable<NodeOutputs> run, IEnumerable<NodeOutputs> keep)
    {
        var kept = new HashSet<HostTensor>(Tensors.In(keep.SelectMany(o => o.Values)), ReferenceEqualityComparer.Instance);
        foreach (var t in Tensors.In(run.SelectMany(o => o.Values)))
            if (!_held.ContainsKey(t) && !kept.Contains(t)) t.Dispose();
    }
}
