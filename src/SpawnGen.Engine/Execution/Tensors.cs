using SpawnGen.Engine.Types;

namespace SpawnGen.Engine.Execution;

/// <summary>The tensors (IMAGE / LATENT / MASK, with what they carry) inside node values: single values and ComfyUI's lists.</summary>
public static class Tensors
{
    /// <summary>Every distinct tensor in <paramref name="values"/>, the masks a latent carries included.</summary>
    public static IEnumerable<HostTensor> In(IEnumerable<object?> values)
    {
        var seen = new HashSet<HostTensor>(ReferenceEqualityComparer.Instance);
        var stack = new Stack<object?>(values);
        while (stack.Count > 0)
        {
            switch (stack.Pop())
            {
                case HostTensor t when seen.Add(t):
                    yield return t;
                    foreach (var c in t.Carried) stack.Push(c);
                    break;
                case IReadOnlyList<object?> list:
                    foreach (var item in list) stack.Push(item);
                    break;
                case IEnumerable<HostTensor> tensors: // ImageBatch[] and the like (never a float[] or a byte[]: those are walked by no one)
                    foreach (var item in tensors) stack.Push(item);
                    break;
            }
        }
    }

    /// <summary>Brings every tensor in <paramref name="values"/> to the host (each device tensor copied once).</summary>
    public static async Task ToHostAsync(IEnumerable<object?> values)
    {
        foreach (var t in In(values))
            if (!t.OnHost) await t.ToHostAsync().ConfigureAwait(false);
    }
}
