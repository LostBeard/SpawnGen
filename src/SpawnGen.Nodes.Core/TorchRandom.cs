namespace SpawnGen.Nodes.Core;

/// <summary>
/// PyTorch's CPU generator: <c>torch.manual_seed(seed)</c> then <c>torch.randn(n)</c> in float32, so nodes that draw noise on the CPU
/// (ComfyUI's Add Noise to Image) give ComfyUI's numbers. The engine is mt19937 seeded with the low 32 bits of the seed.
/// </summary>
public sealed class TorchRandom
{
    readonly uint[] _mt = new uint[624];
    int _index = 624;

    public TorchRandom(long seed)
    {
        _mt[0] = (uint)seed;
        for (uint i = 1; i < 624; i++) _mt[i] = 1812433253u * (_mt[i - 1] ^ (_mt[i - 1] >> 30)) + i;
    }

    /// <summary>The next 32 random bits (mt19937 with the standard tempering).</summary>
    public uint NextUInt32()
    {
        if (_index >= 624)
        {
            for (int k = 0; k < 624; k++)
            {
                uint y = (_mt[k] & 0x80000000u) | (_mt[(k + 1) % 624] & 0x7fffffffu);
                _mt[k] = _mt[(k + 397) % 624] ^ (y >> 1) ^ ((y & 1) != 0 ? 0x9908b0dfu : 0u);
            }
            _index = 0;
        }
        uint v = _mt[_index++];
        v ^= v >> 11;
        v ^= (v << 7) & 0x9d2c5680u;
        v ^= (v << 15) & 0xefc60000u;
        v ^= v >> 18;
        return v;
    }

    /// <summary><c>uniform_real_distribution&lt;float&gt;(0, 1)</c>: 24 random bits over 2^24.</summary>
    float NextFloat() => (NextUInt32() & 0xFFFFFF) * (1f / (1 << 24));

    /// <summary><c>uniform_real_distribution&lt;double&gt;(0, 1)</c>: 53 bits of two draws (first draw high) over 2^53.</summary>
    double NextDouble()
    {
        ulong hi = NextUInt32(), lo = NextUInt32();
        return ((hi << 32 | lo) & ((1UL << 53) - 1)) * (1.0 / (1UL << 53));
    }

    /// <summary>
    /// <c>torch.randn(n)</c> on a freshly seeded generator. 16 or more values: normal_fill (all uniforms first, Box-Muller in blocks of 16,
    /// the last partial block redrawn); fewer: the scalar double-precision normal_distribution, whose second value is cached for the next draw.
    /// </summary>
    public static float[] Randn(long seed, int n)
    {
        var g = new TorchRandom(seed);
        var data = new float[n];
        if (n >= 16)
        {
            for (int i = 0; i < n; i++) data[i] = g.NextFloat();
            for (int i = 0; i + 16 <= n; i += 16) NormalFill16(data, i);
            if (n % 16 != 0)
            {
                for (int i = n - 16; i < n; i++) data[i] = g.NextFloat();
                NormalFill16(data, n - 16);
            }
            return data;
        }

        double? cached = null;
        for (int i = 0; i < n; i++)
        {
            if (cached is { } next) { data[i] = (float)next; cached = null; continue; }
            double u1 = g.NextDouble(), u2 = g.NextDouble();
            double r = Math.Sqrt(-2.0 * Math.Log(1.0 - u2)), theta = 2.0 * Math.PI * u1;
            cached = r * Math.Sin(theta);
            data[i] = (float)(r * Math.Cos(theta));
        }
        return data;
    }

    // normal_fill_16: u1 = 1 - data[j] (so log never sees 0), u2 = data[j + 8]; radius and angle in float, theta through double pi.
    static void NormalFill16(float[] data, int o)
    {
        for (int j = 0; j < 8; j++)
        {
            float u1 = 1 - data[o + j], u2 = data[o + j + 8];
            float radius = MathF.Sqrt(-2 * MathF.Log(u1));
            float theta = (float)(2.0f * Math.PI * u2);
            data[o + j] = radius * MathF.Cos(theta);
            data[o + j + 8] = radius * MathF.Sin(theta);
        }
    }
}

/// <summary>
/// numpy's legacy generator as ComfyUI's dataset nodes use it (<c>np.random.seed(seed)</c>, then <c>randint</c> or <c>permutation</c>): the same
/// mt19937 as <see cref="TorchRandom"/>, seeded the same way (init_genrand with the 32-bit seed), with numpy's bounded draws.
/// </summary>
public sealed class NumpyRandom
{
    readonly TorchRandom _mt;

    /// <summary><c>np.random.seed(seed)</c>; numpy takes 0 to 2^32 - 1.</summary>
    public NumpyRandom(long seed)
    {
        if (seed is < 0 or > uint.MaxValue) throw new ArgumentOutOfRangeException(nameof(seed), "Seed must be between 0 and 2**32 - 1");
        _mt = new TorchRandom(seed);
    }

    /// <summary><c>randint(low, high)</c>, high excluded (ranges below 2^32: 32-bit draws, masked and redrawn until in range; an empty range draws nothing).</summary>
    public long RandInt(long low, long high)
    {
        if (high <= low) throw new ArgumentException("low >= high");
        ulong range = (ulong)(high - 1 - low);
        if (range == 0) return low;
        if (range > uint.MaxValue) throw new NotSupportedException("Ranges of 2^32 and more draw 64-bit numbers; the dataset nodes never need them.");
        return low + Interval((uint)range);
    }

    /// <summary><c>permutation(n)</c>: <c>arange(n)</c> shuffled by Fisher-Yates from the end, each swap partner drawn from 0..i.</summary>
    public int[] Permutation(int n)
    {
        var order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        for (int i = n - 1; i > 0; i--)
        {
            int j = (int)Interval((uint)i);
            (order[i], order[j]) = (order[j], order[i]);
        }
        return order;
    }

    /// <summary>numpy's random_interval / bounded masked draw: 0..max, drawing 32 bits masked to max's bit length until the value fits.</summary>
    uint Interval(uint max)
    {
        if (max == 0) return 0;
        uint mask = max;
        mask |= mask >> 1; mask |= mask >> 2; mask |= mask >> 4; mask |= mask >> 8; mask |= mask >> 16;
        uint value;
        while ((value = _mt.NextUInt32() & mask) > max) { }
        return value;
    }
}
