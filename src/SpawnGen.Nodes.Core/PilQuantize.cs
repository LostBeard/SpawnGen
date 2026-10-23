namespace SpawnGen.Nodes.Core;

/// <summary>
/// Pillow's colour quantisation, as ComfyUI's Quantize node uses it: <c>Image.quantize(colors)</c> on an RGB picture (libImaging/Quant.c, median cut,
/// with its colour hash from QuantHash.c and box heap from QuantHeap.c) and <c>Image.quantize(palette=..., dither=...)</c> (Convert.c topalette with the
/// palette cache of Palette.c, and its Floyd-Steinberg). Pictures are packed 8-bit RGB, three bytes a pixel, rows top to bottom.
/// Everything is ported operation for operation, the orders of the hash table, the merge sort and the heap included, because they decide ties.
/// </summary>
public static class PilQuantize
{
    /// <summary>
    /// The palette <c>quantize(colors)</c> builds (median cut, method 0, no k-means): packed RGB, one entry per median-cut box, so it has fewer than
    /// <paramref name="colors"/> entries when the picture has fewer colours. Pillow maps the picture's own pixels too; ComfyUI only keeps the palette.
    /// </summary>
    public static byte[] MedianCutPalette(byte[] rgb, int colors)
    {
        if (colors is < 1 or > 256) throw new ArgumentOutOfRangeException(nameof(colors), "bad number of colors");
        int n = rgb.Length / 3;
        if (n == 0) throw new ArgumentException("quantization error", nameof(rgb));
        var px = new uint[n];
        for (int i = 0; i < n; i++) px[i] = rgb[i * 3] | (uint)rgb[i * 3 + 1] << 8 | (uint)rgb[i * 3 + 2] << 16 | 0xFF000000u;

        // 1. Count the colours; past 65536 entries the hash drops one more low bit of every channel (d->scale) and merges what then collides.
        var hash = new PixelHash();
        for (int i = 0; i < n; i++)
        {
            hash.CountPixel(px[i]);
            while (hash.Count > 65536)
            {
                hash.Scale++;
                hash.RehashSumming();
            }
        }

        // 2. One linked list per channel of the (scaled) colours, each sorted high to low.
        var heads = new PixelList?[3];
        hash.ForEach((key, count) =>
        {
            var p = new PixelList(Scaled(key, hash.Scale), (int)count);
            for (int i = 0; i < 3; i++)
            {
                p.Next[i] = heads[i];
                p.Prev[i] = null;
                if (heads[i] is { } first) first.Prev[i] = p;
                heads[i] = p;
            }
        });
        for (int i = 0; i < 3; i++) heads[i] = MergeSort(heads[i], i);

        // 3. Median cut, 4. every colour labelled with its box, 5. each box's mean over all the pixels in it, rounded half up.
        var root = MedianCut(heads, (uint)n, colors);
        uint entries = 0;
        Annotate(root, hash, ref entries);
        var sum = new uint[entries * 3];
        var count = new uint[entries];
        for (int i = 0; i < n; i++)
        {
            if (!hash.Lookup(px[i], out uint box) || box >= entries) throw new InvalidOperationException("quantization error");
            unchecked
            {
                sum[box * 3] += px[i] & 0xFF;
                sum[box * 3 + 1] += px[i] >> 8 & 0xFF;
                sum[box * 3 + 2] += px[i] >> 16 & 0xFF;
            }
            count[box]++;
        }
        var palette = new byte[entries * 3];
        for (int i = 0; i < palette.Length; i++) palette[i] = (byte)(int)(.5 + (double)sum[i] / count[i / 3]);
        return palette;
    }

    /// <summary>
    /// <c>quantize(palette=...)</c> of an RGB picture: the palette index of every pixel, the nearest entry by squared distance (Pillow's cache answers for
    /// the low corner of each 4x4x4 cell of colours, the lowest index winning a tie), with Floyd-Steinberg error diffusion when <paramref name="dither"/>.
    /// </summary>
    public static byte[] ToPalette(byte[] rgb, int w, int h, byte[] palette, bool dither)
    {
        int size = palette.Length / 3;
        if (size == 0) throw new ArgumentException("empty palette", nameof(palette));
        var cache = new short[64 * 64 * 64];
        Array.Fill(cache, (short)0x100);
        int Nearest(int r, int g, int b)
        {
            int slot = (r >> 2) + (g >> 2) * 64 + (b >> 2) * 64 * 64;
            if (cache[slot] == 0x100) CacheUpdate(cache, palette, size, r, g, b);
            return cache[slot];
        }

        var o = new byte[w * h];
        if (!dither)
        {
            for (int i = 0; i < o.Length; i++) o[i] = (byte)Nearest(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
            return o;
        }

        // Floyd-Steinberg as Convert.c writes it: left to right on every row, errors in sixteenths kept in one row buffer. The row's last slot gets
        // the blue channel's three running values (b0, b1, b2) for all three channels, as Pillow does.
        var errors = new int[(w + 1) * 3];
        for (int y = 0; y < h; y++)
        {
            int r = 0, r0 = 0, r1 = 0, r2;
            int g = 0, g0 = 0, g1 = 0, g2;
            int b = 0, b0 = 0, b1 = 0, b2 = 0;
            int e = 0;
            for (int x = 0; x < w; x++, e += 3)
            {
                int p = (y * w + x) * 3;
                r = Clip8(rgb[p] + (r + errors[e + 3]) / 16);
                g = Clip8(rgb[p + 1] + (g + errors[e + 4]) / 16);
                b = Clip8(rgb[p + 2] + (b + errors[e + 5]) / 16);
                int index = Nearest(r, g, b);
                o[y * w + x] = (byte)index;
                r -= palette[index * 3];
                g -= palette[index * 3 + 1];
                b -= palette[index * 3 + 2];

                // 7/16 to the right (carried in r, g, b), 3/16 below left, 5/16 below, 1/16 below right.
                int d2;
                r2 = r; d2 = r + r; r += d2; errors[e] = r + r0; r += d2; r0 = r + r1; r1 = r2; r += d2;
                g2 = g; d2 = g + g; g += d2; errors[e + 1] = g + g0; g += d2; g0 = g + g1; g1 = g2; g += d2;
                b2 = b; d2 = b + b; b += d2; errors[e + 2] = b + b0; b += d2; b0 = b + b1; b1 = b2; b += d2;
            }
            errors[e] = b0;
            errors[e + 1] = b1;
            errors[e + 2] = b2;
        }
        return o;
    }

    static int Clip8(int v) => v <= 0 ? 0 : v < 256 ? v : 255;

    /// <summary>ImagingPaletteCacheUpdate: fills the 8x8x8 cache slots (a 32-value box) around (r, g, b) with their nearest palette entries.</summary>
    static void CacheUpdate(short[] cache, byte[] palette, int size, int r, int g, int b)
    {
        int r0 = r & 0xe0, r1 = r0 + 0x1f, rc = (r0 + r1) / 2;
        int g0 = g & 0xe0, g1 = g0 + 0x1f, gc = (g0 + g1) / 2;
        int b0 = b & 0xe0, b1 = b0 + 0x1f, bc = (b0 + b1) / 2;
        static uint Dist(int a, int c) => (uint)((a - c) * (a - c));

        // Step 1: only entries whose nearest point of the box is no farther than the smallest farthest point of any entry can win.
        var dmin = new uint[size];
        uint dmax = uint.MaxValue;
        for (int i = 0; i < size; i++)
        {
            int pr = palette[i * 3], pg = palette[i * 3 + 1], pb = palette[i * 3 + 2];
            uint tmin = pr < r0 ? Dist(pr, r0) : pr > r1 ? Dist(pr, r1) : 0;
            uint tmax = pr <= rc ? Dist(pr, r1) : Dist(pr, r0);
            tmin += pg < g0 ? Dist(pg, g0) : pg > g1 ? Dist(pg, g1) : 0;
            tmax += pg <= gc ? Dist(pg, g1) : Dist(pg, g0);
            tmin += pb < b0 ? Dist(pb, b0) : pb > b1 ? Dist(pb, b1) : 0;
            tmax += pb <= bc ? Dist(pb, b1) : Dist(pb, b0);
            dmin[i] = tmin;
            if (tmax < dmax) dmax = tmax;
        }

        // Step 2: incremental distances to the low corner of every slot (steps of 4).
        const int Box = 8, Step = 4;
        var d = new uint[Box * Box * Box];
        var c = new byte[Box * Box * Box];
        Array.Fill(d, uint.MaxValue);
        for (int i = 0; i < size; i++)
        {
            if (dmin[i] > dmax) continue;
            int ri = r0 - palette[i * 3], gi = g0 - palette[i * 3 + 1], bi = b0 - palette[i * 3 + 2];
            int rd = ri * ri + gi * gi + bi * bi;
            ri = ri * (2 * Step) + Step * Step;
            gi = gi * (2 * Step) + Step * Step;
            bi = bi * (2 * Step) + Step * Step;
            int rx = ri, j = 0;
            for (int ir = 0; ir < Box; ir++)
            {
                int gd = rd, gx = gi;
                for (int ig = 0; ig < Box; ig++)
                {
                    int bd = gd, bx = bi;
                    for (int ib = 0; ib < Box; ib++)
                    {
                        if ((uint)bd < d[j]) { d[j] = (uint)bd; c[j] = (byte)i; }
                        bd += bx;
                        bx += 2 * Step * Step;
                        j++;
                    }
                    gd += gx;
                    gx += 2 * Step * Step;
                }
                rd += rx;
                rx += 2 * Step * Step;
            }
        }

        // Step 3: store.
        int k = 0;
        for (int sr = r0; sr < r1; sr += 4)
            for (int sg = g0; sg < g1; sg += 4)
                for (int sb = b0; sb < b1; sb += 4)
                    cache[(sr >> 2) + (sg >> 2) * 64 + (sb >> 2) * 64 * 64] = c[k++];
    }

    // Pixel (Quant.c's union): r in the low byte, then g, b, a.
    static uint Scaled(uint key, int s) => (key & 0xFF) >> s | ((key >> 8 & 0xFF) >> s) << 8 | ((key >> 16 & 0xFF) >> s) << 16;
    static uint Unscaled(uint key, int s) => (key & 0xFF) << s & 0xFF | ((key >> 8 & 0xFF) << s & 0xFF) << 8 | ((key >> 16 & 0xFF) << s & 0xFF) << 16;

    sealed class PixelList(uint p, int count)
    {
        public readonly PixelList?[] Next = new PixelList?[3], Prev = new PixelList?[3];
        public readonly uint P = p;
        public bool Flag;
        public readonly int Count = count;
        public int V(int axis) => (int)(P >> (8 * axis) & 0xFF);
    }

    sealed class BoxNode
    {
        public BoxNode? L, R;
        public readonly PixelList?[] Head = new PixelList?[3], Tail = new PixelList?[3];
        public int Axis = -1, Volume = -1;
        public uint PixelCount;
    }

    /// <summary>mergesort_pixels: top-down on the linked list, the first half the longer one, descending; a tie takes the second half's element first.</summary>
    static PixelList? MergeSort(PixelList? head, int i)
    {
        if (head is null || head.Next[i] is null)
        {
            if (head is not null) { head.Next[i] = null; head.Prev[i] = null; }
            return head;
        }
        PixelList? c = head, t = head;
        while (c is not null && t is not null)
        {
            c = c.Next[i];
            t = t.Next[i]?.Next[i];
        }
        if (c is not null)
        {
            if (c.Prev[i] is { } before) before.Next[i] = null;
            c.Prev[i] = null;
        }
        var a = MergeSort(head, i);
        var b = MergeSort(c, i);
        head = null;
        PixelList? p = null;
        while (a is not null && b is not null)
        {
            if (a.V(i) > b.V(i)) { c = a; a = a.Next[i]; }
            else { c = b; b = b.Next[i]; }
            c.Prev[i] = p;
            c.Next[i] = null;
            if (p is not null) p.Next[i] = c;
            p = c;
            head ??= c;
        }
        if (a is not null) { c!.Next[i] = a; a.Prev[i] = c; }
        else if (b is not null) { c!.Next[i] = b; b.Prev[i] = c; }
        return head;
    }

    static int BoxVolume(BoxNode b)
    {
        if (b.Volume >= 0) return b.Volume;
        if (b.Head[0] is null) b.Volume = 0;
        else
            b.Volume = (b.Head[0]!.V(0) - b.Tail[0]!.V(0) + 1) * (b.Head[1]!.V(1) - b.Tail[1]!.V(1) + 1) * (b.Head[2]!.V(2) - b.Tail[2]!.V(2) + 1);
        return b.Volume;
    }

    /// <summary>median_cut: the box with the most pixels (QuantHeap order) is split next; boxes of one colour are dropped from the heap.</summary>
    static BoxNode MedianCut(PixelList?[] hl, uint pixelCount, int colors)
    {
        var heap = new BoxHeap();
        var root = new BoxNode { PixelCount = pixelCount };
        for (int i = 0; i < 3; i++)
        {
            var t = hl[i];
            while (t is not null && t.Next[i] is not null) t = t.Next[i];
            root.Head[i] = hl[i];
            root.Tail[i] = t;
        }
        heap.Add(root);
        while (--colors != 0)
        {
            BoxNode? node;
            do
            {
                if (!heap.Remove(out node)) return root;
            } while (BoxVolume(node!) == 1);
            Split(node!);
            heap.Add(node!.L!);
            heap.Add(node.R!);
        }
        return root;
    }

    /// <summary>split: along the channel with the widest luminance-weighted range (77 r, 150 g, 29 b; the first on a tie).</summary>
    static void Split(BoxNode node)
    {
        int[] f =
        [
            (node.Head[0]!.V(0) - node.Tail[0]!.V(0)) * 77,
            (node.Head[1]!.V(1) - node.Tail[1]!.V(1)) * 150,
            (node.Head[2]!.V(2) - node.Tail[2]!.V(2)) * 29,
        ];
        int best = f[0], axis = 0;
        for (int i = 1; i < 3; i++) if (best < f[i]) { best = f[i]; axis = i; }
        node.Axis = axis;
        var (heads, tails, counts) = SplitLists(node.Head, node.Tail, axis, node.PixelCount);
        var left = new BoxNode { PixelCount = counts[0] };
        var right = new BoxNode { PixelCount = counts[1] };
        for (int i = 0; i < 3; i++)
        {
            left.Head[i] = heads[0, i]; left.Tail[i] = tails[0, i];
            right.Head[i] = heads[1, i]; right.Tail[i] = tails[1, i];
            node.Head[i] = null; node.Tail[i] = null;
        }
        node.L = left;
        node.R = right;
    }

    /// <summary>
    /// splitlists: walk the sorted axis until more than half the pixels are on the left, keep the rest of that value on the left too, put the others
    /// right (if none are left over, the lowest value goes right instead), then split all three lists keeping their order.
    /// </summary>
    static (PixelList?[,], PixelList?[,], uint[]) SplitLists(PixelList?[] h, PixelList?[] t, int axis, uint pixelCount)
    {
        var nCount = new uint[2];
        int nRight = 0;
        uint left = 0;
        var c = h[axis];
        while (c is not null)
        {
            left += (uint)c.Count;
            nCount[0] += (uint)c.Count;
            c.Flag = false;
            c = c.Next[axis];
            if (left * 2 > pixelCount) break;
        }
        if (c is not null)
        {
            int splitColourVal = c.Prev[axis]!.V(axis);
            for (; c is not null; c = c.Next[axis])
            {
                if (splitColourVal != c.V(axis)) break;
                c.Flag = false;
                nCount[0] += (uint)c.Count;
            }
        }
        for (; c is not null; c = c.Next[axis])
        {
            c.Flag = true;
            nRight++;
            nCount[1] += (uint)c.Count;
        }
        if (nRight == 0)
        {
            int splitColourVal = t[axis]!.V(axis);
            for (c = t[axis]; c is not null; c = c.Prev[axis])
            {
                if (splitColourVal != c.V(axis)) break;
                c.Flag = true;
                nRight++;
                nCount[0] -= (uint)c.Count;
                nCount[1] += (uint)c.Count;
            }
        }

        var nh = new PixelList?[2, 3];
        var nt = new PixelList?[2, 3];
        for (int i = 0; i < 3; i++)
        {
            PixelList? l = null, r = null, next;
            for (c = h[i]; c is not null; c = next)
            {
                next = c.Next[i];
                if (c.Flag)
                {
                    if (r is not null) r.Next[i] = c; else nh[1, i] = c;
                    c.Prev[i] = r;
                    r = c;
                }
                else
                {
                    if (l is not null) l.Next[i] = c; else nh[0, i] = c;
                    c.Prev[i] = l;
                    l = c;
                }
            }
            if (l is not null) l.Next[i] = null;
            if (r is not null) r.Next[i] = null;
            nt[0, i] = l;
            nt[1, i] = r;
        }
        return (nh, nt, nCount);
    }

    /// <summary>annotate_hash_table: numbers the leaf boxes left to right and labels every colour in the hash with its box.</summary>
    static void Annotate(BoxNode n, PixelHash hash, ref uint box)
    {
        if (n.L is not null && n.R is not null)
        {
            Annotate(n.L, hash, ref box);
            Annotate(n.R, hash, ref box);
            return;
        }
        if (n.L is not null || n.R is not null) throw new InvalidOperationException("quantization error");
        for (var p = n.Head[0]; p is not null; p = p.Next[0]) hash.Insert(Unscaled(p.P, hash.Scale), box);
        if (n.Head[0] is not null) box++;
    }

    /// <summary>QuantHeap: a binary max-heap on pixel count, 1-based, with Pillow's sift-up (stops at equal) and sift-down (moves past equal).</summary>
    sealed class BoxHeap
    {
        readonly List<BoxNode?> heap = [null];
        int count;

        static int Cmp(BoxNode a, BoxNode b) => (int)a.PixelCount - (int)b.PixelCount;

        public bool Remove(out BoxNode? r)
        {
            r = null;
            if (count == 0) return false;
            r = heap[1];
            var v = heap[count--]!;
            int k, l;
            for (k = 1; k * 2 <= count; k = l)
            {
                l = k * 2;
                if (l < count && Cmp(heap[l]!, heap[l + 1]!) < 0) l++;
                if (Cmp(v, heap[l]!) > 0) break;
                heap[k] = heap[l];
            }
            heap[k] = v;
            return true;
        }

        public void Add(BoxNode val)
        {
            int k = ++count;
            if (heap.Count <= k) heap.Add(null);
            while (k != 1)
            {
                if (Cmp(val, heap[k / 2]!) <= 0) break;
                heap[k] = heap[k / 2];
                k >>= 1;
            }
            heap[k] = val;
        }
    }

    /// <summary>
    /// QuantHash with Quant.c's pixel_hash / pixel_cmp: chained buckets kept sorted by the hash of the scaled colour (equal hashes are the same key),
    /// grown and shrunk by a factor of three like Pillow, whose iteration order (bucket, then chain) decides the order of the colour lists.
    /// </summary>
    sealed class PixelHash
    {
        sealed class Node
        {
            public Node? Next;
            public uint Key, Value;
        }

        Node?[] table = new Node?[11];
        uint length = 11;
        public uint Count;
        public int Scale;

        uint Hash(uint key)
        {
            uint r = (key & 0xFF) >> Scale, g = (key >> 8 & 0xFF) >> Scale, b = (key >> 16 & 0xFF) >> Scale;
            return unchecked(r * 463 ^ (g << 8) * 10069 ^ (b << 16) * 64997);
        }

        int Cmp(uint a, uint b)
        {
            uint ha = Hash(a), hb = Hash(b);
            return ha == hb ? 0 : ha < hb ? -1 : 1;
        }

        /// <summary>hashtable_insert_or_update_computed with new_count_func / exists_count_func.</summary>
        public void CountPixel(uint key)
        {
            ref Node? slot = ref table[Hash(key) % length];
            uint hk = Hash(key);
            Node? prev = null;
            for (var n = slot; n is not null; prev = n, n = n.Next)
            {
                uint hn = Hash(n.Key);
                if (hn == hk) { n.Value++; return; }
                if (hn > hk) break;
            }
            var t = new Node { Key = key, Value = 1 };
            Link(ref slot, prev, t);
            Count++;
            Resize();
        }

        /// <summary>hashtable_insert (resizing): replaces the value of an equal key, else adds it.</summary>
        public void Insert(uint key, uint value)
        {
            ref Node? slot = ref table[Hash(key) % length];
            Node? prev = null;
            for (var n = slot; n is not null; prev = n, n = n.Next)
            {
                int i = Cmp(n.Key, key);
                if (i == 0) { n.Value = value; return; }
                if (i > 0) break;
            }
            Link(ref slot, prev, new Node { Key = key, Value = value });
            Count++;
            Resize();
        }

        public bool Lookup(uint key, out uint value)
        {
            for (var n = table[Hash(key) % length]; n is not null; n = n.Next)
            {
                int i = Cmp(n.Key, key);
                if (i == 0) { value = n.Value; return true; }
                if (i > 0) break;
            }
            value = 0;
            return false;
        }

        public void ForEach(Action<uint, uint> f)
        {
            for (uint x = 0; x < length; x++)
                for (var n = table[x]; n is not null; n = n.Next) f(n.Key, n.Value);
        }

        /// <summary>hashtable_rehash_compute(rehash_collide): same length, under the current scale; colliding keys keep the later key and add the counts.</summary>
        public void RehashSumming() => Rehash(length, summing: true);

        static void Link(ref Node? slot, Node? prev, Node node)
        {
            if (prev is null) { node.Next = slot; slot = node; }
            else { node.Next = prev.Next; prev.Next = node; }
        }

        void Resize()
        {
            uint newSize = length;
            if (Count * 3 < length) newSize = FindPrime(length / 2 - 1, -1);
            else if (length * 3 < Count) newSize = FindPrime(length * 2 + 1, +1);
            if (newSize < 11) newSize = length;
            if (newSize != length) Rehash(newSize, summing: false);
        }

        /// <summary>_findPrime: its divisor test reads <c>!start % t</c>, which is never true, so it only steps to a number ending in 1, 3, 7, 9, 11 or 13 (mod 16).</summary>
        static uint FindPrime(uint start, int dir)
        {
            ReadOnlySpan<int> unit = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0];
            while (start > 1 && unit[(int)(start & 0x0f)] == 0) start = (uint)(start + dir);
            return start;
        }

        /// <summary>_hashtable_rehash: every old bucket in order, every chain in order, re-inserted (sorted) into the new table.</summary>
        void Rehash(uint newSize, bool summing)
        {
            var old = table;
            table = new Node?[newSize];
            length = newSize;
            Count = 0;
            foreach (var chain in old)
            {
                Node? next;
                for (var node = chain; node is not null; node = next)
                {
                    next = node.Next;
                    InsertNode(node, summing);
                }
            }
        }

        void InsertNode(Node node, bool summing)
        {
            ref Node? slot = ref table[Hash(node.Key) % length];
            Node? prev = null;
            for (var n = slot; n is not null; prev = n, n = n.Next)
            {
                int i = Cmp(n.Key, node.Key);
                if (i == 0)
                {
                    n.Key = node.Key;
                    n.Value = summing ? n.Value + node.Value : node.Value;
                    return;
                }
                if (i > 0) break;
            }
            Link(ref slot, prev, node);
            Count++;
        }
    }
}
