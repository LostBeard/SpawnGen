using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// JPEG decoder with no dependencies: baseline, extended-sequential and progressive Huffman files, 8 bit, greyscale or YCbCr (or RGB marked by Adobe), any chroma subsampling
/// (chroma is upsampled with a triangle filter like libjpeg's default), restart intervals, and the EXIF orientation (applied, as ComfyUI's loader does). Arithmetic coding,
/// 12-bit samples and CMYK are rejected with a clear message. The result is an RGB <see cref="ImageBatch"/> in 0..1.
/// </summary>
public static class JpegReader
{
    public static bool IsJpeg(ReadOnlySpan<byte> data) => data.Length > 3 && data[0] == 0xFF && data[1] == 0xD8;

    public static ImageBatch Decode(byte[] data) => new Decoder(data, applyOrientation: true).Decode();

    /// <summary>Decodes with or without the EXIF orientation (Pillow's <c>Image.open</c> leaves it alone; ComfyUI's Load Image applies it).</summary>
    public static ImageBatch Decode(byte[] data, bool applyOrientation) => new Decoder(data, applyOrientation).Decode();

    static readonly byte[] ZigZag =
    [
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
    ];

    sealed class Huffman
    {
        public readonly int[] MaxCode = new int[18], MinCode = new int[17], ValPtr = new int[17];
        public readonly byte[] Values;

        public Huffman(byte[] counts, byte[] values)
        {
            Values = values;
            int code = 0, k = 0;
            for (int l = 1; l <= 16; l++)
            {
                ValPtr[l] = k;
                MinCode[l] = code;
                code += counts[l - 1];
                k += counts[l - 1];
                MaxCode[l] = counts[l - 1] == 0 ? -1 : code - 1;
                code <<= 1;
            }
            MaxCode[17] = int.MaxValue;
        }
    }

    sealed class Component
    {
        public int Id, H, V, Tq;
        public int BlocksPerLine, BlocksPerColumn;           // blocks that hold picture content (non-interleaved scans)
        public int BlocksPerLineForMcu, BlocksPerColumnForMcu; // blocks stored (whole MCUs)
        public short[] Coefficients = [];
        public Huffman? Dc, Ac;
        public int Pred;
    }

    sealed class Decoder(byte[] d, bool applyOrientation)
    {
        int _pos;
        int _width, _height, _hMax, _vMax;
        bool _progressive;
        int _restartInterval;
        int _adobeTransform = -1, _orientation = 1;
        readonly ushort[][] _quant = new ushort[4][];
        readonly Huffman?[] _dcTables = new Huffman?[4], _acTables = new Huffman?[4];
        Component[] _components = [];
        // bit reader
        int _bitBuffer, _bitCount;
        bool _hitMarker;
        int _eobRun;

        int U16(int p) => (d[p] << 8) | d[p + 1];

        public ImageBatch Decode()
        {
            if (!IsJpeg(d)) throw new InvalidDataException("Not a JPEG file.");
            _pos = 2;
            bool sawFrame = false, sawScan = false;
            while (_pos < d.Length - 1)
            {
                if (d[_pos] != 0xFF) { _pos++; continue; }
                int marker = d[_pos + 1];
                if (marker == 0xFF) { _pos++; continue; }
                _pos += 2;
                if (marker == 0xD8 || (marker >= 0xD0 && marker <= 0xD7) || marker == 0x01 || marker == 0) continue;
                if (marker == 0xD9) break;
                int len = U16(_pos), body = _pos + 2, end = _pos + len;
                if (end > d.Length) throw new InvalidDataException("JPEG is truncated.");
                switch (marker)
                {
                    case 0xE1: ReadExif(body, end); break;
                    case 0xEE:
                        if (end - body >= 12 && d[body] == 'A' && d[body + 1] == 'd' && d[body + 2] == 'o' && d[body + 3] == 'b' && d[body + 4] == 'e') _adobeTransform = d[body + 11];
                        break;
                    case 0xDB: ReadQuant(body, end); break;
                    case 0xC0: case 0xC1: case 0xC2:
                        _progressive = marker == 0xC2;
                        ReadFrame(body);
                        sawFrame = true;
                        break;
                    case 0xC3: case 0xC5: case 0xC6: case 0xC7: case 0xC9: case 0xCA: case 0xCB: case 0xCD: case 0xCE: case 0xCF:
                        throw new NotSupportedException(marker >= 0xC9 ? "Arithmetic-coded JPEG is not supported." : "Lossless and hierarchical JPEG are not supported.");
                    case 0xC4: ReadHuffman(body, end); break;
                    case 0xDD: _restartInterval = U16(body); break;
                    case 0xDA:
                        if (!sawFrame) throw new InvalidDataException("JPEG scan before its frame header.");
                        _pos = end;
                        ReadScan(body);
                        sawScan = true;
                        continue; // _pos is already past the entropy-coded data
                }
                _pos = end;
            }
            if (!sawFrame || !sawScan) throw new InvalidDataException("JPEG has no picture data.");
            return Output();
        }

        void ReadExif(int p, int end)
        {
            // "Exif\0\0" then a TIFF header; only the orientation tag of the first IFD matters.
            if (end - p < 14 || d[p] != 'E' || d[p + 1] != 'x' || d[p + 2] != 'i' || d[p + 3] != 'f') return;
            int t = p + 6;
            bool little = d[t] == 'I';
            int U16t(int o) => little ? d[o] | (d[o + 1] << 8) : (d[o] << 8) | d[o + 1];
            int U32t(int o) => little ? d[o] | (d[o + 1] << 8) | (d[o + 2] << 16) | (d[o + 3] << 24) : (d[o] << 24) | (d[o + 1] << 16) | (d[o + 2] << 8) | d[o + 3];
            if (U16t(t + 2) != 42) return;
            int ifd = t + U32t(t + 4);
            if (ifd < t || ifd + 2 > end) return;
            int entries = U16t(ifd);
            for (int i = 0; i < entries && ifd + 2 + (i + 1) * 12 <= end; i++)
            {
                int e = ifd + 2 + i * 12;
                if (U16t(e) == 0x0112) { int v = U16t(e + 8); if (v is >= 1 and <= 8) _orientation = v; return; }
            }
        }

        void ReadQuant(int p, int end)
        {
            while (p < end)
            {
                int pq = d[p] >> 4, tq = d[p] & 15;
                p++;
                if (tq > 3) throw new InvalidDataException("Bad JPEG quantisation table index.");
                var table = new ushort[64];
                for (int i = 0; i < 64; i++)
                {
                    table[ZigZag[i]] = pq == 0 ? d[p] : (ushort)U16(p);
                    p += pq == 0 ? 1 : 2;
                }
                _quant[tq] = table;
            }
        }

        void ReadFrame(int p)
        {
            if (d[p] != 8) throw new NotSupportedException($"JPEG with {d[p]}-bit samples is not supported (8 is).");
            _height = U16(p + 1);
            _width = U16(p + 3);
            int n = d[p + 5];
            if (_width == 0 || _height == 0) throw new InvalidDataException("JPEG has no size (DNL is not supported).");
            if (n is not (1 or 3)) throw new NotSupportedException($"JPEG with {n} components is not supported (greyscale and colour are).");
            _components = new Component[n];
            for (int i = 0; i < n; i++)
            {
                int o = p + 6 + i * 3;
                _components[i] = new Component { Id = d[o], H = d[o + 1] >> 4, V = d[o + 1] & 15, Tq = d[o + 2] };
                if (_components[i].H is < 1 or > 4 || _components[i].V is < 1 or > 4) throw new InvalidDataException("Bad JPEG sampling factors.");
            }
            _hMax = _components.Max(c => c.H);
            _vMax = _components.Max(c => c.V);
            int mcusPerLine = (_width + 8 * _hMax - 1) / (8 * _hMax), mcusPerColumn = (_height + 8 * _vMax - 1) / (8 * _vMax);
            foreach (var c in _components)
            {
                c.BlocksPerLine = ((_width * c.H + _hMax - 1) / _hMax + 7) / 8;
                c.BlocksPerColumn = ((_height * c.V + _vMax - 1) / _vMax + 7) / 8;
                c.BlocksPerLineForMcu = mcusPerLine * c.H;
                c.BlocksPerColumnForMcu = mcusPerColumn * c.V;
                c.Coefficients = new short[c.BlocksPerLineForMcu * c.BlocksPerColumnForMcu * 64];
            }
        }

        void ReadHuffman(int p, int end)
        {
            while (p < end)
            {
                int tc = d[p] >> 4, th = d[p] & 15;
                p++;
                var counts = d.AsSpan(p, 16).ToArray();
                p += 16;
                int total = counts.Sum(c => (int)c);
                var values = d.AsSpan(p, total).ToArray();
                p += total;
                if (th > 3) throw new InvalidDataException("Bad JPEG Huffman table index.");
                (tc == 0 ? _dcTables : _acTables)[th] = new Huffman(counts, values);
            }
        }

        // --- entropy-coded data ---

        int NextByte()
        {
            if (_hitMarker || _pos >= d.Length) return 0;
            int b = d[_pos];
            if (b == 0xFF)
            {
                int b2 = _pos + 1 < d.Length ? d[_pos + 1] : 0xD9;
                if (b2 == 0) { _pos += 2; return 0xFF; }
                _hitMarker = true; // a marker ends the segment; feed zeros until the caller moves on
                return 0;
            }
            _pos++;
            return b;
        }

        int Bit()
        {
            if (_bitCount == 0) { _bitBuffer = NextByte(); _bitCount = 8; }
            _bitCount--;
            return (_bitBuffer >> _bitCount) & 1;
        }

        int Receive(int n) { int v = 0; while (n-- > 0) v = (v << 1) | Bit(); return v; }

        static int Extend(int v, int n) => n == 0 ? 0 : v < (1 << (n - 1)) ? v - (1 << n) + 1 : v;

        int DecodeHuffman(Huffman h)
        {
            int code = 0;
            for (int l = 1; l <= 16; l++)
            {
                code = (code << 1) | Bit();
                if (h.MaxCode[l] >= 0 && code <= h.MaxCode[l] && code >= h.MinCode[l])
                    return h.Values[h.ValPtr[l] + code - h.MinCode[l]];
            }
            throw new InvalidDataException("Bad JPEG Huffman code.");
        }

        void ReadScan(int p)
        {
            int n = d[p];
            var scan = new Component[n];
            for (int i = 0; i < n; i++)
            {
                int id = d[p + 1 + i * 2], tables = d[p + 2 + i * 2];
                var c = _components.FirstOrDefault(x => x.Id == id) ?? throw new InvalidDataException("JPEG scan names an unknown component.");
                c.Dc = _dcTables[tables >> 4];
                c.Ac = _acTables[tables & 15];
                scan[i] = c;
            }
            int ss = d[p + 1 + n * 2], se = d[p + 2 + n * 2], ah = d[p + 3 + n * 2] >> 4, al = d[p + 3 + n * 2] & 15;
            if (!_progressive) { ss = 0; se = 63; ah = al = 0; }

            Action<Component, int> decodeBlock = !_progressive ? Baseline
                : ss == 0 ? (ah == 0 ? DcFirst(al) : DcRefine(al))
                : (ah == 0 ? AcFirst(ss, se, al) : AcRefine(ss, se, al));

            int total;
            int mcusPerLine = (_width + 8 * _hMax - 1) / (8 * _hMax);
            if (n == 1) total = scan[0].BlocksPerLine * scan[0].BlocksPerColumn;
            else total = mcusPerLine * ((_height + 8 * _vMax - 1) / (8 * _vMax));

            _bitCount = 0; _hitMarker = false; _eobRun = 0;
            foreach (var c in scan) c.Pred = 0;
            int interval = _restartInterval > 0 ? _restartInterval : total;
            int mcu = 0;
            while (mcu < total)
            {
                foreach (var c in scan) c.Pred = 0;
                _eobRun = 0;
                _bitCount = 0;
                for (int i = 0; i < interval && mcu < total; i++, mcu++)
                {
                    if (n == 1)
                    {
                        var c = scan[0];
                        int row = mcu / c.BlocksPerLine, col = mcu % c.BlocksPerLine;
                        decodeBlock(c, (row * c.BlocksPerLineForMcu + col) * 64);
                    }
                    else
                    {
                        int mrow = mcu / mcusPerLine, mcol = mcu % mcusPerLine;
                        foreach (var c in scan)
                            for (int v = 0; v < c.V; v++)
                                for (int h = 0; h < c.H; h++)
                                    decodeBlock(c, ((mrow * c.V + v) * c.BlocksPerLineForMcu + mcol * c.H + h) * 64);
                    }
                }
                // Move to the next restart marker (or the end of the scan).
                _bitCount = 0;
                if (!_hitMarker)
                    while (_pos < d.Length - 1 && !(d[_pos] == 0xFF && d[_pos + 1] != 0 && d[_pos + 1] != 0xFF)) _pos++;
                if (_pos + 1 < d.Length && d[_pos] == 0xFF && d[_pos + 1] >= 0xD0 && d[_pos + 1] <= 0xD7) { _pos += 2; _hitMarker = false; }
                else if (mcu < total) break; // data ended early: keep what was decoded
            }
            // Leave _pos at the marker that ended the scan.
            if (!(d[Math.Min(_pos, d.Length - 1)] == 0xFF))
                while (_pos < d.Length - 1 && !(d[_pos] == 0xFF && d[_pos + 1] != 0 && d[_pos + 1] != 0xFF && !(d[_pos + 1] >= 0xD0 && d[_pos + 1] <= 0xD7))) _pos++;
        }

        void Baseline(Component c, int offset)
        {
            int t = DecodeHuffman(c.Dc ?? throw NoTable());
            c.Pred += Extend(Receive(t), t);
            c.Coefficients[offset] = (short)c.Pred;
            var ac = c.Ac ?? throw NoTable();
            for (int k = 1; k < 64;)
            {
                int rs = DecodeHuffman(ac), s = rs & 15, r = rs >> 4;
                if (s == 0) { if (r < 15) break; k += 16; continue; }
                k += r;
                if (k > 63) break;
                c.Coefficients[offset + ZigZag[k]] = (short)Extend(Receive(s), s);
                k++;
            }
        }

        static InvalidDataException NoTable() => new("JPEG scan uses a Huffman table that was not defined.");

        Action<Component, int> DcFirst(int al) => (c, offset) =>
        {
            int t = DecodeHuffman(c.Dc ?? throw NoTable());
            c.Pred += Extend(Receive(t), t);
            c.Coefficients[offset] = (short)(c.Pred << al);
        };

        Action<Component, int> DcRefine(int al) => (c, offset) =>
        {
            if (Bit() == 1) c.Coefficients[offset] |= (short)(1 << al);
        };

        Action<Component, int> AcFirst(int ss, int se, int al) => (c, offset) =>
        {
            if (_eobRun > 0) { _eobRun--; return; }
            var ac = c.Ac ?? throw NoTable();
            for (int k = ss; k <= se;)
            {
                int rs = DecodeHuffman(ac), s = rs & 15, r = rs >> 4;
                if (s == 0)
                {
                    if (r < 15) { _eobRun = Receive(r) + (1 << r) - 1; break; }
                    k += 16;
                    continue;
                }
                k += r;
                if (k > 63) break;
                c.Coefficients[offset + ZigZag[k]] = (short)(Extend(Receive(s), s) * (1 << al));
                k++;
            }
        };

        Action<Component, int> AcRefine(int ss, int se, int al) => (c, offset) =>
        {
            int p1 = 1 << al, m1 = -1 << al;
            int k = ss;
            var co = c.Coefficients;
            var ac = c.Ac ?? throw NoTable();
            if (_eobRun == 0)
            {
                for (; k <= se; k++)
                {
                    int rs = DecodeHuffman(ac), r = rs >> 4, s = rs & 15;
                    if (s != 0) s = Bit() == 1 ? p1 : m1;
                    else if (r != 15) { _eobRun = (1 << r) + (r > 0 ? Receive(r) : 0); break; }
                    // Skip r still-zero coefficients, refining the nonzero ones passed on the way.
                    do
                    {
                        int idx = offset + ZigZag[k];
                        if (co[idx] != 0)
                        {
                            if (Bit() == 1 && (co[idx] & p1) == 0) co[idx] = (short)(co[idx] >= 0 ? co[idx] + p1 : co[idx] + m1);
                        }
                        else if (--r < 0) break;
                        k++;
                    } while (k <= se);
                    if (s != 0 && k <= se) co[offset + ZigZag[k]] = (short)s;
                }
            }
            if (_eobRun > 0)
            {
                for (; k <= se; k++)
                {
                    int idx = offset + ZigZag[k];
                    if (co[idx] != 0 && Bit() == 1 && (co[idx] & p1) == 0) co[idx] = (short)(co[idx] >= 0 ? co[idx] + p1 : co[idx] + m1);
                }
                _eobRun--;
            }
        };

        // --- pixels ---

        static readonly float[] CosTable = BuildCos();
        static float[] BuildCos()
        {
            var t = new float[64];
            for (int x = 0; x < 8; x++)
                for (int u = 0; u < 8; u++)
                    t[x * 8 + u] = (float)(Math.Cos((2 * x + 1) * u * Math.PI / 16) * (u == 0 ? Math.Sqrt(0.125) : 0.5));
            return t;
        }

        /// <summary>Dequantises and inverse-transforms every block of a component into a plane of whole 0..255 samples.</summary>
        float[] Plane(Component c)
        {
            var q = _quant[c.Tq] ?? throw new InvalidDataException("JPEG uses a quantisation table that was not defined.");
            int stride = c.BlocksPerLineForMcu * 8;
            var plane = new float[stride * c.BlocksPerColumnForMcu * 8];
            var tmp = new float[64];
            var blk = new float[64];
            for (int by = 0; by < c.BlocksPerColumnForMcu; by++)
                for (int bx = 0; bx < c.BlocksPerLineForMcu; bx++)
                {
                    int off = (by * c.BlocksPerLineForMcu + bx) * 64;
                    for (int i = 0; i < 64; i++) blk[i] = c.Coefficients[off + i] * q[i];
                    // rows: tmp[y][x] = sum_u blk[y][u] * cos[x][u]
                    for (int y = 0; y < 8; y++)
                        for (int x = 0; x < 8; x++)
                        {
                            float s = 0;
                            for (int u = 0; u < 8; u++) s += blk[y * 8 + u] * CosTable[x * 8 + u];
                            tmp[y * 8 + x] = s;
                        }
                    for (int x = 0; x < 8; x++)
                        for (int y = 0; y < 8; y++)
                        {
                            float s = 0;
                            for (int v = 0; v < 8; v++) s += tmp[v * 8 + x] * CosTable[y * 8 + v];
                            plane[(by * 8 + y) * stride + bx * 8 + x] = Math.Clamp(MathF.Round(s + 128f), 0f, 255f); // libjpeg range-limits each component before colour conversion
                        }
                }
            return plane;
        }

        /// <summary>The component's sample for picture pixel (x, y): a triangle filter over the subsampled plane, like libjpeg's fancy upsampling.</summary>
        static float Sample(float[] plane, int stride, int validW, int validH, int x, int y, int hs, int vs, int hMax, int vMax)
        {
            if (hs == hMax && vs == vMax) return plane[y * stride + x];
            float fx = (x + 0.5f) * hs / hMax - 0.5f, fy = (y + 0.5f) * vs / vMax - 0.5f;
            int x0 = (int)MathF.Floor(fx), y0 = (int)MathF.Floor(fy);
            float tx = fx - x0, ty = fy - y0;
            int xa = Math.Clamp(x0, 0, validW - 1), xb = Math.Clamp(x0 + 1, 0, validW - 1), ya = Math.Clamp(y0, 0, validH - 1), yb = Math.Clamp(y0 + 1, 0, validH - 1);
            float top = plane[ya * stride + xa] * (1 - tx) + plane[ya * stride + xb] * tx;
            float bottom = plane[yb * stride + xa] * (1 - tx) + plane[yb * stride + xb] * tx;
            return top * (1 - ty) + bottom * ty;
        }

        ImageBatch Output()
        {
            var planes = _components.Select(Plane).ToArray();
            var img = new ImageBatch(1, _height, _width, 3);
            bool ycc = _components.Length == 3 && _adobeTransform != 0;
            if (_components.Length == 3 && _adobeTransform < 0 && _components[0].Id == 'R' && _components[1].Id == 'G' && _components[2].Id == 'B') ycc = false;
            var vals = new float[3];
            for (int y = 0; y < _height; y++)
                for (int x = 0; x < _width; x++)
                {
                    for (int i = 0; i < _components.Length; i++)
                    {
                        var c = _components[i];
                        int validW = (_width * c.H + _hMax - 1) / _hMax, validH = (_height * c.V + _vMax - 1) / _vMax;
                        vals[i] = Sample(planes[i], c.BlocksPerLineForMcu * 8, validW, validH, x, y, c.H, c.V, _hMax, _vMax);
                    }
                    float r, g, b;
                    if (_components.Length == 1) r = g = b = vals[0];
                    else if (ycc)
                    {
                        float Y = vals[0], cb = vals[1] - 128f, cr = vals[2] - 128f;
                        r = Y + 1.402f * cr; g = Y - 0.344136f * cb - 0.714136f * cr; b = Y + 1.772f * cb;
                    }
                    else { r = vals[0]; g = vals[1]; b = vals[2]; }
                    int o = (y * _width + x) * 3;
                    img.Data[o] = Math.Clamp(MathF.Round(r), 0f, 255f) / 255f;
                    img.Data[o + 1] = Math.Clamp(MathF.Round(g), 0f, 255f) / 255f;
                    img.Data[o + 2] = Math.Clamp(MathF.Round(b), 0f, 255f) / 255f;
                }
            return applyOrientation ? Orient(img, _orientation) : img;
        }

        /// <summary>Applies the EXIF orientation (1 = as stored; 2 mirror; 3 rotate 180; 4 flip; 5 transpose; 6 rotate 90 cw; 7 transverse; 8 rotate 270 cw).</summary>
        static ImageBatch Orient(ImageBatch img, int orientation)
        {
            if (orientation == 1) return img;
            int w = img.Width, h = img.Height;
            bool swap = orientation >= 5;
            int nw = swap ? h : w, nh = swap ? w : h;
            var o = new ImageBatch(1, nh, nw, 3);
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    var (nx, ny) = orientation switch
                    {
                        2 => (w - 1 - x, y),
                        3 => (w - 1 - x, h - 1 - y),
                        4 => (x, h - 1 - y),
                        5 => (y, x),
                        6 => (h - 1 - y, x),
                        7 => (h - 1 - y, w - 1 - x),
                        _ => (y, w - 1 - x),
                    };
                    Array.Copy(img.Data, (y * w + x) * 3, o.Data, (ny * nw + nx) * 3, 3);
                }
            return o;
        }
    }
}
