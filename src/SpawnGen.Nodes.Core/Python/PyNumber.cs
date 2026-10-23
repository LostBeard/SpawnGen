using System.Globalization;
using System.Numerics;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// Python's number rules for the text and math nodes: <c>repr(float)</c>, exactly rounded decimal digits (what <c>format</c> and <c>round</c> use),
/// <c>int(str)</c> / <c>float(str)</c> parsing, and int / float conversions and comparisons without rounding surprises.
/// </summary>
public static class PyNumber
{
    static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

    /// <summary>Python's <c>repr(x)</c> / <c>str(x)</c> of a float: the shortest digits that read back the same, fixed notation for 1e-4 &lt;= |x| &lt; 1e16.</summary>
    public static string FloatRepr(double x)
    {
        if (double.IsNaN(x)) return "nan";
        if (double.IsInfinity(x)) return x > 0 ? "inf" : "-inf";
        var (digits, decpt) = ShortestDigits(x);
        var sign = double.IsNegative(x) ? "-" : "";
        if (decpt <= -4 || decpt > 16) return sign + Scientific(digits, decpt - 1, keepPoint: false);
        return sign + FixedFromDigits(digits, decpt, minFraction: 1);
    }

    /// <summary>The shortest round-trip digits of |x| (no leading or trailing zeros; "0" for zero) and the decimal point position: |x| = 0.DIGITS * 10^decpt.</summary>
    public static (string Digits, int DecPt) ShortestDigits(double x)
    {
        if (x == 0) return ("0", 1);
        // .NET Core 3.0+ formats "R" as the shortest string that parses back to the same double, which is also what Python's repr picks.
        var s = Math.Abs(x).ToString("R", Inv);
        int exp = 0;
        int e = s.IndexOfAny(['E', 'e']);
        if (e >= 0) { exp = int.Parse(s[(e + 1)..], Inv); s = s[..e]; }
        int dot = s.IndexOf('.');
        string intPart = dot < 0 ? s : s[..dot], frac = dot < 0 ? "" : s[(dot + 1)..];
        var all = intPart + frac;
        int decpt = intPart.Length + exp;
        int lead = 0;
        while (lead < all.Length - 1 && all[lead] == '0') lead++;
        all = all[lead..];
        decpt -= lead;
        all = all.TrimEnd('0');
        if (all.Length == 0) return ("0", 1);
        return (all, decpt);
    }

    /// <summary><c>d.ddde+XX</c> from a digit string whose first digit has the decimal exponent <paramref name="exp"/>.</summary>
    internal static string Scientific(string digits, int exp, bool keepPoint, char e = 'e')
    {
        var mant = digits.Length > 1 ? digits[0] + "." + digits[1..] : keepPoint ? digits + "." : digits;
        return mant + e + (exp < 0 ? "-" : "+") + Math.Abs(exp).ToString("00", Inv);
    }

    /// <summary>Fixed notation of 0.DIGITS * 10^decpt with at least <paramref name="minFraction"/> fraction digits (zeros added).</summary>
    internal static string FixedFromDigits(string digits, int decpt, int minFraction)
    {
        string intPart, frac;
        if (decpt <= 0) { intPart = "0"; frac = new string('0', -decpt) + digits; }
        else if (decpt >= digits.Length) { intPart = digits + new string('0', decpt - digits.Length); frac = ""; }
        else { intPart = digits[..decpt]; frac = digits[decpt..]; }
        if (frac.Length < minFraction) frac += new string('0', minFraction - frac.Length);
        return frac.Length == 0 ? intPart : intPart + "." + frac;
    }

    // ---- exact decimal digits ------------------------------------------------------------------------------------

    /// <summary>|x| as m * 2^e with an integer m (exact).</summary>
    static (BigInteger M, int E) Decompose(double x)
    {
        long bits = BitConverter.DoubleToInt64Bits(Math.Abs(x));
        int rawExp = (int)((bits >> 52) & 0x7FF);
        long frac = bits & 0xFFFFFFFFFFFFFL;
        return rawExp == 0 ? (frac, -1074) : (frac | (1L << 52), rawExp - 1075);
    }

    /// <summary>|x| as an exact fraction num / den.</summary>
    static (BigInteger Num, BigInteger Den) Exact(double x)
    {
        var (m, e) = Decompose(x);
        return e >= 0 ? (m << e, BigInteger.One) : (m, BigInteger.One << -e);
    }

    /// <summary>num / den rounded to an integer, ties to even (non-negative inputs).</summary>
    static BigInteger RoundHalfEven(BigInteger num, BigInteger den)
    {
        var q = BigInteger.DivRem(num, den, out var r);
        var twice = r * 2;
        if (twice > den || (twice == den && !q.IsEven)) q += 1;
        return q;
    }

    /// <summary>|x| * 10^places rounded half to even (places may be negative): the digits of <c>'%.{places}f'</c>.</summary>
    public static BigInteger FixedDigits(double x, int places)
    {
        var (num, den) = Exact(x);
        if (places >= 0) num *= BigInteger.Pow(10, places);
        else den *= BigInteger.Pow(10, -places);
        return RoundHalfEven(num, den);
    }

    /// <summary>
    /// |x| rounded to <paramref name="count"/> significant digits (half to even): the digit string (exactly <paramref name="count"/> long, trailing zeros kept)
    /// and the decimal exponent of its first digit. Zero gives zeros and exponent 0.
    /// </summary>
    public static (string Digits, int Exp) SignificantDigits(double x, int count)
    {
        if (x == 0) return (new string('0', count), 0);
        var (num, den) = Exact(x);
        int k = (int)Math.Floor(Math.Log10(Math.Abs(x)));
        // Make sure 10^k <= |x| < 10^(k+1) exactly (the logarithm can be off by one near powers of ten).
        while (Compare(num, den, k) < 0) k--;
        while (Compare(num, den, k + 1) >= 0) k++;
        int shift = count - 1 - k;
        BigInteger n = shift >= 0 ? RoundHalfEven(num * BigInteger.Pow(10, shift), den) : RoundHalfEven(num, den * BigInteger.Pow(10, -shift));
        var s = n.ToString(Inv);
        if (s.Length > count) { s = s[..count]; k++; } // rounded up to the next power of ten
        return (s, k);
    }

    /// <summary>Sign of num/den - 10^k.</summary>
    static int Compare(BigInteger num, BigInteger den, int k) =>
        k >= 0 ? num.CompareTo(den * BigInteger.Pow(10, k)) : (num * BigInteger.Pow(10, -k)).CompareTo(den);

    /// <summary>Python's <c>round(x, ndigits)</c> for a float: exact decimal rounding (half to even), the result read back as the nearest float.</summary>
    public static double Round(double x, long ndigits)
    {
        if (double.IsNaN(x) || double.IsInfinity(x) || x == 0) return x;
        if (ndigits > 323) return x;
        if (ndigits < -308) return 0.0 * x;
        var n = FixedDigits(x, (int)ndigits);
        var text = (double.IsNegative(x) ? "-" : "") + n.ToString(Inv) + "e" + (-ndigits).ToString(Inv);
        var r = double.Parse(text, NumberStyles.Float, Inv);
        if (double.IsInfinity(r)) throw new OverflowException("rounded value too large to represent");
        return r;
    }

    /// <summary>Python's <c>round(x)</c> of a float: the nearest integer, ties to even.</summary>
    public static BigInteger RoundToInt(double x)
    {
        if (double.IsNaN(x)) throw new ArgumentException("cannot convert float NaN to integer");
        if (double.IsInfinity(x)) throw new OverflowException("cannot convert float infinity to integer");
        return ToBigInteger(Math.Round(x, MidpointRounding.ToEven));
    }

    // ---- int <-> float ---------------------------------------------------------------------------------------------

    /// <summary>An integral double as an exact integer (<c>int(x)</c> after truncation).</summary>
    public static BigInteger ToBigInteger(double integral)
    {
        if (double.IsNaN(integral)) throw new ArgumentException("cannot convert float NaN to integer");
        if (double.IsInfinity(integral)) throw new OverflowException("cannot convert float infinity to integer");
        integral = Math.Truncate(integral);
        if (Math.Abs(integral) < 9e15) return new BigInteger((long)integral);
        var (num, den) = Exact(integral);
        var v = num / den;
        return integral < 0 ? -v : v;
    }

    /// <summary>Python's <c>float(n)</c>: the nearest double (ties to even), OverflowError when it is too large.</summary>
    public static double ToDouble(BigInteger n)
    {
        if (BigInteger.Abs(n) < (BigInteger.One << 53)) return (double)(long)n;
        bool neg = n.Sign < 0;
        var a = BigInteger.Abs(n);
        long bits = (long)a.GetBitLength();
        int shift = (int)(bits - 54); // keep 54 bits: 53 + one rounding bit, the rest is sticky
        var top = a >> shift;
        bool sticky = !(a & ((BigInteger.One << shift) - 1)).IsZero;
        long t = (long)top;
        long mant = t >> 1;
        if ((t & 1) == 1 && (sticky || (mant & 1) == 1)) mant++;
        int exp = shift + 1;
        if (mant == 1L << 53) { mant >>= 1; exp++; }
        if (exp + 52 > 1023) throw new OverflowException("int too large to convert to float");
        double r = Math.ScaleB(mant, exp);
        if (double.IsInfinity(r)) throw new OverflowException("int too large to convert to float");
        return neg ? -r : r;
    }

    /// <summary>Python's exact int / float comparison (no rounding of the int). NaN compares as unordered (<c>null</c>).</summary>
    public static int? CompareIntFloat(BigInteger a, double b)
    {
        if (double.IsNaN(b)) return null;
        if (double.IsPositiveInfinity(b)) return -1;
        if (double.IsNegativeInfinity(b)) return 1;
        var fl = Math.Floor(b);
        int c = a.CompareTo(ToBigInteger(fl));
        if (c != 0) return c;
        return fl == b ? 0 : -1; // a == floor(b) < b
    }

    /// <summary>Python's int / int true division: the correctly rounded quotient.</summary>
    public static double TrueDivide(BigInteger a, BigInteger b)
    {
        if (b.IsZero) throw new DivideByZeroException("division by zero");
        var lim = BigInteger.One << 53;
        if (BigInteger.Abs(a) <= lim && BigInteger.Abs(b) <= lim) return (double)(long)a / (double)(long)b;
        bool neg = (a.Sign < 0) ^ (b.Sign < 0);
        a = BigInteger.Abs(a); b = BigInteger.Abs(b);
        // Scale so the quotient has 55+ bits, then round the integer quotient (sticky bit from the remainder).
        long shift = (long)b.GetBitLength() - (long)a.GetBitLength() + 55;
        var num = shift > 0 ? a << (int)shift : a;
        var den = shift < 0 ? b << (int)-shift : b;
        var q = BigInteger.DivRem(num, den, out var r);
        if (!r.IsZero) q |= 1; // sticky
        double result;
        try { result = ScaleExact(q, -shift); }
        catch (OverflowException) { throw new OverflowException("integer division result too large for a float"); }
        return neg ? -result : result;
    }

    /// <summary>q * 2^e rounded once to a double (q has a sticky low bit).</summary>
    static double ScaleExact(BigInteger q, long e)
    {
        long bits = (long)q.GetBitLength();
        // Target exponent of the lowest kept bit, allowing for subnormals.
        long lowest = Math.Max(bits + e - 53, -1074);
        int drop = (int)(lowest - e);
        BigInteger mant;
        if (drop > 0)
        {
            var kept = q >> drop;
            var rem = q - (kept << drop);
            var half = BigInteger.One << (drop - 1);
            if (rem > half || (rem == half && !kept.IsEven)) kept += 1;
            mant = kept;
        }
        else { mant = q << -drop; }
        double r = Math.ScaleB((double)(long)mant, (int)lowest);
        if (double.IsInfinity(r)) throw new OverflowException("result too large");
        return r;
    }

    // ---- parsing ---------------------------------------------------------------------------------------------------

    /// <summary>Python's whitespace (<c>str.isspace</c>): the ASCII controls 9-13 and 28-31, space, and the Unicode separators.</summary>
    public static bool IsSpace(char c) => c switch
    {
        >= '\t' and <= '\r' => true,
        >= '\x1c' and <= '\x20' => true,
        '\x85' or '\xa0' or '\u1680' or '\u2028' or '\u2029' or '\u202F' or '\u205F' or '\u3000' => true,
        >= '\u2000' and <= '\u200A' => true,
        _ => false,
    };

    /// <summary>Python converts every Unicode decimal digit to its ASCII digit before parsing a number (and Unicode spaces to a space).</summary>
    static string AsciiDigits(string s)
    {
        var sb = new System.Text.StringBuilder(s.Length);
        foreach (var c in s)
        {
            if (c > 127 && char.IsDigit(c)) sb.Append((char)('0' + (int)char.GetNumericValue(c)));
            else if (c > 127 && IsSpace(c)) sb.Append(' ');
            else sb.Append(c);
        }
        return sb.ToString().Trim(' ', '\t', '\n', '\r', '\f', '\v', '\x1c', '\x1d', '\x1e', '\x1f');
    }

    /// <summary>Digits with single underscores between them (Python's <c>digitpart</c>), in the given base.</summary>
    static bool IsDigitPart(string s, int start, int end, int @base)
    {
        if (start >= end) return false;
        bool prevDigit = false;
        for (int i = start; i < end; i++)
        {
            if (s[i] == '_') { if (!prevDigit || i == end - 1) return false; prevDigit = false; continue; }
            if (DigitValue(s[i]) is not { } d || d >= @base) return false;
            prevDigit = true;
        }
        return true;
    }

    static int? DigitValue(char c) => c switch
    {
        >= '0' and <= '9' => c - '0',
        >= 'a' and <= 'z' => c - 'a' + 10,
        >= 'A' and <= 'Z' => c - 'A' + 10,
        _ => null,
    };

    /// <summary>Python's <c>int(text, base)</c> (base 0 reads a 0x / 0o / 0b prefix). Null when the text is not an integer.</summary>
    public static BigInteger? TryParseInt(string text, int @base = 10)
    {
        var s = AsciiDigits(text);
        int i = 0;
        bool neg = false;
        if (i < s.Length && s[i] is '+' or '-') { neg = s[i] == '-'; i++; }
        if (@base == 0 || @base == 16 || @base == 8 || @base == 2)
        {
            if (i + 1 < s.Length && s[i] == '0' && char.ToLowerInvariant(s[i + 1]) is 'x' or 'o' or 'b')
            {
                int pb = char.ToLowerInvariant(s[i + 1]) switch { 'x' => 16, 'o' => 8, _ => 2 };
                if (@base == 0 || @base == pb)
                {
                    @base = pb;
                    i += 2;
                    if (i < s.Length && s[i] == '_') i++; // 0x_ff is allowed
                }
            }
            if (@base == 0)
            {
                @base = 10;
                // base 0 refuses leading zeros on a non-zero decimal ("010")
                var body = s[i..].Replace("_", "");
                if (body.Length > 1 && body[0] == '0' && body.Any(c => c != '0')) return null;
            }
        }
        if (!IsDigitPart(s, i, s.Length, @base)) return null;
        BigInteger v = BigInteger.Zero;
        for (; i < s.Length; i++)
            if (s[i] != '_') v = v * @base + DigitValue(s[i])!.Value;
        return neg ? -v : v;
    }

    /// <summary>Python's <c>float(text)</c>: decimals with optional exponent and underscores between digits, "inf", "infinity", "nan". Null when invalid.</summary>
    public static double? TryParseFloat(string text)
    {
        var s = AsciiDigits(text);
        int i = 0;
        bool neg = false;
        if (i < s.Length && s[i] is '+' or '-') { neg = s[i] == '-'; i++; }
        var rest = s[i..].ToLowerInvariant();
        if (rest is "inf" or "infinity") return neg ? double.NegativeInfinity : double.PositiveInfinity;
        if (rest == "nan") return double.NaN;
        // numeric: digitpart [. [digitpart]] [exp] | . digitpart [exp]
        int p = i, n = s.Length;
        int intStart = p;
        while (p < n && (char.IsAsciiDigit(s[p]) || s[p] == '_')) p++;
        int intEnd = p;
        bool hasInt = intEnd > intStart;
        if (hasInt && !IsDigitPart(s, intStart, intEnd, 10)) return null;
        bool hasFrac = false;
        if (p < n && s[p] == '.')
        {
            p++;
            int fs = p;
            while (p < n && (char.IsAsciiDigit(s[p]) || s[p] == '_')) p++;
            if (p > fs) { if (!IsDigitPart(s, fs, p, 10)) return null; hasFrac = true; }
        }
        if (!hasInt && !hasFrac) return null;
        if (p < n && s[p] is 'e' or 'E')
        {
            p++;
            if (p < n && s[p] is '+' or '-') p++;
            int es = p;
            while (p < n && (char.IsAsciiDigit(s[p]) || s[p] == '_')) p++;
            if (!IsDigitPart(s, es, p, 10)) return null;
        }
        if (p != n) return null;
        var clean = s[i..].Replace("_", "");
        var v = double.Parse(clean, NumberStyles.Float, Inv);
        return neg ? -v : v;
    }
}
