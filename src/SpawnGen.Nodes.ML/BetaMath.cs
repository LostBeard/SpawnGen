namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md) from the mathematics: NIST DLMF 8.17 (the incomplete beta function, continued fraction 8.17.22 and the
// symmetry 8.17.4), the modified Lentz algorithm for continued fractions, and Lanczos' approximation of the gamma function (g = 7, n = 9 coefficients,
// as published by P. Godfrey). Checked against scipy.stats.beta (BSD-3) by running it.

/// <summary>The beta distribution's CDF (the regularized incomplete beta function I_x(a, b)) and its inverse, in double precision.</summary>
public static class BetaMath
{
    /// <summary>I_x(a, b): the probability that a Beta(a, b) variable is at most <paramref name="x"/> (0 below 0, 1 above 1).</summary>
    public static double Cdf(double x, double a, double b)
    {
        if (!(a > 0) || !(b > 0)) throw new ArgumentOutOfRangeException(a > 0 ? nameof(b) : nameof(a), "Beta parameters must be positive.");
        if (double.IsNaN(x)) return double.NaN;
        if (x <= 0) return 0;
        if (x >= 1) return 1;
        // The continued fraction converges fast below the mean-like point (a + 1) / (a + b + 2); above it, DLMF 8.17.4: I_x(a, b) = 1 - I_{1-x}(b, a).
        return x < (a + 1) / (a + b + 2) ? Front(x, a, b) : 1 - Front(1 - x, b, a);
    }

    /// <summary>
    /// The quantile: the x in [0, 1] with I_x(a, b) = <paramref name="q"/>. The CDF rises monotonically, so x is found by bisection down to adjacent
    /// doubles (the last bit the CDF can resolve).
    /// </summary>
    public static double Ppf(double q, double a, double b)
    {
        if (!(a > 0) || !(b > 0)) throw new ArgumentOutOfRangeException(a > 0 ? nameof(b) : nameof(a), "Beta parameters must be positive.");
        if (double.IsNaN(q) || q < 0 || q > 1) return double.NaN;
        if (q == 0) return 0;
        if (q == 1) return 1;
        if (a == b && q == 0.5) return 0.5; // a symmetric distribution's median, exactly (scipy rounds it up by an ulp; the bisection lands an ulp below)
        double lo = 0, hi = 1;
        while (true)
        {
            double mid = lo + (hi - lo) / 2;
            if (mid <= lo || mid >= hi) break;
            if (Cdf(mid, a, b) < q) lo = mid; else hi = mid;
        }
        // Of the two neighbours, the one whose CDF is closer to q.
        return Math.Abs(Cdf(lo, a, b) - q) < Math.Abs(Cdf(hi, a, b) - q) ? lo : hi;
    }

    /// <summary>I_x(a, b) by DLMF 8.17.22: x^a (1 - x)^b / (a B(a, b)) times the continued fraction 1 / (1 + d1 / (1 + d2 / (1 + ...))).</summary>
    static double Front(double x, double a, double b)
    {
        double logPrefix = a * Math.Log(x) + b * Log1P(-x) - LogBeta(a, b) - Math.Log(a);
        return Math.Exp(logPrefix) * ContinuedFraction(x, a, b);
    }

    /// <summary>
    /// 1 / (1 + d1 / (1 + d2 / ...)) with d(2m+1) = -(a + m)(a + b + m) x / ((a + 2m)(a + 2m + 1)) and d(2m) = m (b - m) x / ((a + 2m - 1)(a + 2m)),
    /// by the modified Lentz algorithm.
    /// </summary>
    static double ContinuedFraction(double x, double a, double b)
    {
        const double tiny = 1e-300, eps = 1e-16;
        // The fraction is b0 + a1 / (b1 + a2 / (b2 + ...)) with b0 = 0, a1 = 1 and every b = 1, a(k+1) = d(k).
        double f = tiny, c = f, d = 0;
        for (int k = 1; k <= 10000; k++)
        {
            double num;
            if (k == 1) num = 1;
            else
            {
                int j = k - 1, m = j / 2;
                num = j % 2 == 1
                    ? -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
                    : m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m));
            }
            d = 1 + num * d;
            if (Math.Abs(d) < tiny) d = tiny;
            c = 1 + num / c;
            if (Math.Abs(c) < tiny) c = tiny;
            d = 1 / d;
            double delta = c * d;
            f *= delta;
            if (Math.Abs(delta - 1) < eps) break;
        }
        return f;
    }

    /// <summary>ln(1 + x) without losing x's digits when x is small: ln(u) x / (u - 1) with u = 1 + x rounded (Goldberg 1991, theorem 4).</summary>
    static double Log1P(double x)
    {
        double u = 1 + x;
        return u == 1 ? x : Math.Log(u) * x / (u - 1);
    }

    static double LogBeta(double a, double b) => LogGamma(a) + LogGamma(b) - LogGamma(a + b);

    static readonly double[] Lanczos =
    [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
        12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
    ];

    /// <summary>ln Gamma(z) for z > 0: Lanczos' approximation (g = 7), with the reflection formula below 0.5.</summary>
    internal static double LogGamma(double z)
    {
        if (z < 0.5) return Math.Log(Math.PI / Math.Abs(Math.Sin(Math.PI * z))) - LogGamma(1 - z);
        z -= 1;
        double sum = Lanczos[0];
        for (int i = 1; i < Lanczos.Length; i++) sum += Lanczos[i] / (z + i);
        double t = z + 7.5;
        return 0.5 * Math.Log(2 * Math.PI) + (z + 0.5) * Math.Log(t) - t + Math.Log(sum);
    }
}
