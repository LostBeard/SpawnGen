namespace SpawnGen.Nodes.ML;

/// <summary>The Beta distribution's quantile function, which the beta scheduler needs (scipy's <c>beta.ppf</c>): the regularized incomplete beta function inverted by bisection.</summary>
public static class BetaMath
{
    static double LogGamma(double x)
    {
        // Lanczos approximation (g = 7, n = 9).
        double[] c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
        if (x < 0.5) return Math.Log(Math.PI / Math.Abs(Math.Sin(Math.PI * x))) - LogGamma(1 - x);
        x -= 1;
        double a = c[0], t = x + 7.5;
        for (int i = 1; i < 9; i++) a += c[i] / (x + i);
        return 0.5 * Math.Log(2 * Math.PI) + (x + 0.5) * Math.Log(t) - t + Math.Log(a);
    }

    /// <summary>The regularized incomplete beta function I_x(a, b): the Beta(a, b) distribution's CDF at x.</summary>
    public static double Cdf(double x, double a, double b)
    {
        if (x <= 0) return 0;
        if (x >= 1) return 1;
        double front = Math.Exp(LogGamma(a + b) - LogGamma(a) - LogGamma(b) + a * Math.Log(x) + b * Math.Log(1 - x));
        // The continued fraction converges fast for x below the mean; use the symmetry I_x(a, b) = 1 - I_(1-x)(b, a) above it.
        return x < (a + 1) / (a + b + 2) ? front * Fraction(x, a, b) / a : 1 - front * Fraction(1 - x, b, a) / b;
    }

    static double Fraction(double x, double a, double b)
    {
        // Lentz's method for the continued fraction of the incomplete beta function.
        const double tiny = 1e-300;
        double qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap;
        if (Math.Abs(d) < tiny) d = tiny;
        d = 1 / d;
        double h = d;
        for (int m = 1; m <= 300; m++)
        {
            int m2 = 2 * m;
            double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1 + aa * d; if (Math.Abs(d) < tiny) d = tiny;
            c = 1 + aa / c; if (Math.Abs(c) < tiny) c = tiny;
            d = 1 / d; h *= d * c;
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1 + aa * d; if (Math.Abs(d) < tiny) d = tiny;
            c = 1 + aa / c; if (Math.Abs(c) < tiny) c = tiny;
            d = 1 / d;
            double delta = d * c;
            h *= delta;
            if (Math.Abs(delta - 1) < 1e-14) break;
        }
        return h;
    }

    /// <summary>The x with Beta(a, b) CDF equal to <paramref name="q"/> (0 for q ≤ 0, 1 for q ≥ 1).</summary>
    public static double Ppf(double q, double a, double b)
    {
        if (q <= 0) return 0;
        if (q >= 1) return 1;
        double lo = 0, hi = 1;
        for (int i = 0; i < 100; i++)
        {
            double mid = (lo + hi) / 2;
            if (Cdf(mid, a, b) < q) lo = mid; else hi = mid;
        }
        return (lo + hi) / 2;
    }
}
