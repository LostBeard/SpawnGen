using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// ComfyUI's <c>comfy/audio.py</c> (adapted from TorchAudio, BSD-2-Clause): band-limited sinc resampling and the bass, peaking and treble biquads.
/// The arithmetic is float32 in the same order as torch and scipy's <c>lfilter</c>, so results match to rounding.
/// </summary>
public static class AudioDsp
{
    /// <summary>
    /// <c>comfy.audio.resample</c> with its defaults (Hann-windowed sinc, 6 zero crossings, rolloff 0.99): the rates are reduced by their gcd, each output
    /// phase gets its own kernel, and the result has <c>ceil(float32(new * length / orig))</c> samples.
    /// </summary>
    public static float[] Resample(float[] waveform, int origFreq, int newFreq, int lowpassFilterWidth = 6, double rolloff = 0.99)
    {
        if (origFreq <= 0 || newFreq <= 0) throw new ArgumentException("Sample rates must be positive");
        if (origFreq == newFreq) return waveform;
        int divisor = Gcd(origFreq, newFreq), orig = origFreq / divisor, next = newFreq / divisor;
        double baseFreq = Math.Min(orig, next) * rolloff;
        int width = (int)Math.Ceiling(lowpassFilterWidth * orig / baseFreq);
        int taps = 2 * width + orig;

        // kernel[phase][tap]: torch builds it in the waveform's dtype (float32), every Python float cast to float32 first.
        float baseF = (float)baseFreq, scale = (float)(baseFreq / orig), pi = (float)Math.PI, lw = lowpassFilterWidth;
        var kernel = new float[next][];
        for (int j = 0; j < next; j++)
        {
            var row = kernel[j] = new float[taps];
            for (int i = 0; i < taps; i++)
            {
                float t = (float)-j / next + (float)(i - width) / orig;
                t = Math.Clamp(t * baseF, -lw, lw);
                float window = MathF.Cos(t * pi / lw / 2);
                window *= window;
                t *= pi;
                row[i] = (t == 0 ? 1f : MathF.Sin(t) / t) * (window * scale);
            }
        }

        // conv1d with stride orig over the signal padded by width zeros in front and width + orig behind; phases interleave.
        int length = waveform.Length, frames = length / orig + 1;
        int target = (int)Math.Ceiling((float)((double)next * length / orig));
        var output = new float[target];
        for (int l = 0; l < frames; l++)
            for (int j = 0; j < next; j++)
            {
                int o = l * next + j;
                if (o >= target) return output;
                var row = kernel[j];
                double sum = 0;
                for (int i = 0; i < taps; i++)
                {
                    int s = l * orig + i - width;
                    if ((uint)s < (uint)length) sum += waveform[s] * row[i];
                }
                output[o] = (float)sum;
            }
        return output;
    }

    /// <summary>Every channel of <paramref name="clip"/> resampled to <paramref name="newRate"/> (the clip itself when the rate is already that).</summary>
    public static AudioClip Resample(AudioClip clip, int newRate) =>
        clip.SampleRate == newRate ? clip : new AudioClip([.. clip.Channels.Select(c => Resample(c, clip.SampleRate, newRate))], newRate);

    /// <summary>
    /// ComfyUI's <c>match_audio_sample_rates</c>: the clip with the lower rate is resampled up to the higher one.
    /// </summary>
    public static (AudioClip First, AudioClip Second, int SampleRate) MatchSampleRates(AudioClip first, AudioClip second)
    {
        int rate = Math.Max(first.SampleRate, second.SampleRate);
        return (Resample(first, rate), Resample(second, rate), rate);
    }

    /// <summary><c>bass_biquad</c>: a low shelf at <paramref name="centralFreq"/> boosting or cutting by <paramref name="gain"/> dB.</summary>
    public static float[] Bass(float[] waveform, int sampleRate, double gain, double centralFreq = 100, double q = 0.707)
    {
        float cf = (float)centralFreq, qf = (float)q, g = (float)gain;
        float w0 = 2f * MathF.PI * cf / sampleRate;
        float alpha = MathF.Sin(w0) / 2f / qf;
        float a = MathF.Exp(g / 40f * (float)Math.Log(10));
        float temp1 = 2f * MathF.Sqrt(a) * alpha, temp2 = (a - 1f) * MathF.Cos(w0), temp3 = (a + 1f) * MathF.Cos(w0);
        float b0 = a * (a + 1f - temp2 + temp1), b1 = 2f * a * (a - 1f - temp3), b2 = a * (a + 1f - temp2 - temp1);
        float a0 = a + 1f + temp2 + temp1, a1 = -2f * (a - 1f + temp3), a2 = a + 1f + temp2 - temp1;
        // ComfyUI normalises these by a0 itself (lfilter then divides by a0 / a0 = 1).
        return Biquad(waveform, b0 / a0, b1 / a0, b2 / a0, a0 / a0, a1 / a0, a2 / a0);
    }

    /// <summary><c>equalizer_biquad</c>: a peaking filter at <paramref name="centerFreq"/> with bandwidth <paramref name="q"/>.</summary>
    public static float[] Equalizer(float[] waveform, int sampleRate, double centerFreq, double gain, double q = 0.707)
    {
        float cf = (float)centerFreq, qf = (float)q, g = (float)gain;
        float w0 = 2f * MathF.PI * cf / sampleRate;
        float a = MathF.Exp(g / 40f * (float)Math.Log(10));
        float alpha = MathF.Sin(w0) / 2f / qf;
        return Biquad(waveform, 1f + alpha * a, -2f * MathF.Cos(w0), 1f - alpha * a, 1f + alpha / a, -2f * MathF.Cos(w0), 1f - alpha / a);
    }

    /// <summary><c>treble_biquad</c>: a high shelf at <paramref name="centralFreq"/>.</summary>
    public static float[] Treble(float[] waveform, int sampleRate, double gain, double centralFreq = 3000, double q = 0.707)
    {
        float cf = (float)centralFreq, qf = (float)q, g = (float)gain;
        float w0 = 2f * MathF.PI * cf / sampleRate;
        float alpha = MathF.Sin(w0) / 2f / qf;
        float a = MathF.Exp(g / 40f * (float)Math.Log(10));
        float temp1 = 2f * MathF.Sqrt(a) * alpha, temp2 = (a - 1f) * MathF.Cos(w0), temp3 = (a + 1f) * MathF.Cos(w0);
        float b0 = a * (a + 1f + temp2 + temp1), b1 = -2f * a * (a - 1f + temp3), b2 = a * (a + 1f + temp2 - temp1);
        float a0 = a + 1f - temp2 + temp1, a1 = 2f * (a - 1f - temp3), a2 = a + 1f - temp2 - temp1;
        return Biquad(waveform, b0, b1, b2, a0, a1, a2);
    }

    /// <summary>
    /// <c>_biquad</c>: scipy's float32 <c>lfilter</c> (coefficients divided by a0 once, direct form II transposed, zero initial state), then clamped to -1..1.
    /// </summary>
    public static float[] Biquad(float[] x, float b0, float b1, float b2, float a0, float a1, float a2)
    {
        b0 /= a0; b1 /= a0; b2 /= a0; a1 /= a0; a2 /= a0;
        var y = new float[x.Length];
        float z0 = 0, z1 = 0;
        for (int n = 0; n < x.Length; n++)
        {
            float xn = x[n], yn = z0 + b0 * xn;
            z0 = z1 + xn * b1 - yn * a1;
            z1 = xn * b2 - yn * a2;
            y[n] = Math.Clamp(yn, -1f, 1f);
        }
        return y;
    }

    static int Gcd(int a, int b)
    {
        while (b != 0) (a, b) = (b, a % b);
        return a;
    }
}
