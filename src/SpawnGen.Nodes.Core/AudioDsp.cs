using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// Written clean-room (docs/CLEANROOM.md). The resampler and the biquads are ported from torchaudio 2.11.1 (BSD-2-Clause, see THIRD-PARTY-NOTICES.md):
// torchaudio.functional.resample (_get_sinc_resample_kernel, _apply_sinc_resample_kernel) and bass_biquad / equalizer_biquad / treble_biquad with
// lfilter, in float32 as torchaudio computes them for a float32 waveform. ComfyUI's audio nodes were measured against them black-box
// (tools/parity/make_audio_fixtures.py).

/// <summary>Sample-rate conversion and tone filters for <see cref="AudioClip"/> channels (float32 samples).</summary>
public static class AudioDsp
{
    /// <summary>
    /// torchaudio's band-limited sinc resampling with a Hann window (lowpass_filter_width zero crossings, rolloff of the lower Nyquist): the rates are
    /// reduced by their gcd, the kernel is computed in float32, and the result has ceil(float32(new * length / orig)) samples. The same rate gives the
    /// same array.
    /// </summary>
    public static float[] Resample(float[] waveform, int origFreq, int newFreq, int lowpassFilterWidth = 6, double rolloff = 0.99)
    {
        if (origFreq <= 0 || newFreq <= 0) throw new ArgumentException("Original frequency and desired frequency should be positive.");
        if (origFreq == newFreq) return waveform;
        int gcd = Gcd(origFreq, newFreq), orig = origFreq / gcd, next = newFreq / gcd;
        double baseFreq = Math.Min(orig, next) * rolloff;
        int width = (int)Math.Ceiling(lowpassFilterWidth * orig / baseFreq);
        int taps = 2 * width + orig;
        float fBase = (float)baseFreq, fPi = MathF.PI, fWidth = lowpassFilterWidth, scale = (float)(baseFreq / orig);
        var kernel = new float[next * taps];
        for (int j = 0; j < next; j++)
            for (int k = 0; k < taps; k++)
            {
                float t = (float)-j / next + (float)(k - width) / orig;
                t *= fBase;
                t = Math.Clamp(t, -fWidth, fWidth);
                float c = MathF.Cos(t * fPi / fWidth / 2f);
                float window = c * c;
                t *= fPi;
                float sinc = t == 0f ? 1f : MathF.Sin(t) / t;
                kernel[j * taps + k] = sinc * (window * scale);
            }

        int length = waveform.Length;
        int target = (int)MathF.Ceiling((float)((double)next * length / orig));
        var output = new float[target];
        for (int i = 0; i < target; i++)
        {
            int frame = i / next, phase = i % next, start = frame * orig - width; // padded index frame * orig is sample frame * orig - width
            double sum = 0;
            for (int k = 0; k < taps; k++)
            {
                int s = start + k;
                if ((uint)s < (uint)length) sum += waveform[s] * kernel[phase * taps + k];
            }
            output[i] = (float)sum;
        }
        return output;
    }

    /// <summary>Every channel resampled to <paramref name="newRate"/>.</summary>
    public static AudioClip Resample(AudioClip clip, int newRate) =>
        clip.SampleRate == newRate ? clip : new AudioClip([.. clip.Channels.Select(c => Resample(c, clip.SampleRate, newRate))], newRate);

    /// <summary>The clip at the lower rate resampled up to the higher one.</summary>
    public static (AudioClip First, AudioClip Second, int SampleRate) MatchSampleRates(AudioClip first, AudioClip second)
    {
        int rate = Math.Max(first.SampleRate, second.SampleRate);
        return (Resample(first, rate), Resample(second, rate), rate);
    }

    static int Gcd(int a, int b)
    {
        while (b != 0) (a, b) = (b, a % b);
        return a;
    }

    /// <summary>The angular frequency, sin(w0) and cos(w0) torchaudio's biquads use, in float32.</summary>
    static (float Alpha, float Cos) Angle(double freq, int sampleRate, double q)
    {
        float w0 = (float)(2 * Math.PI) * (float)freq / sampleRate;
        return (MathF.Sin(w0) / 2f / (float)q, MathF.Cos(w0));
    }

    static float Amplitude(double gain) => MathF.Exp((float)gain / 40f * (float)Math.Log(10));

    /// <summary>torchaudio's bass_biquad: a low shelf of <paramref name="gain"/> dB below <paramref name="centralFreq"/> (Audio EQ Cookbook / SoX).</summary>
    public static float[] Bass(float[] waveform, int sampleRate, double gain, double centralFreq = 100, double q = 0.707)
    {
        var (alpha, cos) = Angle(centralFreq, sampleRate, q);
        float a = Amplitude(gain);
        float temp1 = 2f * MathF.Sqrt(a) * alpha, temp2 = (a - 1f) * cos, temp3 = (a + 1f) * cos;
        float b0 = a * ((a + 1f) - temp2 + temp1), b1 = 2f * a * ((a - 1f) - temp3), b2 = a * ((a + 1f) - temp2 - temp1);
        float a0 = (a + 1f) + temp2 + temp1, a1 = -2f * ((a - 1f) + temp3), a2 = (a + 1f) + temp2 - temp1;
        return Biquad(waveform, b0 / a0, b1 / a0, b2 / a0, a0 / a0, a1 / a0, a2 / a0);
    }

    /// <summary>torchaudio's equalizer_biquad: a peaking filter of <paramref name="gain"/> dB at <paramref name="centerFreq"/>.</summary>
    public static float[] Equalizer(float[] waveform, int sampleRate, double centerFreq, double gain, double q = 0.707)
    {
        var (alpha, cos) = Angle(centerFreq, sampleRate, q);
        float a = MathF.Exp((float)gain / 40f * (float)Math.Log(10));
        return Biquad(waveform, 1f + alpha * a, -2f * cos, 1f - alpha * a, 1f + alpha / a, -2f * cos, 1f - alpha / a);
    }

    /// <summary>torchaudio's treble_biquad: a high shelf of <paramref name="gain"/> dB above <paramref name="centralFreq"/>.</summary>
    public static float[] Treble(float[] waveform, int sampleRate, double gain, double centralFreq = 3000, double q = 0.707)
    {
        var (alpha, cos) = Angle(centralFreq, sampleRate, q);
        float a = Amplitude(gain);
        float temp1 = 2f * MathF.Sqrt(a) * alpha, temp2 = (a - 1f) * cos, temp3 = (a + 1f) * cos;
        float b0 = a * ((a + 1f) + temp2 + temp1), b1 = -2f * a * ((a - 1f) + temp3), b2 = a * ((a + 1f) + temp2 - temp1);
        float a0 = (a + 1f) - temp2 + temp1, a1 = 2f * ((a - 1f) - temp3), a2 = (a + 1f) - temp2 - temp1;
        return Biquad(waveform, b0 / a0, b1 / a0, b2 / a0, a0 / a0, a1 / a0, a2 / a0);
    }

    /// <summary>
    /// A biquad with zero initial state, as measured from ComfyUI: the coefficients divided by a0, then scipy.signal.lfilter's float32 direct form II
    /// transposed recursion (y = z0 + b0 x; z0 = (z1 + b1 x) - a1 y; z1 = b2 x - a2 y; scipy is BSD-3-Clause), the result clamped to -1..1.
    /// </summary>
    public static float[] Biquad(float[] x, float b0, float b1, float b2, float a0, float a1, float a2)
    {
        (b0, b1, b2, a1, a2) = (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0);
        var y = new float[x.Length];
        float z0 = 0, z1 = 0;
        for (int n = 0; n < x.Length; n++)
        {
            float xn = x[n], yn = z0 + b0 * xn;
            z0 = z1 + b1 * xn - a1 * yn;
            z1 = b2 * xn - a2 * yn;
            y[n] = yn;
        }
        for (int n = 0; n < y.Length; n++) y[n] = Math.Clamp(y[n], -1f, 1f);
        return y;
    }
}
