using System.Buffers.Binary;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>WAV files without a dependency: reads PCM (8, 16, 24, 32 bit) and 32-bit float, any channel count; writes 16-bit PCM.</summary>
public static class WavFile
{
    public static bool LooksLikeWav(ReadOnlySpan<byte> data) =>
        data.Length >= 12 && data[0] == 'R' && data[1] == 'I' && data[2] == 'F' && data[3] == 'F' && data[8] == 'W' && data[9] == 'A' && data[10] == 'V' && data[11] == 'E';

    public static AudioClip Decode(byte[] data)
    {
        if (!LooksLikeWav(data)) throw new InvalidDataException("Not a WAV file.");
        int format = 0, channels = 0, rate = 0, bits = 0;
        int dataStart = -1, dataLength = 0;
        for (int pos = 12; pos + 8 <= data.Length;)
        {
            string id = System.Text.Encoding.ASCII.GetString(data, pos, 4);
            int size = (int)Math.Min(BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(pos + 4)), (uint)(data.Length - pos - 8));
            if (id == "fmt " && size >= 16)
            {
                format = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(pos + 8));
                channels = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(pos + 10));
                rate = (int)BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(pos + 12));
                bits = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(pos + 22));
                // WAVE_FORMAT_EXTENSIBLE: the real format is the first two bytes of the sub-format GUID.
                if (format == 0xFFFE && size >= 26) format = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(pos + 32));
            }
            else if (id == "data") { dataStart = pos + 8; dataLength = size; break; }
            pos += 8 + size + (size & 1);
        }
        if (channels == 0 || dataStart < 0) throw new InvalidDataException("The WAV file has no format or no sound data.");
        bool isFloat = format == 3;
        if (!(format == 1 && bits is 8 or 16 or 24 or 32) && !(isFloat && bits == 32)) throw new NotSupportedException($"WAV format {format} with {bits} bits is not supported (PCM 8/16/24/32 and float 32 are).");

        int bytesPerSample = bits / 8, frames = dataLength / (bytesPerSample * channels);
        var result = new float[channels][];
        for (int c = 0; c < channels; c++) result[c] = new float[frames];
        for (int f = 0; f < frames; f++)
            for (int c = 0; c < channels; c++)
            {
                int p = dataStart + (f * channels + c) * bytesPerSample;
                result[c][f] = bits switch
                {
                    8 => (data[p] - 128) / 128f,
                    16 => BinaryPrimitives.ReadInt16LittleEndian(data.AsSpan(p)) / 32768f,
                    24 => ((data[p] | (data[p + 1] << 8) | (data[p + 2] << 16)) << 8 >> 8) / 8388608f,
                    _ => isFloat ? BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(p)) : BinaryPrimitives.ReadInt32LittleEndian(data.AsSpan(p)) / 2147483648f,
                };
            }
        return new AudioClip(result, rate);
    }

    /// <summary>16-bit PCM; samples outside -1..1 are clipped.</summary>
    public static byte[] Encode(AudioClip clip)
    {
        int channels = clip.ChannelCount, frames = clip.Length, dataLength = frames * channels * 2;
        var bytes = new byte[44 + dataLength];
        System.Text.Encoding.ASCII.GetBytes("RIFF").CopyTo(bytes, 0);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4), (uint)(36 + dataLength));
        System.Text.Encoding.ASCII.GetBytes("WAVEfmt ").CopyTo(bytes, 8);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(16), 16);
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(20), 1);
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(22), (ushort)channels);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(24), (uint)clip.SampleRate);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(28), (uint)(clip.SampleRate * channels * 2));
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(32), (ushort)(channels * 2));
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(34), 16);
        System.Text.Encoding.ASCII.GetBytes("data").CopyTo(bytes, 36);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(40), (uint)dataLength);
        for (int f = 0; f < frames; f++)
            for (int c = 0; c < channels; c++)
            {
                // The same scale the reader divides by (32768), so a clip survives a round trip within half a step; +1 itself clips to 32767.
                int v = (int)Math.Clamp(MathF.Round(clip.Channels[c][f] * 32768f), -32768f, 32767f);
                BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(44 + (f * channels + c) * 2), (short)v);
            }
        return bytes;
    }
}
