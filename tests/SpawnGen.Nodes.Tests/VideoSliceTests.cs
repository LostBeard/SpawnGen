using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Trim Video ("Video Slice") against ComfyUI (bef11fc, 2026-11-13): a 12-frame, 4 fps clip whose frame i holds i / 100, trimmed through Create
/// Video -> Video Slice -> Get Video Components. ComfyUI re-encodes the result, so its frame values come back lossy (frame 4 read 0.041); the kept
/// frames and errors below are what it gave, and SpawnGen keeps the frames themselves exactly.
/// </summary>
public class VideoSliceTests
{
    static VideoClip Clip(AudioClip? audio = null)
    {
        var frames = new ImageBatch(12, 2, 2, 3);
        for (int f = 0; f < 12; f++)
            for (int i = 0; i < 12; i++) frames.Data[f * 12 + i] = f / 100f;
        return new VideoClip(frames, 4.0, audio);
    }

    static int[] Kept(VideoClip v) => [.. Enumerable.Range(0, v.Frames.Batch).Select(f => (int)Math.Round(v.Frames.Data[f * 12] * 100))];

    [TestCase(0.0, 0.0, false, new[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 })]
    [TestCase(0.5, 1.0, false, new[] { 2, 3, 4, 5 })]
    [TestCase(0.6, 1.0, false, new[] { 3, 4, 5, 6 })]
    [TestCase(0.0, 1.1, false, new[] { 0, 1, 2, 3, 4 })]
    [TestCase(-1.0, 0.0, false, new[] { 8, 9, 10, 11 })]
    [TestCase(-1.3, 0.5, false, new[] { 7, 8 })]
    [TestCase(-1.0, 2.0, true, new[] { 8, 9, 10, 11 })]
    [TestCase(-5.0, 0.0, false, new[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 })]
    [TestCase(-5.0, 1.0, true, new[] { 0, 1, 2, 3 })]
    [TestCase(-0.5, 0.25, true, new[] { 10 })]
    [TestCase(2.5, 0.4, true, new[] { 10, 11 })]
    [TestCase(2.0, 1.0, false, new[] { 8, 9, 10, 11 })]
    [TestCase(0.0, 3.0, true, new[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 })]
    [TestCase(0.26, 0.26, false, new[] { 2 })]
    [TestCase(0.24, 0.5, false, new[] { 1, 2 })]
    [TestCase(0.125, 0.25, false, new[] { 1 })]
    [TestCase(1.0, 0.1, false, new[] { 4 })]
    [TestCase(2.9, 0.0, false, new int[0])]
    [TestCase(3.0, 0.0, false, new int[0])]
    public void The_kept_frames_are_ComfyUIs(double start, double duration, bool strict, int[] frames)
    {
        var trimmed = new VideoSlice().Execute(Clip(), start, duration, strict);
        Assert.That(Kept(trimmed), Is.EqualTo(frames));
        Assert.That(trimmed.Fps, Is.EqualTo(4.0));
    }

    [TestCase(2.5, 2.0, false, "Source duration: 3.0\nStart time: 2.5\nTarget duration: 2.0\n")]
    [TestCase(2.5, 2.0, true, "Source duration: 3.0\nStart time: 2.5\nTarget duration: 2.0\n")]
    [TestCase(2.5, 0.6, false, "Source duration: 3.0\nStart time: 2.5\nTarget duration: 0.6\n")]
    [TestCase(2.0, 1.01, false, "Source duration: 3.0\nStart time: 2.0\nTarget duration: 1.01\n")]
    [TestCase(0.0, 3.2, true, "Source duration: 3.0\nStart time: 0.0\nTarget duration: 3.2\n")]
    [TestCase(5.0, 0.0, false, "Source duration: 3.0\nStart time: 5.0\nTarget duration: 0.0\n")]
    public void Past_the_end_fails_with_ComfyUIs_message(double start, double duration, bool strict, string details)
    {
        var error = Assert.Throws<InvalidOperationException>(() => new VideoSlice().Execute(Clip(), start, duration, strict));
        Assert.That(error!.Message, Is.EqualTo("Failed to slice video:\n" + details));
    }

    [Test]
    public void The_sound_is_cut_to_the_same_stretch()
    {
        var samples = Enumerable.Range(0, 48000 * 3).Select(i => i / 1e6f).ToArray();
        var trimmed = new VideoSlice().Execute(Clip(new AudioClip([samples], 48000)), 0.5, 1.0, false);
        Assert.That(trimmed.Audio!.Length, Is.EqualTo(48000));
        Assert.That(trimmed.Audio.Channels[0][0], Is.EqualTo(24000 / 1e6f));
        var tail = new VideoSlice().Execute(Clip(new AudioClip([samples], 48000)), -1.0, 0.0, false);
        Assert.That(tail.Audio!.Length, Is.EqualTo(48000), "the last second");
        Assert.That(tail.Audio.Channels[0][^1], Is.EqualTo(samples[^1]));
    }
}
