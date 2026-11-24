namespace SpawnGen.Editor.Tests;

/// <summary>The advice shown after a GPU device failure, against the error texts the RX 590 produced in Chrome (2026-11-23 / 24).</summary>
public class RunFailureAdviceTests
{
    [Test]
    public void A_lost_device_says_queue_again()
    {
        var advice = RunFailureAdvice.For("WebGPU device has been lost and cannot accept commands.");
        Assert.That(advice, Is.EqualTo(RunFailureAdvice.DeviceLost));
        Assert.That(advice, Does.Contain("Queue again"));
        Assert.That(RunFailureAdvice.For("[GE final sync, 225 ops total] WebGPU device has been lost and cannot accept commands. || reclaims-this-process=0"),
            Is.EqualTo(RunFailureAdvice.DeviceLost));
    }

    [Test]
    public void A_refused_new_device_says_restart_the_browser_not_queue_again()
    {
        var advice = RunFailureAdvice.For(
            "Failed to execute 'requestDevice' on 'GPUAdapter': D3D12 create command queue failed with DXGI_ERROR_DEVICE_REMOVED (0x887A0005)");
        Assert.That(advice, Is.EqualTo(RunFailureAdvice.NoDevice));
        Assert.That(advice, Does.Contain("close the browser completely"));
        Assert.That(RunFailureAdvice.For("test hook: the browser refused a new GPU device"), Is.EqualTo(RunFailureAdvice.NoDevice));
    }

    [Test]
    public void Other_failures_get_no_gpu_advice()
    {
        Assert.That(RunFailureAdvice.For("guide level 1 has 5341 voxels, decoder has 0"), Is.Empty);
        Assert.That(RunFailureAdvice.For("A requested file or directory could not be found at the time an operation was processed."), Is.Empty);
        Assert.That(RunFailureAdvice.For(null), Is.Empty);
    }
}
