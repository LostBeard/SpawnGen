namespace SpawnGen.Editor;

/// <summary>
/// What to tell the user after a run failed on the GPU, appended to the failure's own text. Two cases, both measured in Chrome on an AMD
/// RX 590 (2026-11-23 / 24):
/// the device was lost during the run (the driver reset it): the next Queue opens a new device;
/// the browser then refuses a new device ("requestDevice ... DXGI_ERROR_DEVICE_REMOVED"): after a real driver reset Chrome gave that page no
/// device at all (asked from plain JavaScript 1, 16 and 46 seconds later), while a newly started browser got one at once, so no Queue in
/// that page can work and the advice is to restart the browser.
/// </summary>
public static class RunFailureAdvice
{
    public const string DeviceLost =
        " The GPU was reset (often a run that needed more GPU memory than was free). Queue again: SpawnGen opens the GPU anew."
        + " If that fails too, close the browser completely and open it again.";

    public const string NoDevice =
        " The browser is giving this page no GPU after the reset, and queueing again will not change that:"
        + " close the browser completely (every window) and open it again.";

    /// <summary>The advice for <paramref name="failure"/> (a run's error text), or "" when it is not a GPU device failure.</summary>
    public static string For(string? failure)
    {
        if (string.IsNullOrEmpty(failure)) return "";
        if (failure.Contains("requestDevice", StringComparison.OrdinalIgnoreCase) || failure.Contains("DEVICE_REMOVED", StringComparison.OrdinalIgnoreCase)
            || failure.Contains("refused a new GPU device", StringComparison.OrdinalIgnoreCase))
            return NoDevice;
        if (failure.Contains("device has been lost", StringComparison.OrdinalIgnoreCase) || failure.Contains("device was lost", StringComparison.OrdinalIgnoreCase))
            return DeviceLost;
        return "";
    }
}
