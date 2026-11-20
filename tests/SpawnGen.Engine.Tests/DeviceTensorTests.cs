using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using SpawnDev.ILGPU;
using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Engine.Tests;

/// <summary>The accelerator the device test nodes use (one per test).</summary>
static class TestDevice
{
    public static Accelerator Accelerator = null!;
    /// <summary>Every device tensor the nodes made, to check what was freed.</summary>
    public static readonly List<ImageBatch> Made = [];
    /// <summary>What the host node saw: whether its input was on the host when it ran.</summary>
    public static bool? HostNodeSawHost;
}

/// <summary>Makes a 1 x 2 x 2 x 3 picture of <c>value</c> on the device (no host copy).</summary>
[Node(Category = "test"), DeviceTensors]
public sealed class DeviceFill
{
    public ImageBatch Execute(double value)
    {
        var buffer = TestDevice.Accelerator.Allocate1D<float>(12);
        buffer.MemSet((byte)0);
        var data = Enumerable.Repeat((float)value, 12).ToArray();
        buffer.View.CopyFromCPU(data);
        var image = ImageBatch.FromDevice(1, 2, 2, 3, buffer);
        TestDevice.Made.Add(image);
        return image;
    }
}

/// <summary>Adds 1 to every value on the device: a device node that takes and gives device tensors.</summary>
[Node(Category = "test"), DeviceTensors]
public sealed class DeviceAddOne
{
    public async Task<ImageBatch> ExecuteAsync(ImageBatch image)
    {
        var view = await image.ToDeviceAsync(TestDevice.Accelerator);
        var outBuf = TestDevice.Accelerator.Allocate1D<float>(image.Length);
        var host = await view.CopyToHostAsync(); // the test node's arithmetic; a real node runs a kernel
        outBuf.View.CopyFromCPU(host.Select(v => v + 1).ToArray());
        var result = ImageBatch.FromDevice(image.Shape[0], image.Shape[1], image.Shape[2], image.Shape[3], outBuf);
        TestDevice.Made.Add(result);
        return result;
    }
}

/// <summary>DeviceFill that always runs (never cached): its output is held by no cache entry.</summary>
[Node(Category = "test", AlwaysExecute = true), DeviceTensors]
public sealed class DeviceFillAlways
{
    public ImageBatch Execute(double value) => new DeviceFill().Execute(value);
}

/// <summary>A host node: reads Data (the executor must have brought the picture down).</summary>
[Node(Category = "test", IsOutput = true)]
public sealed class HostSum
{
    public UiOutput Execute(ImageBatch image)
    {
        TestDevice.HostNodeSawHost = image.OnHost;
        return new UiOutput("sum", new[] { (double)image.Data.Sum() });
    }
}

/// <summary>A device output node that passes its input on (as Save Animated WEBP does) and reports where it was.</summary>
[Node(Category = "test", IsOutput = true), DeviceTensors]
public sealed class DevicePassOn
{
    public (ImageBatch Image, UiOutput Ui) Execute(ImageBatch image) => (image, new UiOutput("where", new[] { image.OnHost ? "host" : "device" }));
}

public class DeviceTensorTests
{
    Context _context = null!;
    NodeRegistry _registry = null!;
    PromptExecutor _executor = null!;

    [SetUp]
    public void SetUp()
    {
        _context = Context.Create(b => b.CPU());
        TestDevice.Accelerator = _context.CreateCPUAccelerator(0);
        TestDevice.Made.Clear();
        TestDevice.HostNodeSawHost = null;
        _registry = new NodeRegistry().AddAssembly(typeof(DeviceTensorTests).Assembly);
        _executor = new PromptExecutor(_registry);
    }

    [TearDown]
    public void TearDown()
    {
        _executor.Cache.Clear();
        TestDevice.Accelerator.Dispose();
        _context.Dispose();
    }

    [Test]
    public async Task A_host_node_gets_a_device_tensor_on_the_host()
    {
        var g = new PromptGraph();
        g.Add("1", "DeviceFill", ("value", 2.0));
        g.Add("2", "HostSum", ("image", PromptInput.Link("1")));
        var r = await _executor.ExecuteAsync(g);
        Assert.That(TestDevice.HostNodeSawHost, Is.True, "the executor brings a host node's tensor inputs to the host before it runs");
        Assert.That(((double[])r.Outputs["2"].Ui!["sum"]!)[0], Is.EqualTo(24));
        Assert.That(TestDevice.Made[0].OnDevice, Is.True, "the device copy stays for later device nodes");
    }

    [Test]
    public async Task A_device_chain_never_touches_the_host()
    {
        var g = new PromptGraph();
        g.Add("1", "DeviceFill", ("value", 2.0));
        g.Add("2", "DeviceAddOne", ("image", PromptInput.Link("1")));
        g.Add("3", "DevicePassOn", ("image", PromptInput.Link("2")));
        var r = await _executor.ExecuteAsync(g);
        Assert.That(((string[])r.Outputs["3"].Ui!["where"]!)[0], Is.EqualTo("device"));
        Assert.That(TestDevice.Made.All(t => !t.OnHost), "no tensor of the chain was copied to the host");
        var passed = (ImageBatch)r.Outputs["3"].Values[0]!;
        await passed.ToHostAsync();
        Assert.That(passed.Data, Is.All.EqualTo(3f));
    }

    [Test]
    public async Task Clearing_the_cache_frees_the_device_buffers()
    {
        var g = new PromptGraph();
        g.Add("1", "DeviceFill", ("value", 2.0));
        g.Add("2", "DeviceAddOne", ("image", PromptInput.Link("1")));
        g.Add("3", "DevicePassOn", ("image", PromptInput.Link("2")));
        await _executor.ExecuteAsync(g);
        Assert.That(TestDevice.Made.All(t => t.OnDevice), "cached outputs keep their device buffers between runs");
        _executor.Cache.Clear();
        Assert.That(TestDevice.Made.All(t => !t.OnDevice), "the cache frees a tensor's buffer with the last entry holding it");
    }

    [Test]
    public async Task A_passed_on_tensor_lives_until_its_last_cache_entry_goes()
    {
        var g = new PromptGraph();
        g.Add("1", "DeviceFill", ("value", 1.0));
        g.Add("3", "DevicePassOn", ("image", PromptInput.Link("1")));
        await _executor.ExecuteAsync(g);
        var fill = TestDevice.Made[0];
        Assert.That(_executor.Cache.Holds(fill));
        // Rerun with a new value: DeviceFill's old entry stays (other signature); the old tensor is still held.
        g.Add("1", "DeviceFill", ("value", 5.0));
        await _executor.ExecuteAsync(g);
        Assert.That(fill.OnDevice, "still held by the first run's entry");
        _executor.Cache.Clear();
        Assert.That(fill.OnDevice, Is.False);
    }

    [Test]
    public async Task A_run_frees_the_device_tensors_no_entry_holds()
    {
        // DeviceFillAlways is never cached and HostSum is the output (handed back is its UI, not the picture): nothing holds the picture after the run.
        var g = new PromptGraph();
        g.Add("1", "DeviceFillAlways", ("value", 2.0));
        g.Add("2", "HostSum", ("image", PromptInput.Link("1")));
        await _executor.ExecuteAsync(g);
        var made = TestDevice.Made.Single();
        Assert.That(_executor.Cache.Holds(made), Is.False);
        Assert.That(made.OnDevice, Is.False, "its device buffer was freed when the run ended");
        Assert.That(made.Data, Is.All.EqualTo(2f), "the host copy the host node used stays readable");
    }

    [Test]
    public void A_device_only_tensor_says_why_Data_is_unavailable()
    {
        var buffer = TestDevice.Accelerator.Allocate1D<float>(12);
        using var image = ImageBatch.FromDevice(1, 2, 2, 3, buffer);
        var ex = Assert.Throws<InvalidOperationException>(() => _ = image.Data);
        Assert.That(ex!.Message, Does.Contain("ToHostAsync"));
    }
}
