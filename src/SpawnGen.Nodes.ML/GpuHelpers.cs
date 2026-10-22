using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnGen.Nodes.ML;

/// <summary>How a host hands the nodes their <see cref="MlRuntime"/> when creating it takes asynchronous work (the browser opens its GPU asynchronously).</summary>
public interface IMlRuntimeProvider
{
    Task<MlRuntime> GetAsync();
}

/// <summary>Host to GPU tensor plumbing shared by the ML nodes. Latents and embeddings are small, so values cross nodes on the host.</summary>
internal static class GpuHelpers
{
    static readonly ConditionalWeakTable<Accelerator, ElementWiseKernels> Kernels = new();

    /// <summary>The runtime: a registered <see cref="MlRuntime"/> (desktop, tests) or whatever the host's <see cref="IMlRuntimeProvider"/> creates (browser).</summary>
    public static async ValueTask<MlRuntime> RuntimeAsync(IServiceProvider services)
    {
        if (services.GetService(typeof(MlRuntime)) is MlRuntime runtime) return runtime;
        if (services.GetService(typeof(IMlRuntimeProvider)) is IMlRuntimeProvider provider) return await provider.GetAsync();
        throw new InvalidOperationException("No MlRuntime or IMlRuntimeProvider is registered; the ML nodes need one to reach the GPU.");
    }

    /// <summary>Uploads <paramref name="data"/> and wraps it as a tensor. Dispose the buffer when the run is done.</summary>
    public static (MemoryBuffer1D<float, Stride1D.Dense> Buffer, Tensor Tensor) Upload(Accelerator accelerator, float[] data, int[] shape)
    {
        var buffer = accelerator.Allocate1D(data);
        return (buffer, new Tensor(buffer.View, shape));
    }

    /// <summary>
    /// Reads the first <paramref name="count"/> values of a tensor back to the host. Asynchronous on purpose: a browser GPU cannot be read
    /// synchronously. The tensor may be a view into a larger pooled buffer, so it is copied out first.
    /// </summary>
    public static async Task<float[]> ReadAsync(Accelerator accelerator, Tensor tensor, int count)
    {
        var kernels = Kernels.GetValue(accelerator, a => new ElementWiseKernels(a));
        using var copy = accelerator.Allocate1D<float>(count);
        kernels.Scale(tensor.Data.SubView(0, count), copy.View, count, 1f);
        await accelerator.SynchronizeAsync();
        return await copy.CopyToHostAsync<float>(0, count);
    }
}
