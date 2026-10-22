using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnGen.Nodes.ML;

/// <summary>Host to GPU tensor plumbing shared by the ML nodes. Latents and embeddings are small, so values cross nodes on the host.</summary>
internal static class GpuHelpers
{
    public static MlRuntime Runtime(IServiceProvider services) =>
        services.GetService(typeof(MlRuntime)) as MlRuntime
        ?? throw new InvalidOperationException("MlRuntime is not registered in the service provider; the ML nodes need it to reach the GPU.");

    /// <summary>Uploads <paramref name="data"/> and wraps it as a tensor. Dispose the buffer when the run is done.</summary>
    public static (MemoryBuffer1D<float, Stride1D.Dense> Buffer, Tensor Tensor) Upload(Accelerator accelerator, float[] data, int[] shape)
    {
        var buffer = accelerator.Allocate1D(data);
        return (buffer, new Tensor(buffer.View, shape));
    }

    public static async Task<float[]> ReadAsync(Accelerator accelerator, Tensor tensor, int count)
    {
        await accelerator.SynchronizeAsync();
        return tensor.Data.SubView(0, count).GetAsArray1D();
    }
}
