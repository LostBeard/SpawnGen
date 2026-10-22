using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.SampleNodes;

/// <summary>The smallest kind of node: values in, value out. The class is found by its <c>[Node]</c> attribute; the sockets and widgets come from the Execute signature.</summary>
[Node(Category = "sample", DisplayName = "Reverse Text")]
public sealed class SampleReverseText
{
    public string Execute([Input(Multiline = true)] string text = "") => new string(text.Reverse().ToArray());
}

/// <summary>A node with a picture: mixes a colour into every pixel. Pictures are <see cref="ImageBatch"/>es of floats in 0..1 (batch, height, width, channels).</summary>
[Node(Category = "sample", DisplayName = "Tint Image")]
public sealed class SampleTintImage
{
    public ImageBatch Execute(ImageBatch image,
        [Input(Min = 0, Max = 1, Step = 0.01)] float red = 1f,
        [Input(Min = 0, Max = 1, Step = 0.01)] float green = 0.5f,
        [Input(Min = 0, Max = 1, Step = 0.01)] float blue = 0f,
        [Input(Min = 0, Max = 1, Step = 0.01)] float strength = 0.3f)
    {
        var result = new ImageBatch(image.Batch, image.Height, image.Width, image.Channels);
        float[] tint = [red, green, blue];
        for (int i = 0; i < image.Data.Length; i++)
        {
            int channel = i % image.Channels;
            result.Data[i] = channel < 3 ? image.Data[i] * (1 - strength) + tint[channel] * strength : image.Data[i];
        }
        return result;
    }
}
