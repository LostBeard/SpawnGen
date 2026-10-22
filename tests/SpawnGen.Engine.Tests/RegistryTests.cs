using SpawnGen.Engine.Nodes;

namespace SpawnGen.Engine.Tests;

public class RegistryTests
{
    static NodeRegistry Registry() => new NodeRegistry().AddAssembly(typeof(RegistryTests).Assembly);

    [Test]
    public void Schema_comes_from_the_execute_signature()
    {
        var s = Registry().GetSchema("AddNumbers")!;
        Assert.That(s.DisplayName, Is.EqualTo("Add Numbers"));
        Assert.That(s.Category, Is.EqualTo("test"));
        Assert.That(s.Inputs.Select(i => (i.Name, i.Type.Name, i.Required)),
            Is.EqualTo(new[] { ("a", "INT", true), ("b", "INT", false) }));
        Assert.That(s.Outputs.Select(o => o.Type.Name), Is.EqualTo(new[] { "INT" }));
    }

    [Test]
    public void Parameter_names_are_snake_case_so_comfy_workflows_round_trip()
    {
        var s = Registry().GetSchema("Widgets")!;
        Assert.That(s.Inputs.Select(i => i.Name), Is.EqualTo(new[] { "steps", "cfg_scale", "text", "flag", "mode", "pick" }));
    }

    [Test]
    public void Widgets_carry_range_default_and_options()
    {
        var s = Registry().GetSchema("Widgets")!;
        var steps = s.FindInput("steps")!.Widget!;
        Assert.That((steps.Kind, steps.Min, steps.Max, steps.Step, steps.Default), Is.EqualTo((WidgetKind.Int, 0.0, 100.0, 5.0, (object)20L)));
        Assert.That(s.FindInput("cfg_scale")!.Widget!.Kind, Is.EqualTo(WidgetKind.Float));
        Assert.That(s.FindInput("text")!.Widget!.Multiline, Is.True);
        Assert.That(s.FindInput("flag")!.Widget!.Kind, Is.EqualTo(WidgetKind.Boolean));
        var mode = s.FindInput("mode")!.Widget!;
        Assert.That((mode.Kind, mode.Default), Is.EqualTo((WidgetKind.Combo, (object)"Fast")));
        Assert.That(mode.ResolveOptions(), Is.EqualTo(new[] { "Fast", "Slow" }));
        Assert.That(s.FindInput("pick")!.Widget!.ResolveOptions(), Is.EqualTo(new[] { "a", "b" }));
    }

    [Test]
    public void Options_provider_is_evaluated_on_demand()
    {
        var w = Registry().GetSchema("ModelPicker")!.FindInput("model")!.Widget!;
        Assert.That(w.Kind, Is.EqualTo(WidgetKind.Combo));
        Assert.That(w.ResolveOptions(), Is.EqualTo(new[] { "sd-turbo", "flux2" }));
    }

    [Test]
    public void Tuple_element_names_become_output_names()
    {
        var s = Registry().GetSchema("Split")!;
        Assert.That(s.Outputs.Select(o => (o.Name, o.Type.Name)),
            Is.EqualTo(new[] { ("low", "INT"), ("high", "INT"), ("label", "STRING") }));
    }

    [Test]
    public void Ui_output_is_not_a_socket_and_marks_nothing_extra()
    {
        var s = Registry().GetSchema("Sink")!;
        Assert.That(s.Outputs, Is.Empty);
        Assert.That(s.IsOutputNode, Is.True);
    }

    [Test]
    public void Socket_attribute_names_value_classes()
    {
        var s = Registry().GetSchema("ImageToLatent")!;
        Assert.That(s.FindInput("image")!.Type, Is.EqualTo(SocketType.Image));
        Assert.That(s.FindInput("image")!.Widget, Is.Null);
        Assert.That(s.Outputs[0].Type, Is.EqualTo(SocketType.Latent));
    }

    [Test]
    public void Async_and_context_parameters_are_not_inputs()
    {
        var s = Registry().GetSchema("AsyncDouble")!;
        Assert.That(s.Inputs.Select(i => i.Name), Is.EqualTo(new[] { "value" }));
    }

    [Test]
    public void A_node_without_exactly_one_execute_is_rejected()
    {
        Assert.Throws<InvalidOperationException>(() => new NodeRegistry().Add<NoExecute>());
    }

    public sealed class NoExecute { } // deliberately unmarked so AddAssembly skips it

    [TestCase("latentImage", "latent_image")]
    [TestCase("cfg", "cfg")]
    [TestCase("CFGScale", "cfg_scale")]
    [TestCase("batchSize", "batch_size")]
    [TestCase("seed2", "seed2")]
    public void Snake_case(string input, string expected) =>
        Assert.That(NodeRegistry.ToSnakeCase(input), Is.EqualTo(expected));

    [Test]
    public void Socket_types_accept_widening_and_unions()
    {
        Assert.That(SocketType.Float.Accepts(SocketType.Int), Is.True);
        Assert.That(SocketType.Int.Accepts(SocketType.Float), Is.False);
        Assert.That(new SocketType("INT,FLOAT").Accepts(SocketType.Float), Is.True);
        Assert.That(SocketType.Any.Accepts(SocketType.Image), Is.True);
        Assert.That(SocketType.Image.Accepts(SocketType.Latent), Is.False);
    }
}
