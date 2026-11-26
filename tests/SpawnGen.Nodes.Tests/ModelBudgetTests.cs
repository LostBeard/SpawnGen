using ILGPU;
using ILGPU.Runtime.CPU;
using SpawnDev.ILGPU.ML;
using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Models kept in device memory by a budget (MlRuntime over SpawnDev.ILGPU.ML's ModelResidency, TJ's direction of 2026-11-25): with a budget
/// set every model stays loaded while the kept ones fit it, whatever workflow loaded them; the least recently used idle one goes when the
/// next would not fit; a model the running prompt uses is never dropped; with no budget (and a device that reports no free memory, as
/// the CPU accelerator here and every browser) the older rules stay. Sizes are stand-ins: the decisions are what is tested, on any budget,
/// so a machine far larger than the test PC is covered by a budget far larger than the models.
/// </summary>
public class ModelBudgetTests
{
    public sealed class Sized(string name, long bytes) : IDisposable, IDeviceSized
    {
        public static readonly List<string> Events = [];
        public string Name => name;
        public bool Disposed { get; private set; }
        public long DeviceBytes => Disposed ? 0 : bytes;
        public void Dispose() { Disposed = true; lock (Events) Events.Add("dispose " + name); }
        public static Task<Sized> Load(string name, long bytes) { lock (Events) Events.Add("load " + name); return Task.FromResult(new Sized(name, bytes)); }
    }

    /// <summary>A loader node: its output IS the model (as MODEL / CLIP / VAE outputs are), so the result cache holds it.</summary>
    [Node(Category = "test")]
    public sealed class LoadsModel
    {
        public static int Runs;
        public async Task<Sized> ExecuteAsync(IServiceProvider services, string name = "a", long bytes = 60L * 1024 * 1024)
        {
            Runs++;
            var runtime = (MlRuntime)services.GetService(typeof(MlRuntime))!;
            return await runtime.GetOrLoadAsync("model:" + name, () => Sized.Load(name, bytes), bytes);
        }
    }

    [Node(Category = "test", IsOutput = true)]
    public sealed class UsesModel
    {
        public string Execute(Sized model, string tag = "") => model.Disposed ? throw new ObjectDisposedException(model.Name) : model.Name + tag;
    }

    sealed class Services(params object[] instances) : IServiceProvider
    {
        public object? GetService(Type serviceType) => instances.FirstOrDefault(serviceType.IsInstanceOfType);
    }

    static long Mb(long megabytes) => megabytes * 1024 * 1024;

    static string[] Seen() { lock (Sized.Events) return [.. Sized.Events]; }

    static MlRuntime Runtime(ILGPU.Runtime.Accelerator cpu, long? budget)
    {
        lock (Sized.Events) Sized.Events.Clear();
        return new MlRuntime(new DesktopModelSource(Path.GetTempPath(), MlRuntime.RepoFolders), cpu) { ModelMemoryBytes = budget };
    }

    [Test]
    public async Task Without_a_budget_the_manager_is_not_used()
    {
        using var context = Context.Create(b => b.CPU());
        using var cpu = context.CreateCPUAccelerator(0);
        using var runtime = Runtime(cpu, null);
        Assert.That(runtime.KeepsModelsByBudget, Is.False, "the CPU accelerator reports no free memory, like a browser");
        var a = await runtime.GetOrLoadAsync("model:a", () => Sized.Load("a", Mb(60)), Mb(60));
        var again = await runtime.GetOrLoadAsync("model:a", () => Sized.Load("a", Mb(60)), Mb(60));
        Assert.Multiple(() =>
        {
            Assert.That(again, Is.SameAs(a));
            Assert.That(runtime.Residency.Resident, Is.Empty);
            Assert.That(Seen(), Is.EqualTo(new[] { "load a" }));
        });
    }

    [Test]
    public async Task A_large_budget_keeps_every_workflows_models()
    {
        using var context = Context.Create(b => b.CPU());
        using var cpu = context.CreateCPUAccelerator(0);
        using var runtime = Runtime(cpu, Mb(1_000_000));
        Assert.That(runtime.KeepsModelsByBudget, Is.True);
        var a = await runtime.GetOrLoadAsync("model:a", () => Sized.Load("a", Mb(60)), Mb(60));
        await runtime.EndPromptAsync();
        var b = await runtime.GetOrLoadAsync("model:b", () => Sized.Load("b", Mb(60)), Mb(60));
        await runtime.EndPromptAsync();
        var c = await runtime.GetOrLoadPipelineAsync("pipe:c", () => Sized.Load("c", Mb(60)), Mb(60));
        await runtime.EndPromptAsync();
        var aAgain = await runtime.GetOrLoadAsync("model:a", () => Sized.Load("a", Mb(60)), Mb(60));
        await runtime.EndPromptAsync();
        Assert.Multiple(() =>
        {
            Assert.That(aAgain, Is.SameAs(a), "the first workflow's model was still loaded");
            Assert.That(Seen(), Is.EqualTo(new[] { "load a", "load b", "load c" }), "nothing was dropped and nothing loaded twice");
            Assert.That(runtime.Residency.Resident.Select(m => m.Key).Order(), Is.EqualTo(new[] { "model:a", "model:b", "pipe:c" }));
            Assert.That(a.Disposed || b.Disposed || c.Disposed, Is.False);
        });
    }

    [Test]
    public async Task A_small_budget_drops_the_least_recently_used_idle_model_when_the_next_does_not_fit()
    {
        using var context = Context.Create(b => b.CPU());
        using var cpu = context.CreateCPUAccelerator(0);
        using var runtime = Runtime(cpu, Mb(130));
        var dropped = new List<string>();
        runtime.ModelDropped += dropped.Add;
        var a = await runtime.GetOrLoadAsync("model:a", () => Sized.Load("a", Mb(60)), Mb(60));
        await runtime.EndPromptAsync();
        var b = await runtime.GetOrLoadAsync("model:b", () => Sized.Load("b", Mb(60)), Mb(60));
        await runtime.EndPromptAsync();
        Assert.That(dropped, Is.Empty, "two models of 60 fit 130");
        // a is used again, so b is now the least recently used; a third model of 60 needs room
        await runtime.GetOrLoadAsync("model:a", () => Sized.Load("a", Mb(60)), Mb(60));
        await runtime.EndPromptAsync();
        var c = await runtime.GetOrLoadAsync("model:c", () => Sized.Load("c", Mb(60)), Mb(60));
        await runtime.EndPromptAsync();
        Assert.Multiple(() =>
        {
            Assert.That(dropped, Is.EqualTo(new[] { "model:b" }));
            Assert.That(b.Disposed, Is.True);
            Assert.That(a.Disposed || c.Disposed, Is.False);
            Assert.That(runtime.Residency.Resident.Select(m => m.Key).Order(), Is.EqualTo(new[] { "model:a", "model:c" }));
        });
        // the dropped one loads again when it is next used
        var bAgain = await runtime.GetOrLoadAsync("model:b", () => Sized.Load("b", Mb(60)), Mb(60));
        await runtime.EndPromptAsync();
        Assert.That(bAgain, Is.Not.SameAs(b));
        Assert.That(Seen().Count(e => e == "load b"), Is.EqualTo(2));
    }

    [Test]
    public async Task A_model_the_running_prompt_uses_is_never_dropped()
    {
        using var context = Context.Create(b => b.CPU());
        using var cpu = context.CreateCPUAccelerator(0);
        using var runtime = Runtime(cpu, Mb(100));
        var dropped = new List<string>();
        runtime.ModelDropped += dropped.Add;
        // one prompt uses two models that together pass the budget: both stay (the second loads over the budget, with a log line)
        var a = await runtime.GetOrLoadAsync("model:a", () => Sized.Load("a", Mb(60)), Mb(60));
        var b = await runtime.GetOrLoadAsync("model:b", () => Sized.Load("b", Mb(60)), Mb(60));
        var aAgain = await runtime.GetOrLoadAsync("model:a", () => Sized.Load("a", Mb(60)), Mb(60));
        Assert.Multiple(() =>
        {
            Assert.That(dropped, Is.Empty);
            Assert.That(a.Disposed || b.Disposed, Is.False);
            Assert.That(aAgain, Is.SameAs(a));
            Assert.That(runtime.Residency.Resident.Single(m => m.Key == "model:a").Leases, Is.EqualTo(1), "one lease per model and prompt");
        });
        await runtime.EndPromptAsync();
        Assert.That(runtime.Residency.Resident.All(m => m.Leases == 0), Is.True, "the prompt's leases ended");
        // the next prompt's model makes room among the idle ones
        await runtime.GetOrLoadAsync("model:c", () => Sized.Load("c", Mb(60)), Mb(60));
        Assert.That(dropped, Is.Not.Empty);
        await runtime.EndPromptAsync();
    }

    [Test]
    public async Task Loader_nodes_run_again_each_prompt_and_get_the_kept_model()
    {
        using var context = Context.Create(b => b.CPU());
        using var cpu = context.CreateCPUAccelerator(0);
        using var runtime = Runtime(cpu, Mb(130));
        var executor = new PromptExecutor(new NodeRegistry().Add<LoadsModel>().Add<UsesModel>(), new Services(runtime)) { ReacquireModels = () => runtime.KeepsModelsByBudget };
        runtime.ModelDropped += _ => executor.Cache.DropModelEntries();
        LoadsModel.Runs = 0;

        int turn = 0;
        async Task Run(string name)
        {
            // the node that uses the model has something new to do each time (a cached result that needs no model asks for none)
            var g = new PromptGraph();
            g.Add("1", "LoadsModel", ("name", name));
            g.Add("2", "UsesModel", ("model", PromptInput.Link("1")), ("tag", (++turn).ToString()));
            try { await executor.ExecuteAsync(g); }
            finally { await runtime.EndPromptAsync(); }
        }

        await Run("a");
        await Run("a");
        Assert.Multiple(() =>
        {
            Assert.That(LoadsModel.Runs, Is.EqualTo(2), "the loader ran again (its cached output held a model), leasing it for the second prompt");
            Assert.That(Seen(), Is.EqualTo(new[] { "load a" }), "and got the kept model: nothing loaded twice");
        });
        // three workflows in turn under a budget for two: the oldest goes, and coming back to it works (it loads again; no disposed model is used)
        await Run("b");
        await Run("c");
        await Run("a");
        Assert.Multiple(() =>
        {
            Assert.That(Seen().Count(e => e == "load a"), Is.EqualTo(2), "a was dropped for c and loaded again");
            Assert.That(Seen().Count(e => e.StartsWith("dispose ")), Is.GreaterThanOrEqualTo(2));
            Assert.That(runtime.Residency.Resident.Count, Is.EqualTo(2));
        });
    }

    [Test]
    public async Task A_model_of_unknown_size_is_taken_to_be_as_large_as_the_largest_measured()
    {
        using var context = Context.Create(b => b.CPU());
        using var cpu = context.CreateCPUAccelerator(0);
        using var runtime = Runtime(cpu, Mb(150));
        var a = await runtime.GetOrLoadAsync("model:a", () => Sized.Load("a", Mb(60)), Mb(60));
        await runtime.EndPromptAsync();
        // b's size is not given and it never loaded before: room is made for the largest size measured so far (a's 60), which fits
        var b = await runtime.GetOrLoadAsync("model:b", () => Sized.Load("b", Mb(60)));
        await runtime.EndPromptAsync();
        Assert.That(a.Disposed, Is.False, "150 holds a and a model taken to be as large as a");
        Assert.That(runtime.Residency.KnownSizes["model:b"], Is.EqualTo(Mb(60)), "b's size was measured");
        // c, also unknown, is taken to be 60 too: 180 would pass 150, so the least recently used (a) goes first
        var c = await runtime.GetOrLoadAsync("model:c", () => Sized.Load("c", Mb(60)));
        await runtime.EndPromptAsync();
        Assert.Multiple(() =>
        {
            Assert.That(a.Disposed, Is.True);
            Assert.That(b.Disposed || c.Disposed, Is.False);
        });
    }

    /// <summary>A stand-in for the lazy MODEL / CLIP / VAE objects: nothing on the device until its first use, inside the prompt that leased it.</summary>
    public sealed class Lazy(string name) : IDisposable, IDeviceSized
    {
        public long Held;
        public bool Disposed { get; private set; }
        public long DeviceBytes => Disposed ? 0 : Held;
        public string Name => name;
        public void Dispose() { Disposed = true; Held = 0; }
    }

    [Test]
    public async Task A_lazy_model_asks_for_room_before_it_loads_and_its_size_is_learned_when_its_prompt_ends()
    {
        using var context = Context.Create(b => b.CPU());
        using var cpu = context.CreateCPUAccelerator(0);
        using var runtime = Runtime(cpu, Mb(150));
        var dropped = new List<string>();
        runtime.ModelDropped += dropped.Add;
        var a = await runtime.GetOrLoadAsync("model:a", () => Sized.Load("a", Mb(100)), Mb(100));
        await runtime.EndPromptAsync();
        // the lazy model is acquired empty (a stays: nothing is known to be needed yet beyond a model of a's size... which fits? no: 100 + 100 > 150)
        var lazy = await runtime.GetOrLoadAsync("model:lazy", () => Task.FromResult(new Lazy("lazy")), Mb(1));
        Assert.That(a.Disposed, Is.False, "an empty wrapper with a stated tiny size drops nothing");
        // right before its weights load it asks for their room: the idle model goes
        await runtime.MakeRoomAsync(lazy, Mb(100));
        Assert.That(dropped, Is.EqualTo(new[] { "model:a" }));
        lazy.Held = Mb(100);
        await runtime.EndPromptAsync();
        Assert.That(runtime.Residency.KnownSizes["model:lazy"], Is.EqualTo(Mb(100)), "read again from the model when its lease ended");
    }

    [Test]
    public async Task Without_a_budget_making_room_is_the_old_take_turns_rule()
    {
        using var context = Context.Create(b => b.CPU());
        using var cpu = context.CreateCPUAccelerator(0);
        using var runtime = Runtime(cpu, null);
        await runtime.MakeRoomAsync();   // nothing loaded: nothing to do, and the manager is not touched
        Assert.That(runtime.Residency.Resident, Is.Empty);
    }

    [Test]
    public async Task Measured_sizes_carry_over_to_the_next_session()
    {
        using var context = Context.Create(b => b.CPU());
        using var cpu = context.CreateCPUAccelerator(0);
        string json;
        using (var first = Runtime(cpu, Mb(150)))
        {
            await first.GetOrLoadAsync("model:a", () => Sized.Load("a", Mb(100)));   // size not given: measured
            await first.GetOrLoadAsync("model:b", () => Sized.Load("b", Mb(40)));
            await first.EndPromptAsync();
            json = first.ModelSizesJson();
        }
        Assert.That(json, Is.EqualTo($"{{\"model:a\":{Mb(100)},\"model:b\":{Mb(40)}}}"));
        using var second = Runtime(cpu, Mb(150));
        Assert.Multiple(() =>
        {
            Assert.That(second.SeedModelSizes(json), Is.EqualTo(2));
            Assert.That(second.SeedModelSizes("not json"), Is.EqualTo(0));
            Assert.That(second.SeedModelSizes(null), Is.EqualTo(0));
        });
        // in the new session a (100) is loaded; b, never loaded here and given no size, is known to be 40: both fit 150, a stays.
        // (unknown, b would be taken to be as large as the largest measured, 100, and a would go)
        var a = await second.GetOrLoadAsync("model:a", () => Sized.Load("a", Mb(100)));
        await second.EndPromptAsync();
        await second.GetOrLoadAsync("model:b", () => Sized.Load("b", Mb(40)));
        await second.EndPromptAsync();
        Assert.That(a.Disposed, Is.False);
    }

    [Test]
    public async Task A_lost_device_forgets_the_models_without_touching_them()
    {
        using var context = Context.Create(b => b.CPU());
        using var cpu = context.CreateCPUAccelerator(0);
        using var runtime = Runtime(cpu, Mb(1000));
        var a = await runtime.GetOrLoadAsync("model:a", () => Sized.Load("a", Mb(60)), Mb(60));
        runtime.DeviceLost();
        Assert.Multiple(() =>
        {
            Assert.That(a.Disposed, Is.False, "nothing is disposed into a dead device");
            Assert.That(runtime.Residency.Resident, Is.Empty);
        });
    }
}
