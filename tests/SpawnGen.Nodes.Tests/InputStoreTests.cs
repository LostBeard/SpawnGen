using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// The in-memory input store (the browser's): a file the page read in JS is kept as the host's <see cref="StoredBytes"/> (no copy into .NET,
/// AGENTS.md rule 4), a file handed over as a managed array goes through the host's <c>fromManaged</c>, and host nodes still read bytes.
/// </summary>
public class InputStoreTests
{
    /// <summary>Stands in for the browser's JS bytes: counts how often the bytes were copied out to .NET.</summary>
    sealed class CountingBytes(byte[] data) : StoredBytes
    {
        public int Reads;
        public override long Length => data.LongLength;
        public override byte[] Read() { Reads++; return (byte[])data.Clone(); }
    }

    [Test]
    public async Task A_stored_file_is_kept_as_it_is_and_opened_without_a_copy()
    {
        var store = new MemoryInputStore();
        var file = new CountingBytes([1, 2, 3]);
        Assert.That(await store.SaveAsync("pics/a.png", file), Is.EqualTo("pics/a.png"));
        Assert.That(await store.OpenAsync("pics/a.png"), Is.SameAs(file), "the store hands back the file it was given");
        Assert.That(file.Reads, Is.Zero, "saving and opening never copied the bytes");
        Assert.That(store.Folders(), Is.EqualTo(new[] { "pics" }));
        Assert.That(await store.ReadAsync("pics/a.png"), Is.EqualTo(new byte[] { 1, 2, 3 }), "a host node still reads the bytes");
        Assert.That(file.Reads, Is.EqualTo(1));
    }

    [Test]
    public async Task A_managed_file_goes_through_the_hosts_wrapper()
    {
        var made = new List<CountingBytes>();
        var store = new MemoryInputStore(data => { var b = new CountingBytes(data); made.Add(b); return b; });
        await store.SaveAsync("a.png", new byte[] { 9, 8 });
        Assert.That(made, Has.Count.EqualTo(1));
        Assert.That(await store.OpenAsync("a.png"), Is.SameAs(made[0]));
    }

    [Test]
    public async Task Without_a_wrapper_a_managed_file_is_copied_so_later_changes_to_the_array_do_not_reach_it()
    {
        var store = new MemoryInputStore();
        var data = new byte[] { 5, 6 };
        await store.SaveAsync("a.png", data);
        data[0] = 0;
        Assert.That(await store.ReadAsync("a.png"), Is.EqualTo(new byte[] { 5, 6 }));
    }

    [Test]
    public void A_missing_file_throws_FileNotFound_from_open_and_read()
    {
        var store = new MemoryInputStore();
        Assert.ThrowsAsync<FileNotFoundException>(() => store.OpenAsync("none.png"));
        Assert.ThrowsAsync<FileNotFoundException>(() => store.ReadAsync("none.png"));
    }

    [Test]
    public async Task The_disk_store_opens_a_file_as_managed_bytes()
    {
        var dir = Directory.CreateTempSubdirectory("spawngen-inputs-");
        try
        {
            var store = new DiskInputStore(() => dir.FullName);
            await store.SaveAsync("a.png", new ManagedBytes([4, 5, 6]));
            var opened = await store.OpenAsync("a.png");
            Assert.That(opened, Is.InstanceOf<ManagedBytes>());
            Assert.That(opened.Read(), Is.EqualTo(new byte[] { 4, 5, 6 }));
        }
        finally { dir.Delete(true); }
    }
}
