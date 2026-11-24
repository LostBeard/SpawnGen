namespace SpawnGen.Editor.Tests;

/// <summary>The model cache limit setting (StudioSettings.ModelCacheGb): its default, how it is stored, and the byte value the engine gets.</summary>
public class ModelCacheSettingTests
{
    [Test]
    public void The_default_is_a_limit_not_unlimited()
    {
        var s = new StudioSettings();
        Assert.That(s.ModelCacheGb, Is.EqualTo(StudioSettings.DefaultModelCacheGb));
        Assert.That(s.ModelCacheBytes, Is.EqualTo((long)(StudioSettings.DefaultModelCacheGb * 1e9)));
        Assert.That(StudioSettings.FromJson(null).ModelCacheBytes, Is.Not.Null, "settings saved before the limit existed get the default, not no limit");
        Assert.That(StudioSettings.FromJson("""{"GridSize": 12}""").ModelCacheGb, Is.EqualTo(StudioSettings.DefaultModelCacheGb));
    }

    [Test]
    public void It_survives_a_save_and_zero_means_no_limit()
    {
        var back = StudioSettings.FromJson(new StudioSettings { ModelCacheGb = 12.5 }.ToJson());
        Assert.That(back.ModelCacheGb, Is.EqualTo(12.5));
        Assert.That(back.ModelCacheBytes, Is.EqualTo(12_500_000_000L));
        var none = StudioSettings.FromJson(new StudioSettings { ModelCacheGb = 0 }.ToJson());
        Assert.That(none.ModelCacheGb, Is.EqualTo(0));
        Assert.That(none.ModelCacheBytes, Is.Null);
    }

    [Test]
    public void A_damaged_value_falls_back_to_the_default()
    {
        Assert.That(StudioSettings.FromJson("""{"ModelCacheGb": -5}""").ModelCacheGb, Is.EqualTo(StudioSettings.DefaultModelCacheGb));
        Assert.That(StudioSettings.FromJson("""{"ModelCacheGb": 99999}""").ModelCacheGb, Is.EqualTo(StudioSettings.DefaultModelCacheGb));
        Assert.That(StudioSettings.FromJson("""{"ModelCacheGb": "lots"}""").ModelCacheGb, Is.EqualTo(StudioSettings.DefaultModelCacheGb));
    }
}
