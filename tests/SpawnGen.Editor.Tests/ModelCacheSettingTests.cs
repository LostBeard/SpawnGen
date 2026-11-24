namespace SpawnGen.Editor.Tests;

/// <summary>The model cache limit setting (StudioSettings.ModelCacheLimitGb): off by default, how it is stored, and the byte value the engine gets.</summary>
public class ModelCacheSettingTests
{
    [Test]
    public void By_default_nothing_is_removed()
    {
        var s = new StudioSettings();
        Assert.That(s.ModelCacheLimitGb, Is.EqualTo(0));
        Assert.That(s.ModelCacheBytes, Is.Null, "no limit: every downloaded model is kept until the user removes it");
        Assert.That(StudioSettings.FromJson(null).ModelCacheBytes, Is.Null);
        Assert.That(StudioSettings.FromJson("""{"GridSize": 12}""").ModelCacheBytes, Is.Null);
    }

    [Test]
    public void The_first_versions_saved_default_is_not_taken_as_the_users_choice()
    {
        // 2026-11-24, for a few hours: the setting was ModelCacheGb with a default of 24, written into every saved settings object
        Assert.That(StudioSettings.FromJson("""{"ModelCacheGb": 24}""").ModelCacheBytes, Is.Null);
    }

    [Test]
    public void A_chosen_limit_survives_a_save()
    {
        var back = StudioSettings.FromJson(new StudioSettings { ModelCacheLimitGb = 12.5 }.ToJson());
        Assert.That(back.ModelCacheLimitGb, Is.EqualTo(12.5));
        Assert.That(back.ModelCacheBytes, Is.EqualTo(12_500_000_000L));
    }

    [Test]
    public void A_damaged_value_falls_back_to_no_limit()
    {
        Assert.That(StudioSettings.FromJson("""{"ModelCacheLimitGb": -5}""").ModelCacheBytes, Is.Null);
        Assert.That(StudioSettings.FromJson("""{"ModelCacheLimitGb": 99999}""").ModelCacheBytes, Is.Null);
        Assert.That(StudioSettings.FromJson("""{"ModelCacheLimitGb": "lots"}""").ModelCacheBytes, Is.Null);
    }
}
