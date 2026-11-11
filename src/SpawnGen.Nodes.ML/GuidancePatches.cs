namespace SpawnGen.Nodes.ML;

public static class GuidancePatches
{
    public static float[] RescaleCfg(CfgFunctionArgs a, double multiplier, bool isFlow) => throw new NotImplementedException();
    public static float[] CfgNormPost(PostCfgArgs a, double strength) => throw new NotImplementedException();
    public static float[] CfgNormPre(CfgFunctionArgs a, double strength) => throw new NotImplementedException();
    public static float[] CfgZeroStar(PostCfgArgs a) => throw new NotImplementedException();
    public static Func<PreCfgArgs, float[][]> Apg(double eta, double normThreshold, double momentum) => throw new NotImplementedException();
    public static float[] Mahiro(PostCfgArgs a) => throw new NotImplementedException();
    public static float[] TangentialDamping(float[] condScore, float[] uncondScore) => throw new NotImplementedException();
    public static float[][] Tcfg(PreCfgArgs a) => throw new NotImplementedException();
    public static float[] PerpNeg(float[] pos, float[] neg, float[] nocond, double negScale, double condScale) => throw new NotImplementedException();
    public static float[] FourierFilter(float[] x, int channels, int height, int width, double scaleLow, double scaleHigh, int cutoff) => throw new NotImplementedException();
    public static float[][] FreSca(PreCfgArgs a, double scaleLow, double scaleHigh, int cutoff) => throw new NotImplementedException();
    public static float[] EpsilonScaling(PostCfgArgs a, double factor) => throw new NotImplementedException();
    public static float[] TemporalScoreRescaling(PostCfgArgs a, double k, double variance) => throw new NotImplementedException();
    public static float[] RenormCfg(CfgFunctionArgs a, double cfgTrunc, double renormCfg) => throw new NotImplementedException();
    public static float[] DifferentialDiffusionMask(double sigma, float[] mask, double[] stepSigmas, ModelSampling sampling, double strength) => throw new NotImplementedException();
    public static float[] VideoLinearCfg(CfgFunctionArgs a, double minCfg) => throw new NotImplementedException();
    public static float[] VideoTriangleCfg(CfgFunctionArgs a, double minCfg) => throw new NotImplementedException();
    public static float[] Linspace(float start, float end, int steps) => throw new NotImplementedException();
}
