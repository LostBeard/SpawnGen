namespace SpawnGen.Engine.Nodes;

/// <summary>Converts graph literals and upstream values to the CLR type an <c>Execute</c> parameter expects.</summary>
public static class ValueCoercion
{
    public static bool TryCoerce(object? value, Type target, out object? result)
    {
        result = null;
        if (value is null) return false;
        target = Nullable.GetUnderlyingType(target) ?? target;

        if (target.IsInstanceOfType(value) && !target.IsEnum)
        {
            result = value;
            return true;
        }

        if (target.IsEnum)
        {
            if (value is string name && Enum.TryParse(target, name, ignoreCase: true, out var parsed) && Enum.IsDefined(target, parsed!))
            {
                result = parsed;
                return true;
            }
            if (value is long l && Enum.IsDefined(target, l))
            {
                result = Enum.ToObject(target, l);
                return true;
            }
            return false;
        }

        if (target == typeof(float) || target == typeof(double))
        {
            double d;
            switch (value)
            {
                case double dv: d = dv; break;
                case float fv: d = fv; break;
                case long lv: d = lv; break;
                case int iv: d = iv; break;
                default: return false;
            }
            // Box each branch on its own: a ternary over float and double would promote the float to double.
            result = target == typeof(float) ? (object)(float)d : (object)d;
            return true;
        }

        if (IsIntegral(target))
        {
            long l;
            switch (value)
            {
                case long lv: l = lv; break;
                case int iv: l = iv; break;
                case double dv when dv == Math.Floor(dv) && Math.Abs(dv) < 9e15: l = (long)dv; break;
                case float fv when fv == MathF.Floor(fv): l = (long)fv; break;
                default: return false;
            }
            try
            {
                result = Convert.ChangeType(l, target, System.Globalization.CultureInfo.InvariantCulture);
                return true;
            }
            catch (OverflowException) { return false; }
        }

        return false;
    }

    static bool IsIntegral(Type t) =>
        t == typeof(int) || t == typeof(long) || t == typeof(short) || t == typeof(byte) || t == typeof(uint) || t == typeof(ushort);

    /// <summary>Numeric value of a literal for range checks, or null when it is not a number.</summary>
    public static double? AsDouble(object? value) => value switch
    {
        long l => l,
        int i => i,
        double d => d,
        float f => f,
        _ => null,
    };
}
