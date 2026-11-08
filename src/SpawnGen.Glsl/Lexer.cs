using System.Globalization;
using System.Text;

namespace SpawnGen.Glsl;

/// <summary>A compile error with the line it was found on, as GL's info log reports one.</summary>
public sealed class GlslException(string message, int line) : Exception(line > 0 ? $"ERROR: 0:{line}: {message}" : $"ERROR: {message}")
{
    public int Line { get; } = line;
}

internal enum TokenKind { Ident, IntLit, UIntLit, FloatLit, Punct, End }

internal readonly record struct Token(TokenKind Kind, string Text, int Line)
{
    public bool Is(string punct) => Kind == TokenKind.Punct && Text == punct;
    public override string ToString() => Kind == TokenKind.End ? "end of file" : $"'{Text}'";
}

/// <summary>GLSL ES 3.00 tokens. Preprocessor lines are read here: #version, #extension and #pragma (only "#pragma passes N" means something) are kept aside;
/// macros and conditionals are not supported.</summary>
internal static class Lexer
{
    static readonly string[] Puncts =
    [
        "<<=", ">>=", "++", "--", "<<", ">>", "<=", ">=", "==", "!=", "&&", "||", "^^", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=",
        "(", ")", "[", "]", "{", "}", ".", ",", ";", ":", "?", "+", "-", "*", "/", "%", "<", ">", "=", "!", "~", "&", "|", "^",
    ];

    public static (List<Token> Tokens, List<(string Text, int Line)> Directives) Tokenize(string source)
    {
        var tokens = new List<Token>();
        var directives = new List<(string, int)>();
        int i = 0, line = 1;
        bool lineStart = true;
        while (i < source.Length)
        {
            char c = source[i];
            if (c == '\n') { line++; i++; lineStart = true; continue; }
            if (char.IsWhiteSpace(c)) { i++; continue; }
            if (c == '/' && i + 1 < source.Length && source[i + 1] == '/') { while (i < source.Length && source[i] != '\n') i++; continue; }
            if (c == '/' && i + 1 < source.Length && source[i + 1] == '*')
            {
                int end = source.IndexOf("*/", i + 2, StringComparison.Ordinal);
                if (end < 0) throw new GlslException("unterminated comment", line);
                line += source.AsSpan(i, end - i).Count('\n');
                i = end + 2;
                continue;
            }
            if (c == '#' && lineStart)
            {
                var sb = new StringBuilder();
                int startLine = line;
                while (i < source.Length && source[i] != '\n')
                {
                    if (source[i] == '\\' && i + 1 < source.Length && source[i + 1] == '\n') { i += 2; line++; continue; }
                    sb.Append(source[i++]);
                }
                var text = sb.ToString();
                int cut = text.IndexOf("//", StringComparison.Ordinal);
                if (cut >= 0) text = text[..cut];
                directives.Add((text.Trim(), startLine));
                continue;
            }
            lineStart = false;
            if (char.IsLetter(c) || c == '_')
            {
                int s = i;
                while (i < source.Length && (char.IsLetterOrDigit(source[i]) || source[i] == '_')) i++;
                tokens.Add(new Token(TokenKind.Ident, source[s..i], line));
                continue;
            }
            if (char.IsDigit(c) || (c == '.' && i + 1 < source.Length && char.IsDigit(source[i + 1])))
            {
                tokens.Add(Number(source, ref i, line));
                continue;
            }
            var p = Puncts.FirstOrDefault(p => string.CompareOrdinal(source, i, p, 0, p.Length) == 0)
                ?? throw new GlslException($"'{c}' : unexpected character", line);
            tokens.Add(new Token(TokenKind.Punct, p, line));
            i += p.Length;
        }
        tokens.Add(new Token(TokenKind.End, "", line));
        return (tokens, directives);
    }

    static Token Number(string s, ref int i, int line)
    {
        int start = i;
        if (s[i] == '0' && i + 1 < s.Length && (s[i + 1] == 'x' || s[i + 1] == 'X'))
        {
            i += 2;
            while (i < s.Length && Uri.IsHexDigit(s[i])) i++;
            var hex = s[(start + 2)..i];
            bool u = i < s.Length && (s[i] == 'u' || s[i] == 'U');
            if (u) i++;
            ulong v = ulong.Parse(hex, NumberStyles.HexNumber, CultureInfo.InvariantCulture);
            if (v > uint.MaxValue) throw new GlslException($"'{s[start..i]}' : integer constant overflow", line);
            return new Token(u ? TokenKind.UIntLit : TokenKind.IntLit, ((uint)v).ToString(CultureInfo.InvariantCulture), line);
        }
        bool isFloat = false;
        while (i < s.Length && char.IsDigit(s[i])) i++;
        if (i < s.Length && s[i] == '.') { isFloat = true; i++; while (i < s.Length && char.IsDigit(s[i])) i++; }
        if (i < s.Length && (s[i] == 'e' || s[i] == 'E'))
        {
            int save = i;
            i++;
            if (i < s.Length && (s[i] == '+' || s[i] == '-')) i++;
            if (i < s.Length && char.IsDigit(s[i])) { isFloat = true; while (i < s.Length && char.IsDigit(s[i])) i++; }
            else i = save;
        }
        var text = s[start..i];
        if (isFloat)
        {
            if (i < s.Length && (s[i] == 'f' || s[i] == 'F')) i++; // GLSL ES 3.00 has no suffix, but drivers accept it; so does ANGLE
            return new Token(TokenKind.FloatLit, text, line);
        }
        bool unsigned = i < s.Length && (s[i] == 'u' || s[i] == 'U');
        if (unsigned) i++;
        // A leading 0 is octal in GLSL.
        ulong value = text.Length > 1 && text[0] == '0' ? Convert.ToUInt64(text, 8) : ulong.Parse(text, CultureInfo.InvariantCulture);
        if (value > uint.MaxValue) throw new GlslException($"'{text}' : integer constant overflow", line);
        return new Token(unsigned ? TokenKind.UIntLit : TokenKind.IntLit, ((uint)value).ToString(CultureInfo.InvariantCulture), line);
    }
}
