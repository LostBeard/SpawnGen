using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// A Python <c>re</c> pattern run on .NET's regex engine. The pattern is translated so it means what it means in Python:
/// <c>(?P&lt;name&gt;...)</c> / <c>(?P=name)</c> groups (numbered left to right as in Python; .NET would number named groups last),
/// <c>\Z</c> (the very end), <c>{,n}</c>, possessive quantifiers, octal and <c>\U</c> escapes, and Python's <c>\w</c> / <c>\s</c> classes.
/// Replacement strings follow <c>re.sub</c>: <c>\1</c>, <c>\g&lt;name&gt;</c>, <c>\g&lt;0&gt;</c>, <c>\n</c>; a <c>$</c> is just a dollar sign.
/// Errors Python reports as <c>re.error</c> throw <see cref="PyException"/> with type "error".
/// </summary>
public sealed class PyRegex
{
    static readonly TimeSpan Timeout = TimeSpan.FromSeconds(2);

    public Regex Regex { get; }
    /// <summary>Python's <c>pattern.groups</c>: the number of capturing groups.</summary>
    public int Groups { get; }
    public IReadOnlyDictionary<string, int> GroupIndex { get; }

    /// <summary>The same pattern refusing an empty match where the search starts (Python's rule after an empty match: try a longer one there first).</summary>
    readonly Regex _nonEmptyAtStart;

    PyRegex(Regex regex, Regex nonEmptyAtStart, int groups, Dictionary<string, int> names)
    {
        Regex = regex;
        _nonEmptyAtStart = nonEmptyAtStart;
        Groups = groups;
        GroupIndex = names;
    }

    public static PyRegex Compile(string pattern, bool ignoreCase = false, bool multiline = false, bool dotAll = false)
    {
        var t = new Translator(pattern, ignoreCase, multiline, dotAll);
        var dotnet = t.Run();
        var options = RegexOptions.CultureInvariant
            | (t.IgnoreCase ? RegexOptions.IgnoreCase : 0) | (t.Multiline ? RegexOptions.Multiline : 0) | (t.DotAll ? RegexOptions.Singleline : 0);
        try
        {
            return new PyRegex(new Regex(dotnet, options, Timeout), new Regex("(?:" + dotnet + ")(?!\\G)", options, Timeout), t.GroupCount, t.Names);
        }
        catch (ArgumentException e)
        {
            throw new PyException("error", e.Message);
        }
    }

    /// <summary><c>re.search</c>: the first match, or null.</summary>
    public Match? Search(string s)
    {
        var m = Regex.Match(s);
        return m.Success ? m : null;
    }

    /// <summary>
    /// <c>re.finditer</c> (Python 3.7+). After an empty match Python looks for a non-empty match at the same place before moving on
    /// (<c>re.sub('a*?', '-', 'aaa')</c> is "-------"); .NET would step over that character, so the search is driven here.
    /// </summary>
    public IEnumerable<Match> FindIter(string s)
    {
        int pos = 0;
        bool mustAdvance = false;
        while (pos <= s.Length)
        {
            var m = (mustAdvance ? _nonEmptyAtStart : Regex).Match(s, pos);
            if (!m.Success) yield break;
            yield return m;
            mustAdvance = m.Length == 0;
            pos = m.Index + m.Length;
        }
    }

    /// <summary><c>match.group(i)</c>: null when the group did not take part in the match.</summary>
    public static string? Group(Match m, int i) => m.Groups[i].Success ? m.Groups[i].Value : null;

    /// <summary><c>re.sub(pattern, repl, s, count)</c>; count 0 replaces every match.</summary>
    public string Sub(string repl, string s, int count = 0)
    {
        var parts = ParseTemplate(repl);
        string Expand(Match m)
        {
            var sb = new StringBuilder();
            foreach (var p in parts)
            {
                if (p.Literal is { } lit) sb.Append(lit);
                else if (m.Groups[p.Group].Success) sb.Append(m.Groups[p.Group].Value); // a group that did not match is empty (Python 3.5+)
            }
            return sb.ToString();
        }
        var result = new StringBuilder();
        int last = 0, done = 0;
        foreach (var m in FindIter(s))
        {
            if (count > 0 && done == count) break;
            result.Append(s, last, m.Index - last).Append(Expand(m));
            last = m.Index + m.Length;
            done++;
        }
        return result.Append(s, last, s.Length - last).ToString();
    }

    readonly record struct TemplatePart(string? Literal, int Group);

    /// <summary>Python's <c>re._parser.parse_template</c>.</summary>
    List<TemplatePart> ParseTemplate(string repl)
    {
        var parts = new List<TemplatePart>();
        var lit = new StringBuilder();
        void Flush() { if (lit.Length > 0) { parts.Add(new(lit.ToString(), 0)); lit.Clear(); } }
        void AddGroup(int index)
        {
            if (index > Groups) throw new PyException("error", $"invalid group reference {index}");
            Flush();
            parts.Add(new(null, index));
        }
        int i = 0, n = repl.Length;
        while (i < n)
        {
            char c = repl[i++];
            if (c != '\\') { lit.Append(c); continue; }
            if (i >= n) throw new PyException("error", "bad escape (end of pattern)");
            char e = repl[i++];
            if (e == 'g')
            {
                if (i >= n || repl[i] != '<') throw new PyException("error", "missing <");
                int close = repl.IndexOf('>', i + 1);
                if (close < 0) throw new PyException("error", "missing >, unterminated name");
                var name = repl[(i + 1)..close];
                i = close + 1;
                if (name.Length == 0) throw new PyException("error", "missing group name");
                if (IsIdentifier(name))
                {
                    if (!GroupIndex.TryGetValue(name, out var idx)) throw new PyException("IndexError", $"unknown group name '{name}'");
                    AddGroup(idx);
                }
                else if (name.All(char.IsAsciiDigit) && int.TryParse(name, NumberStyles.None, CultureInfo.InvariantCulture, out var num)) AddGroup(num);
                else throw new PyException("error", $"bad character in group name '{name}'");
            }
            else if (e == '0')
            {
                int v = 0, k = 0;
                while (k < 2 && i < n && repl[i] is >= '0' and <= '7') { v = v * 8 + (repl[i++] - '0'); k++; }
                lit.Append((char)(v & 0xFF));
            }
            else if (char.IsAsciiDigit(e))
            {
                if (i < n && char.IsAsciiDigit(repl[i]))
                {
                    char d2 = repl[i];
                    if (e <= '7' && d2 <= '7' && i + 1 < n && repl[i + 1] is >= '0' and <= '7')
                    {
                        int v = (e - '0') * 64 + (d2 - '0') * 8 + (repl[i + 1] - '0');
                        if (v > 0xFF) throw new PyException("error", $"octal escape value \\{e}{d2}{repl[i + 1]} outside of range 0-0o377");
                        lit.Append((char)v);
                        i += 2;
                        continue;
                    }
                    AddGroup((e - '0') * 10 + (d2 - '0'));
                    i++;
                }
                else AddGroup(e - '0');
            }
            else
            {
                switch (e)
                {
                    case 'a': lit.Append('\a'); break;
                    case 'b': lit.Append('\b'); break;
                    case 'f': lit.Append('\f'); break;
                    case 'n': lit.Append('\n'); break;
                    case 'r': lit.Append('\r'); break;
                    case 't': lit.Append('\t'); break;
                    case 'v': lit.Append('\v'); break;
                    case '\\': lit.Append('\\'); break;
                    default:
                        if (char.IsAsciiLetter(e)) throw new PyException("error", $"bad escape \\{e}");
                        lit.Append('\\').Append(e); // other escapes are kept as they are: "\&" stays "\&"
                        break;
                }
            }
        }
        Flush();
        return parts;
    }

    static bool IsIdentifier(string s) =>
        s.Length > 0 && (char.IsLetter(s[0]) || s[0] == '_') && s.All(c => char.IsLetterOrDigit(c) || c == '_');

    // ---- pattern translation ---------------------------------------------------------------------------------------

    sealed class Translator(string p, bool ignoreCase, bool multiline, bool dotAll)
    {
        public bool IgnoreCase = ignoreCase, Multiline = multiline, DotAll = dotAll;
        bool _verbose, _ascii;
        public int GroupCount;
        public readonly Dictionary<string, int> Names = new(StringComparer.Ordinal);
        readonly HashSet<int> _open = [];
        readonly StringBuilder _o = new();
        int _i;
        int _lastAtom = -1;

        static PyException Error(string message) => new("error", message);

        public string Run()
        {
            ReadGlobalFlags();
            var groupStarts = new Stack<(int OutStart, int Group)>();
            int n = p.Length;
            while (_i < n)
            {
                char c = p[_i];
                if (_verbose && (PyNumber.IsSpace(c) || c == '#'))
                {
                    if (c == '#') while (_i < n && p[_i] != '\n') _i++;
                    else _i++;
                    continue;
                }
                switch (c)
                {
                    case '\\': { int start = _o.Length; Escape(); _lastAtom = start; break; }
                    case '[': { int start = _o.Length; CharClass(); _lastAtom = start; break; }
                    case '(':
                    {
                        int start = _o.Length;
                        int group = OpenGroup();
                        if (group == Backreference) _lastAtom = start; // (?P=name): an atom, nothing to close
                        else if (group == Comment) { } // (?#...): dropped, a quantifier after it repeats what came before
                        else { groupStarts.Push((start, group)); _lastAtom = -1; }
                        break;
                    }
                    case ')':
                    {
                        if (groupStarts.Count == 0) throw Error($"unbalanced parenthesis at position {_i}");
                        var (start, group) = groupStarts.Pop();
                        if (group > 0) _open.Remove(group);
                        _o.Append(')');
                        _i++;
                        _lastAtom = start;
                        break;
                    }
                    case '*' or '+' or '?':
                        if (_lastAtom < 0) throw Error($"nothing to repeat at position {_i}");
                        _o.Append(c);
                        _i++;
                        QuantifierSuffix();
                        break;
                    case '{':
                        if (TryRepeat(out var lo, out var hi))
                        {
                            if (_lastAtom < 0) throw Error($"nothing to repeat at position {_i}");
                            _o.Append('{').Append(lo ?? "0");
                            if (hi != lo || hi is null) _o.Append(',').Append(hi ?? "");
                            _o.Append('}');
                            QuantifierSuffix();
                        }
                        else { int start = _o.Length; _o.Append("\\{"); _i++; _lastAtom = start; }
                        break;
                    case '|':
                        _o.Append('|'); _i++; _lastAtom = -1; break;
                    case '^' or '$':
                        _o.Append(c); _i++; _lastAtom = -1; break;
                    case '.':
                        { int start = _o.Length; _o.Append('.'); _i++; _lastAtom = start; break; }
                    default:
                        { int start = _o.Length; AppendLiteral(c); _i++; _lastAtom = start; break; }
                }
            }
            if (groupStarts.Count > 0) throw Error("missing ), unterminated subpattern");
            return _o.ToString();
        }

        /// <summary>Flags in a leading <c>(?aimsux)</c> group apply to the whole pattern (Python 3.11 refuses them anywhere else).</summary>
        void ReadGlobalFlags()
        {
            while (_i + 2 < p.Length && p[_i] == '(' && p[_i + 1] == '?' && "aiLmsux".Contains(p[_i + 2]))
            {
                int j = _i + 2;
                while (j < p.Length && "aiLmsux".Contains(p[j])) j++;
                if (j >= p.Length || p[j] != ')') return; // a scoped (?i:...) group, handled where it is
                foreach (var f in p[(_i + 2)..j]) SetFlag(f);
                _i = j + 1;
            }
        }

        void SetFlag(char f)
        {
            switch (f)
            {
                case 'i': IgnoreCase = true; break;
                case 'm': Multiline = true; break;
                case 's': DotAll = true; break;
                case 'x': _verbose = true; break;
                case 'a': _ascii = true; break;
                case 'u': break;
                case 'L': throw Error("bad inline flags: cannot use 'L' flag with a str pattern");
            }
        }

        void QuantifierSuffix()
        {
            if (_i < p.Length && p[_i] == '?') { _o.Append('?'); _i++; }
            else if (_i < p.Length && p[_i] == '+')
            {
                // Possessive (Python 3.11+): .NET has none, an atomic group around the repeat is the same thing.
                _o.Insert(_lastAtom, "(?>");
                _o.Append(')');
                _i++;
            }
            if (_i < p.Length && p[_i] is '*' or '+' or '?' || _i < p.Length && p[_i] == '{' && PeekRepeat())
                throw Error($"multiple repeat at position {_i}");
            _lastAtom = -1;
        }

        bool PeekRepeat()
        {
            int save = _i;
            bool ok = TryRepeat(out _, out _);
            _i = save;
            return ok;
        }

        /// <summary>Python's <c>{m}</c>, <c>{m,}</c>, <c>{,n}</c>, <c>{m,n}</c>; anything else is a literal brace.</summary>
        bool TryRepeat(out string? lo, out string? hi)
        {
            lo = hi = null;
            int j = _i + 1, n = p.Length;
            int s = j;
            while (j < n && char.IsAsciiDigit(p[j])) j++;
            var a = p[s..j];
            string? b = a;
            if (j < n && p[j] == ',')
            {
                j++;
                int s2 = j;
                while (j < n && char.IsAsciiDigit(p[j])) j++;
                b = p[s2..j];
                if (b.Length == 0) b = null;
            }
            else if (a.Length == 0) return false;
            if (j >= n || p[j] != '}') return false;
            lo = a.Length == 0 ? null : a;
            hi = b;
            if (lo is not null && hi is not null && System.Numerics.BigInteger.Parse(lo) > System.Numerics.BigInteger.Parse(hi))
                throw Error($"min repeat greater than max repeat at position {_i + 1}");
            _i = j + 1;
            return true;
        }

        int OpenGroup()
        {
            int n = p.Length;
            if (_i + 1 < n && p[_i + 1] == '?')
            {
                int j = _i + 2;
                if (j >= n) throw Error("unexpected end of pattern");
                char k = p[j];
                if (k == 'P')
                {
                    if (j + 1 < n && p[j + 1] == '<')
                    {
                        int close = p.IndexOf('>', j + 2);
                        if (close < 0) throw Error("missing >, unterminated name");
                        var name = p[(j + 2)..close];
                        if (!IsIdentifier(name)) throw Error($"bad character in group name '{name}'");
                        if (Names.ContainsKey(name)) throw Error($"redefinition of group name '{name}'");
                        int g = ++GroupCount;
                        Names[name] = g;
                        _open.Add(g);
                        _o.Append('(');
                        _i = close + 1;
                        return g;
                    }
                    if (j + 1 < n && p[j + 1] == '=')
                    {
                        int close = p.IndexOf(')', j + 2);
                        if (close < 0) throw Error("missing ), unterminated name");
                        var name = p[(j + 2)..close];
                        if (!Names.TryGetValue(name, out var g)) throw Error($"unknown group name '{name}'");
                        if (_open.Contains(g)) throw Error("cannot refer to an open group");
                        // a backreference, not a group: emit it in place of the "(?P=name)" text
                        _o.Append("\\k<").Append(g).Append('>');
                        _i = close + 1;
                        return Backreference;
                    }
                    throw Error($"unknown extension ?P{(j + 1 < n ? p[j + 1] : ' ')}");
                }
                if (k == '#')
                {
                    int close = p.IndexOf(')', j);
                    if (close < 0) throw Error("missing ), unterminated comment");
                    _i = close + 1;
                    return Comment;
                }
                if (k is ':' or '=' or '!' or '>') { _o.Append("(?").Append(k); _i = j + 1; return 0; }
                if (k == '<')
                {
                    if (j + 1 < n && p[j + 1] is '=' or '!') { _o.Append("(?<").Append(p[j + 1]); _i = j + 2; return 0; }
                    throw Error($"unknown extension ?<{(j + 1 < n ? p[j + 1] : ' ')}");
                }
                if (k == '(')
                {
                    int close = p.IndexOf(')', j + 1);
                    if (close < 0) throw Error("missing ), unterminated name");
                    var cond = p[(j + 1)..close];
                    int g;
                    if (cond.All(char.IsAsciiDigit) && cond.Length > 0) g = int.Parse(cond, CultureInfo.InvariantCulture);
                    else if (!Names.TryGetValue(cond, out g)) throw Error($"unknown group name '{cond}'");
                    if (g > GroupCount) throw Error("invalid group reference " + g);
                    _o.Append("(?(").Append(g).Append(')');
                    _i = close + 1;
                    return 0;
                }
                if ("aiLmsux-".Contains(k))
                {
                    // Scoped flags (?flags-flags:...): .NET understands i, m, s, x.
                    int colon = j;
                    while (colon < n && "aiLmsux-".Contains(p[colon])) colon++;
                    if (colon >= n) throw Error("missing -, : or )");
                    if (p[colon] == ')') throw Error($"global flags not at the start of the expression at position {_i}");
                    if (p[colon] != ':') throw Error("unknown flag");
                    var flags = p[j..colon];
                    if (flags.Contains('a') || flags.Contains('L')) throw new NotSupportedException($"The scoped regex flag group (?{flags}:...) is not supported.");
                    flags = flags.Replace("u", "");
                    _o.Append("(?").Append(flags).Append(':');
                    _i = colon + 1;
                    return 0;
                }
                throw Error($"unknown extension ?{k}");
            }
            int group = ++GroupCount;
            _open.Add(group);
            _o.Append('(');
            _i++;
            return group;
        }

        // What OpenGroup returns for the "(?" items that open nothing: a backreference and a comment.
        const int Backreference = -1, Comment = -2;

        void Escape()
        {
            int n = p.Length;
            if (_i + 1 >= n) throw Error("bad escape (end of pattern)");
            char e = p[_i + 1];
            _i += 2;
            switch (e)
            {
                case 'd': _o.Append(_ascii ? "[0-9]" : "\\d"); return;
                case 'D': _o.Append(_ascii ? "[^0-9]" : "\\D"); return;
                case 'w': _o.Append(_ascii ? "[a-zA-Z0-9_]" : "[\\p{L}\\p{N}_]"); return;
                case 'W': _o.Append(_ascii ? "[^a-zA-Z0-9_]" : "[^\\p{L}\\p{N}_]"); return;
                case 's': _o.Append(_ascii ? "[ \\t\\n\\r\\f\\v]" : "[\\s\\x1c-\\x1f]"); return;
                case 'S': _o.Append(_ascii ? "[^ \\t\\n\\r\\f\\v]" : "[^\\s\\x1c-\\x1f]"); return;
                case 'b': _o.Append(_ascii ? "(?:(?<=[a-zA-Z0-9_])(?![a-zA-Z0-9_])|(?<![a-zA-Z0-9_])(?=[a-zA-Z0-9_]))" : "\\b"); return;
                case 'B': _o.Append(_ascii ? "(?:(?<=[a-zA-Z0-9_])(?=[a-zA-Z0-9_])|(?<![a-zA-Z0-9_])(?![a-zA-Z0-9_]))" : "\\B"); return;
                case 'A': _o.Append("\\A"); return;
                case 'Z': _o.Append("\\z"); return; // Python's \Z is the very end (.NET's \Z also matches before a final newline)
            }
            if (e == '0' || (e is >= '1' and <= '9'))
            {
                if (e != '0' && !(e <= '7' && _i + 1 < n && p[_i] is >= '0' and <= '7' && p[_i + 1] is >= '0' and <= '7'))
                {
                    // a group reference of one or two digits
                    int g = e - '0';
                    if (_i < n && char.IsAsciiDigit(p[_i])) { g = g * 10 + (p[_i] - '0'); _i++; }
                    if (g > GroupCount) throw Error($"invalid group reference {g}");
                    if (_open.Contains(g)) throw Error("cannot refer to an open group");
                    _o.Append("\\k<").Append(g).Append('>');
                    return;
                }
                AppendCodePoint(Octal(e));
                return;
            }
            if (TryLiteralEscape(e, out var cp)) { AppendCodePoint(cp); return; }
            if (char.IsAsciiLetter(e)) throw Error($"bad escape \\{e}");
            AppendLiteral(e);
        }

        /// <summary>\0 plus up to two more octal digits, or three octal digits (the caller checked they are there).</summary>
        int Octal(char first)
        {
            int v = first - '0';
            for (int c = 0; c < 2 && _i < p.Length && p[_i] is >= '0' and <= '7'; c++) v = v * 8 + (p[_i++] - '0');
            if (v > 0xFF) throw Error("octal escape value outside of range 0-0o377");
            return v;
        }

        /// <summary>\a \f \n \r \t \v \\ \xhh \uhhhh \Uhhhhhhhh (and \N{...}, which needs the Unicode name list .NET does not have).</summary>
        bool TryLiteralEscape(char e, out int cp)
        {
            cp = 0;
            switch (e)
            {
                case 'a': cp = 7; return true;
                case 'f': cp = 12; return true;
                case 'n': cp = 10; return true;
                case 'r': cp = 13; return true;
                case 't': cp = 9; return true;
                case 'v': cp = 11; return true;
                case '\\': cp = '\\'; return true;
                case 'x': cp = Hex(2, "\\x"); return true;
                case 'u': cp = Hex(4, "\\u"); return true;
                case 'U':
                    cp = Hex(8, "\\U");
                    if (cp > 0x10FFFF) throw Error("bad escape \\U");
                    return true;
                case 'N': throw new NotSupportedException("\\N{...} (characters by Unicode name) is not supported in regex patterns here; use \\uXXXX.");
            }
            return false;
        }

        int Hex(int count, string what)
        {
            int j = _i;
            while (j < p.Length && j - _i < count && Uri.IsHexDigit(p[j])) j++;
            if (j - _i != count) throw Error($"incomplete escape {what}{p[_i..j]}");
            var v = int.Parse(p.AsSpan(_i, count), NumberStyles.AllowHexSpecifier, CultureInfo.InvariantCulture);
            _i = j;
            return v;
        }

        void AppendCodePoint(int cp)
        {
            if (cp > 0xFFFF)
            {
                var s = char.ConvertFromUtf32(cp);
                _o.Append("(?:\\u").Append(((int)s[0]).ToString("X4")).Append("\\u").Append(((int)s[1]).ToString("X4")).Append(')');
            }
            else _o.Append("\\u").Append(cp.ToString("X4"));
        }

        void AppendLiteral(char c)
        {
            if (char.IsAsciiLetterOrDigit(c) || c == '_' || c == ' ' || c > 127 && !char.IsSurrogate(c)) _o.Append(c);
            else if (char.IsHighSurrogate(c) && _i + 1 < p.Length && char.IsLowSurrogate(p[_i + 1]))
            {
                // One Python character: keep the pair together so a quantifier repeats both halves.
                _o.Append("(?:").Append(c).Append(p[_i + 1]).Append(')');
                _i++;
            }
            else _o.Append("\\u").Append(((int)c).ToString("X4"));
        }

        // One item of a character class: a character (code point), a dash, or ready .NET class text (\d, \p{L}, ...).
        readonly record struct ClassItem(int Char, string? Raw, bool Dash);

        /// <summary>
        /// A [...] set. Characters outside the BMP (emoji) are two UTF-16 units in .NET, so they cannot sit in a .NET class: they become alternatives
        /// next to it, <c>(?:[ab]|(surrogate pair))</c>. Ranges or negated sets with such characters are not supported.
        /// </summary>
        void CharClass()
        {
            int n = p.Length;
            _i++;
            bool negated = false;
            if (_i < n && p[_i] == '^') { negated = true; _i++; }
            var items = new List<ClassItem>();
            bool first = true;
            while (true)
            {
                if (_i >= n) throw Error("unterminated character set");
                char c = p[_i];
                if (c == ']' && !first) { _i++; break; }
                first = false;
                if (c == '\\')
                {
                    if (_i + 1 >= n) throw Error("bad escape (end of pattern)");
                    char e = p[_i + 1];
                    _i += 2;
                    string? raw = e switch
                    {
                        'd' => _ascii ? "0-9" : "\\d",
                        'D' => _ascii ? "\\u0000-\\u002F\\u003A-\\uFFFF" : "\\D",
                        'w' => _ascii ? "a-zA-Z0-9_" : "\\p{L}\\p{N}_",
                        'W' => _ascii ? "\\u0000-\\u002F\\u003A-\\u0040\\u005B-\\u005E\\u0060\\u007B-\\uFFFF" : "\\W",
                        's' => _ascii ? " \\t\\n\\r\\f\\v" : "\\s\\x1c-\\x1f",
                        'S' => _ascii ? "\\u0000-\\u0008\\u000E-\\u001F\\u0021-\\uFFFF" : "\\S",
                        _ => null,
                    };
                    if (raw is not null) { items.Add(new(0, raw, false)); continue; }
                    if (e == 'b') { items.Add(new(8, null, false)); continue; }
                    if (e is >= '0' and <= '7')
                    {
                        int v = e - '0';
                        for (int k = 0; k < 2 && _i < n && p[_i] is >= '0' and <= '7'; k++) v = v * 8 + (p[_i++] - '0');
                        if (v > 0xFF) throw Error("octal escape value outside of range 0-0o377");
                        items.Add(new(v, null, false));
                        continue;
                    }
                    if (e is '8' or '9') throw Error($"bad escape \\{e}");
                    if (TryLiteralEscape(e, out var cp)) { items.Add(new(cp, null, false)); continue; }
                    if (char.IsAsciiLetter(e)) throw Error($"bad escape \\{e}");
                    items.Add(new(e, null, false));
                    continue;
                }
                if (c == '-') { items.Add(new(0, null, true)); _i++; continue; }
                if (char.IsHighSurrogate(c) && _i + 1 < n && char.IsLowSurrogate(p[_i + 1]))
                {
                    items.Add(new(char.ConvertToUtf32(c, p[_i + 1]), null, false));
                    _i += 2;
                    continue;
                }
                items.Add(new(c, null, false));
                _i++;
            }

            static bool IsChar(ClassItem it) => !it.Dash && it.Raw is null;
            var body = new StringBuilder();
            var astral = new List<int>();
            for (int k = 0; k < items.Count; k++)
            {
                var it = items[k];
                bool inRange = IsChar(it) && ((k + 2 < items.Count && items[k + 1].Dash && IsChar(items[k + 2]))
                    || (k >= 2 && items[k - 1].Dash && IsChar(items[k - 2])));
                if (it.Dash) body.Append('-');
                else if (it.Raw is { } raw) body.Append(raw);
                else if (it.Char > 0xFFFF)
                {
                    if (inRange || negated) throw new NotSupportedException("Ranges and negated sets with characters outside the Basic Multilingual Plane (emoji) are not supported in regex patterns here.");
                    astral.Add(it.Char);
                }
                else AppendClassChar(body, it.Char);
            }
            if (astral.Count == 0)
            {
                _o.Append('[').Append(negated ? "^" : "").Append(body).Append(']');
                return;
            }
            _o.Append("(?:");
            if (body.Length > 0) _o.Append('[').Append(body).Append("]|");
            _o.Append(string.Join("|", astral.Select(cp =>
            {
                var pair = char.ConvertFromUtf32(cp);
                return "\\u" + ((int)pair[0]).ToString("X4") + "\\u" + ((int)pair[1]).ToString("X4");
            })));
            _o.Append(')');
        }

        static void AppendClassChar(StringBuilder sb, int cp)
        {
            char c = (char)cp;
            if (char.IsAsciiLetterOrDigit(c) || c == ' ' || c > 127 && !char.IsSurrogate(c)) sb.Append(c);
            else sb.Append("\\u").Append(cp.ToString("X4"));
        }
    }
}
