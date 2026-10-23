using System.Globalization;
using System.Numerics;
using System.Text;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// ComfyUI's Math Expression language: <c>simpleeval.simple_eval(expression, names=..., functions=MATH_FUNCTIONS)</c> (simpleeval 1.0) ported to C#.
/// A Python expression is parsed and walked; nothing is compiled or run as code. What simpleeval allows is allowed: numbers, strings, the names it is given,
/// arithmetic with Python's rules (ints never overflow, <c>//</c> and <c>%</c> floor, <c>/</c> is true division), comparisons (chained), <c>and</c> / <c>or</c> /
/// <c>not</c>, <c>x if c else y</c>, subscripts and slices, attributes, f-strings, and the functions sum, min, max, abs, round, pow, sqrt, ceil, floor, log, log2,
/// log10, sin, cos, tan, int, float. Tuples, lists, dicts, lambdas and comprehensions parse but are refused when evaluated, like simpleeval's base evaluator.
/// Python errors throw <see cref="PyException"/> with Python's exception name.
/// </summary>
public static class PyExpression
{
    const int MaxStringLength = 100000;
    const int MaxPower = 4000000;
    const int MaxShift = 10000;
    const int MathPowMaxExponent = 4000; // nodes_math._safe_pow
    static readonly BigInteger MaxShiftBase = PyNumber.ToBigInteger(double.MaxValue);
    static readonly string[] DisallowPrefixes = ["_", "func_"];
    static readonly string[] DisallowMethods = ["format", "format_map", "mro", "tb_frame", "gi_frame", "ag_frame", "cr_frame", "exec"];

    /// <summary>
    /// ComfyMathExpression.execute: evaluates <paramref name="expression"/> with each value under its name and all of them as the list <c>values</c>,
    /// and returns <c>(float(result), int(result), bool(result))</c>. The result must be a finite number (or a bool).
    /// </summary>
    public static (double Float, long Int, bool Bool) EvaluateMath(string expression, IReadOnlyList<KeyValuePair<string, object?>> values)
    {
        if (PyText.Strip(expression).Length == 0) throw new PyException("ValueError", "Expression cannot be empty.");
        var names = new Dictionary<string, object?>(StringComparer.Ordinal);
        foreach (var (k, v) in values) names[k] = Normalize(v);
        names["values"] = values.Select(kv => Normalize(kv.Value)).ToList();
        var result = Evaluate(expression, names);
        if (!(result is bool || result is BigInteger || result is double))
            throw new PyException("ValueError", $"Math Expression '{expression}' must evaluate to a numeric result, got {PyFormat.TypeName(result)}: {PyText.Repr(result)}");
        double f;
        try { f = AsFloat(result); }
        catch (PyException e) when (e.PyType == "OverflowError")
        {
            throw new PyException("ValueError", $"Math Expression '{expression}' produced a result too large to represent as a float: {PyText.Repr(result)}");
        }
        if (!double.IsFinite(f)) throw new PyException("ValueError", $"Math Expression '{expression}' produced a non-finite result: {PyText.Repr(result)}");
        var i = result is double d ? PyNumber.ToBigInteger(d) : AsInt(result!);
        if (i < long.MinValue || i > long.MaxValue) throw new OverflowException($"Math Expression '{expression}': {i} does not fit in an INT (64 bits).");
        return (f, (long)i, Truthy(result));
    }

    /// <summary>Evaluates a simpleeval expression with the given names and ComfyUI's math functions. Values are Python values (BigInteger ints, double floats, ...).</summary>
    public static object? Evaluate(string expression, IReadOnlyDictionary<string, object?> names)
    {
        var parser = new Parser(PyText.Strip(expression));
        var statement = parser.ParseProgram();
        return new Evaluator(names, expression).Eval(statement);
    }

    /// <summary>Node values as the evaluator's Python values: whole numbers become BigInteger, floats double.</summary>
    public static object? Normalize(object? v) => v switch
    {
        long l => new BigInteger(l),
        int i => new BigInteger(i),
        float f => (double)f,
        System.Collections.IList list when v is not string => list.Cast<object?>().Select(Normalize).ToList(),
        _ => v,
    };

    // ==== AST ======================================================================================================

    abstract record Node;
    sealed record Const(object? Value) : Node;
    sealed record Name(string Id) : Node;
    sealed record Unary(string Op, Node Operand) : Node;
    sealed record Binary(string Op, Node Left, Node Right) : Node;
    sealed record BoolOp(bool And, List<Node> Values) : Node;
    sealed record Compare(Node Left, List<string> Ops, List<Node> Comparators) : Node;
    sealed record IfExp(Node Test, Node Body, Node OrElse) : Node;
    sealed record Call(Node Func, List<Node> Args, List<(string? Name, Node Value)> Keywords) : Node;
    sealed record Subscript(Node Value, Node Index) : Node;
    sealed record SliceNode(Node? Lower, Node? Upper, Node? Step) : Node;
    sealed record Attribute(Node Value, string Attr) : Node;
    sealed record JoinedStr(List<Node> Values) : Node;
    sealed record FormattedValue(Node Value, JoinedStr? Spec) : Node;
    /// <summary>Syntax simpleeval's evaluator has no handler for (Tuple, List, Lambda, ...): refused when evaluated.</summary>
    sealed record Unsupported(string Kind) : Node;
    sealed record Ellipsis : Node;

    // ==== tokenizer ================================================================================================

    enum Tok { Name, Number, String, Op, Newline, End }

    sealed record Token(Tok Kind, string Text, int Pos)
    {
        public List<StringPiece>? Pieces { get; init; }
    }

    /// <summary>One string literal of a (possibly concatenated) string: its prefix letters and its raw body.</summary>
    sealed record StringPiece(string Prefix, string Body);

    static PyException Syntax(string message) => new("SyntaxError", message);

    static readonly string[] Ops3 = ["**=", "//=", ">>=", "<<=", "..."];
    static readonly string[] Ops2 = ["!=", "==", "<=", ">=", "**", "//", "<<", ">>", "->", ":=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "@="];
    const string Ops1 = "+-*/%@&|^~<>()[]{},:.;=";

    static List<Token> Tokenize(string s)
    {
        var tokens = new List<Token>();
        int i = 0, n = s.Length, depth = 0;
        bool lineStart = false;
        while (i < n)
        {
            char c = s[i];
            if (c == '\\' && i + 1 < n && (s[i + 1] == '\n' || s[i + 1] == '\r')) { i += s[i + 1] == '\r' && i + 2 < n && s[i + 2] == '\n' ? 3 : 2; continue; }
            if (c is '\n' or '\r')
            {
                i++;
                if (depth == 0)
                {
                    if (tokens.Count > 0 && tokens[^1].Kind != Tok.Newline) tokens.Add(new Token(Tok.Newline, "\n", i));
                    lineStart = true;
                }
                continue;
            }
            if (c is ' ' or '\t' or '\f')
            {
                int start = i;
                while (i < n && s[i] is ' ' or '\t' or '\f') i++;
                // An indented line after a newline (outside brackets) is Python's IndentationError, unless it is blank or a comment.
                if (lineStart && depth == 0 && i < n && s[i] is not ('\n' or '\r' or '#') && start < i) throw new PyException("IndentationError", "unexpected indent");
                continue;
            }
            lineStart = false;
            if (c == '#') { while (i < n && s[i] is not ('\n' or '\r')) i++; continue; }
            if (IsStringStart(s, i, out int prefixLen))
            {
                var pieces = new List<StringPiece>();
                int start = i;
                pieces.Add(ReadString(s, ref i, prefixLen));
                tokens.Add(new Token(Tok.String, s[start..i], start) { Pieces = pieces });
                continue;
            }
            if (char.IsAsciiDigit(c) || (c == '.' && i + 1 < n && char.IsAsciiDigit(s[i + 1])))
            {
                int start = i;
                ReadNumber(s, ref i);
                if (i < n && IsIdStart(s[i])) throw Syntax("invalid decimal literal");
                tokens.Add(new Token(Tok.Number, s[start..i], start));
                continue;
            }
            if (IsIdStart(c))
            {
                int start = i;
                while (i < n && IsIdContinue(s[i])) i++;
                tokens.Add(new Token(Tok.Name, s[start..i], start));
                continue;
            }
            string? op = null;
            foreach (var o in Ops3) if (string.CompareOrdinal(s, i, o, 0, 3) == 0) { op = o; break; }
            if (op is null) foreach (var o in Ops2) if (string.CompareOrdinal(s, i, o, 0, 2) == 0) { op = o; break; }
            if (op is null && Ops1.Contains(c)) op = c.ToString();
            if (op is null) throw Syntax($"invalid character '{c}' (U+{(int)c:X4})");
            if (op is "(" or "[" or "{") depth++;
            else if (op is ")" or "]" or "}") { if (depth == 0) throw Syntax($"unmatched '{op}'"); depth--; }
            if (op == ";") { if (tokens.Count > 0 && tokens[^1].Kind != Tok.Newline) tokens.Add(new Token(Tok.Newline, ";", i)); i++; continue; }
            tokens.Add(new Token(Tok.Op, op, i));
            i += op.Length;
        }
        if (depth > 0) throw Syntax("'(' was never closed");
        tokens.Add(new Token(Tok.End, "", n));
        return tokens;
    }

    static bool IsIdStart(char c) => c == '_' || char.IsLetter(c) || CharUnicodeInfo.GetUnicodeCategory(c) == UnicodeCategory.LetterNumber;

    static bool IsIdContinue(char c) => IsIdStart(c) || char.IsDigit(c) || CharUnicodeInfo.GetUnicodeCategory(c) is
        UnicodeCategory.NonSpacingMark or UnicodeCategory.SpacingCombiningMark or UnicodeCategory.ConnectorPunctuation or UnicodeCategory.DecimalDigitNumber;

    static bool IsStringStart(string s, int i, out int prefixLen)
    {
        prefixLen = 0;
        int j = i;
        while (j < s.Length && j - i < 2 && "rRbBuUfF".Contains(s[j])) j++;
        if (j < s.Length && s[j] is '\'' or '"')
        {
            var prefix = s[i..j].ToLowerInvariant();
            if (prefix is "" or "r" or "u" or "b" or "f" or "br" or "rb" or "fr" or "rf") { prefixLen = j - i; return true; }
        }
        return false;
    }

    static StringPiece ReadString(string s, ref int i, int prefixLen)
    {
        var prefix = s.Substring(i, prefixLen).ToLowerInvariant();
        i += prefixLen;
        char q = s[i];
        bool triple = i + 2 < s.Length && s[i + 1] == q && s[i + 2] == q;
        int open = triple ? 3 : 1;
        i += open;
        int start = i;
        while (true)
        {
            if (i >= s.Length) throw Syntax(triple ? "unterminated triple-quoted string literal" : "unterminated string literal");
            char c = s[i];
            if (c == '\\') { i += 2; continue; }
            if (!triple && (c == '\n' || c == '\r')) throw Syntax("unterminated string literal");
            if (c == q && (!triple || (i + 2 < s.Length && s[i + 1] == q && s[i + 2] == q)))
            {
                var body = s[start..i];
                i += open;
                return new StringPiece(prefix, body);
            }
            i++;
        }
    }

    static void ReadNumber(string s, ref int i)
    {
        int n = s.Length;
        if (s[i] == '0' && i + 1 < n && char.ToLowerInvariant(s[i + 1]) is 'x' or 'o' or 'b')
        {
            i += 2;
            while (i < n && (char.IsAsciiHexDigit(s[i]) || s[i] == '_')) i++;
            return;
        }
        while (i < n && (char.IsAsciiDigit(s[i]) || s[i] == '_')) i++;
        if (i < n && s[i] == '.') { i++; while (i < n && (char.IsAsciiDigit(s[i]) || s[i] == '_')) i++; }
        if (i < n && s[i] is 'e' or 'E')
        {
            int j = i + 1;
            if (j < n && s[j] is '+' or '-') j++;
            if (j < n && char.IsAsciiDigit(s[j])) { i = j; while (i < n && (char.IsAsciiDigit(s[i]) || s[i] == '_')) i++; }
        }
        if (i < n && s[i] is 'j' or 'J') i++;
    }

    static object NumberValue(string text)
    {
        var t = text;
        if (t.EndsWith('j') || t.EndsWith('J')) return new Unsupported("complex number");
        bool bad = t.Contains("__") || t.EndsWith('_') || t.Contains("_.") || t.Contains("._") || t.Contains("_e") || t.Contains("_E") || t.Contains("e_") || t.Contains("E_");
        if (t.Length > 1 && t[0] == '0' && char.ToLowerInvariant(t[1]) is 'x' or 'o' or 'b')
        {
            int b = char.ToLowerInvariant(t[1]) switch { 'x' => 16, 'o' => 8, _ => 2 };
            var v = PyNumber.TryParseInt(t, b);
            if (v is null || t.Length == 2 || t.EndsWith('_')) throw Syntax($"invalid {(b == 16 ? "hexadecimal" : b == 8 ? "octal" : "binary")} literal");
            return v.Value;
        }
        if (bad) throw Syntax("invalid decimal literal");
        if (t.Contains('.') || t.Contains('e') || t.Contains('E'))
            return PyNumber.TryParseFloat(t) ?? throw Syntax("invalid decimal literal");
        var digits = t.Replace("_", "");
        if (digits.Length > 1 && digits[0] == '0' && digits.Any(ch => ch != '0'))
            throw Syntax("leading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers");
        return BigInteger.Parse(digits, CultureInfo.InvariantCulture);
    }

    // ==== parser ===================================================================================================

    sealed class Parser
    {
        readonly List<Token> _t;
        int _p;

        public Parser(string source) { _t = Tokenize(source); }

        Token Peek => _t[_p];
        Token Next() => _t[_p++];
        bool IsOp(string op) => Peek.Kind == Tok.Op && Peek.Text == op;
        bool IsKw(string kw) => Peek.Kind == Tok.Name && Peek.Text == kw;
        void Expect(string op)
        {
            if (!IsOp(op)) throw Syntax(Peek.Kind == Tok.End ? "unexpected EOF while parsing" : $"invalid syntax (expected '{op}')");
            _p++;
        }

        static readonly HashSet<string> Keywords =
        [
            "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del", "elif", "else", "except",
            "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
            "while", "with", "yield",
        ];

        static readonly HashSet<string> StatementKeywords =
            ["assert", "break", "class", "continue", "def", "del", "for", "from", "global", "if", "import", "nonlocal", "pass", "raise", "return", "try", "while", "with", "async"];

        /// <summary>All statements are parsed (a syntax error anywhere fails, as in <c>ast.parse</c>); the first one is what is evaluated.</summary>
        public Node ParseProgram()
        {
            if (Peek.Kind == Tok.End) throw new PyException("InvalidExpression", "Sorry, cannot evaluate empty string");
            Node? first = null;
            while (Peek.Kind != Tok.End)
            {
                var st = ParseStatement();
                first ??= st;
                if (Peek.Kind == Tok.Newline) _p++;
                else if (Peek.Kind != Tok.End) throw Syntax("invalid syntax");
            }
            return first!;
        }

        Node ParseStatement()
        {
            if (Peek.Kind == Tok.Name && StatementKeywords.Contains(Peek.Text))
            {
                var kw = Next().Text;
                // Not an expression: skip the rest of the statement; evaluating it is refused.
                while (Peek.Kind is not (Tok.Newline or Tok.End)) _p++;
                return kw is "import" or "from" ? new Unsupported("import") : new Unsupported(char.ToUpperInvariant(kw[0]) + kw[1..]);
            }
            var expr = ParseTestList();
            if (IsOp("="))
            {
                // Assignment: simpleeval warns and evaluates the value.
                CheckTarget(expr);
                Node value = expr;
                while (IsOp("="))
                {
                    _p++;
                    value = Peek.Kind == Tok.Name && Peek.Text == "yield" ? ParseYield() : ParseTestList();
                    if (IsOp("=")) CheckTarget(value);
                }
                return value;
            }
            if (Peek.Kind == Tok.Op && Peek.Text is "+=" or "-=" or "*=" or "/=" or "//=" or "%=" or "**=" or ">>=" or "<<=" or "&=" or "|=" or "^=" or "@=")
            {
                CheckTarget(expr, augmented: true);
                _p++;
                return ParseTestList();
            }
            if (IsOp(":"))
            {
                // annotated assignment: no handler in simpleeval
                _p++;
                ParseTest();
                if (IsOp("=")) { _p++; ParseTestList(); }
                return new Unsupported("AnnAssign");
            }
            return expr;
        }

        Node ParseYield()
        {
            _p++;
            if (Peek.Kind is not (Tok.Newline or Tok.End) && !IsOp(")")) ParseTestList();
            return new Unsupported("Yield");
        }

        static void CheckTarget(Node n, bool augmented = false)
        {
            switch (n)
            {
                case Name or Attribute or Subscript: return;
                case Unsupported { Kind: "Tuple" or "List" } when !augmented: return;
                case Const { Value: null }: throw Syntax("cannot assign to None");
                case Const { Value: bool }: throw Syntax("cannot assign to True");
                case Const: throw Syntax("cannot assign to literal");
                case Call: throw Syntax("cannot assign to function call");
                default: throw Syntax("cannot assign to expression");
            }
        }

        /// <summary>test (',' test)* [','] : a bare comma makes a tuple.</summary>
        Node ParseTestList()
        {
            var first = ParseTestOrStar();
            if (!IsOp(",")) return first;
            while (IsOp(","))
            {
                _p++;
                if (Peek.Kind is Tok.Newline or Tok.End || IsOp("=") || IsOp(")") || IsOp("]") || IsOp("}")) break;
                ParseTestOrStar();
            }
            return new Unsupported("Tuple");
        }

        Node ParseTestOrStar()
        {
            if (IsOp("*")) { _p++; ParseOrExpr(); return new Unsupported("Starred"); }
            return ParseNamedExprTest();
        }

        Node ParseNamedExprTest()
        {
            var t = ParseTest();
            if (IsOp(":="))
            {
                if (t is not Name) throw Syntax("cannot use assignment expressions with this expression");
                _p++;
                ParseTest();
                return new Unsupported("NamedExpr");
            }
            return t;
        }

        public Node ParseTest()
        {
            if (IsKw("lambda"))
            {
                _p++;
                while (!IsOp(":")) { if (Peek.Kind is Tok.End or Tok.Newline) throw Syntax("invalid syntax"); _p++; }
                _p++;
                ParseTest();
                return new Unsupported("Lambda");
            }
            var body = ParseOrTest();
            if (IsKw("if"))
            {
                _p++;
                var test = ParseOrTest();
                if (!IsKw("else")) throw Syntax("expected 'else' after 'if' expression");
                _p++;
                var orelse = ParseTest();
                return new IfExp(test, body, orelse);
            }
            return body;
        }

        Node ParseOrTest()
        {
            var first = ParseAndTest();
            if (!IsKw("or")) return first;
            var list = new List<Node> { first };
            while (IsKw("or")) { _p++; list.Add(ParseAndTest()); }
            return new BoolOp(false, list);
        }

        Node ParseAndTest()
        {
            var first = ParseNotTest();
            if (!IsKw("and")) return first;
            var list = new List<Node> { first };
            while (IsKw("and")) { _p++; list.Add(ParseNotTest()); }
            return new BoolOp(true, list);
        }

        Node ParseNotTest()
        {
            if (IsKw("not")) { _p++; return new Unary("not", ParseNotTest()); }
            return ParseComparison();
        }

        Node ParseComparison()
        {
            var left = ParseOrExpr();
            List<string>? ops = null;
            List<Node>? comps = null;
            while (true)
            {
                string? op = null;
                if (Peek.Kind == Tok.Op && Peek.Text is "<" or ">" or "==" or ">=" or "<=" or "!=") op = Next().Text;
                else if (IsKw("in")) { _p++; op = "in"; }
                else if (IsKw("not") && _t[_p + 1].Kind == Tok.Name && _t[_p + 1].Text == "in") { _p += 2; op = "not in"; }
                else if (IsKw("is"))
                {
                    _p++;
                    if (IsKw("not")) { _p++; op = "is not"; } else op = "is";
                }
                if (op is null) break;
                (ops ??= []).Add(op);
                (comps ??= []).Add(ParseOrExpr());
            }
            return ops is null ? left : new Compare(left, ops, comps!);
        }

        Node ParseOrExpr() => ParseBinaryLevel(0);

        static readonly string[][] Levels = [["|"], ["^"], ["&"], ["<<", ">>"], ["+", "-"], ["*", "@", "/", "%", "//"]];

        Node ParseBinaryLevel(int level)
        {
            if (level == Levels.Length) return ParseFactor();
            var left = ParseBinaryLevel(level + 1);
            while (Peek.Kind == Tok.Op && Levels[level].Contains(Peek.Text))
            {
                var op = Next().Text;
                left = new Binary(op, left, ParseBinaryLevel(level + 1));
            }
            return left;
        }

        Node ParseFactor()
        {
            if (Peek.Kind == Tok.Op && Peek.Text is "+" or "-" or "~")
            {
                var op = Next().Text;
                return new Unary(op, ParseFactor());
            }
            return ParsePower();
        }

        Node ParsePower()
        {
            Node b;
            if (IsKw("await")) { _p++; ParsePrimary(); b = new Unsupported("Await"); }
            else b = ParsePrimary();
            if (IsOp("**")) { _p++; return new Binary("**", b, ParseFactor()); }
            return b;
        }

        Node ParsePrimary()
        {
            var node = ParseAtom();
            while (true)
            {
                if (IsOp("("))
                {
                    _p++;
                    node = ParseCallArgs(node);
                }
                else if (IsOp("["))
                {
                    _p++;
                    node = new Subscript(node, ParseSubscriptList());
                    Expect("]");
                }
                else if (IsOp("."))
                {
                    _p++;
                    if (Peek.Kind != Tok.Name || Keywords.Contains(Peek.Text)) throw Syntax("invalid syntax");
                    node = new Attribute(node, Next().Text);
                }
                else return node;
            }
        }

        Node ParseCallArgs(Node func)
        {
            var args = new List<Node>();
            var kws = new List<(string?, Node)>();
            while (!IsOp(")"))
            {
                if (IsOp("**")) { _p++; kws.Add((null, ParseTest())); }
                else if (IsOp("*")) { _p++; ParseTest(); args.Add(new Unsupported("Starred")); }
                else if (Peek.Kind == Tok.Name && _t[_p + 1].Kind == Tok.Op && _t[_p + 1].Text == "=")
                {
                    var name = Next().Text;
                    if (Keywords.Contains(name)) throw Syntax("invalid syntax");
                    _p++;
                    kws.Add((name, ParseTest()));
                }
                else
                {
                    var a = ParseNamedExprTest();
                    if (IsKw("for") || IsKw("async")) { SkipComprehension(); a = new Unsupported("GeneratorExp"); }
                    if (kws.Count > 0) throw Syntax("positional argument follows keyword argument");
                    args.Add(a);
                }
                if (!IsOp(",")) break;
                _p++;
            }
            Expect(")");
            return new Call(func, args, kws);
        }

        /// <summary>for targets in iterable [if cond]... : parsed for syntax only (simpleeval's base evaluator has no comprehensions).</summary>
        void SkipComprehension()
        {
            while (IsKw("for") || IsKw("async"))
            {
                if (IsKw("async")) _p++;
                _p++; // for
                ParseOrExpr();
                while (IsOp(",")) { _p++; ParseOrExpr(); }
                if (!IsKw("in")) throw Syntax("invalid syntax");
                _p++;
                ParseOrTest();
                while (IsKw("if")) { _p++; ParseOrTest(); }
            }
        }

        Node ParseSubscriptList()
        {
            var first = ParseSubscript();
            if (!IsOp(",")) return first;
            while (IsOp(",")) { _p++; if (IsOp("]")) break; ParseSubscript(); }
            return new Unsupported("Tuple");
        }

        Node ParseSubscript()
        {
            Node? lower = null, upper = null, step = null;
            if (!IsOp(":")) { lower = ParseNamedExprTest(); if (!IsOp(":")) return lower; }
            _p++; // ':'
            if (!IsOp(":") && !IsOp("]") && !IsOp(",")) upper = ParseTest();
            if (IsOp(":")) { _p++; if (!IsOp("]") && !IsOp(",")) step = ParseTest(); }
            return new SliceNode(lower, upper, step);
        }

        Node ParseAtom()
        {
            var tok = Peek;
            switch (tok.Kind)
            {
                case Tok.Number:
                    _p++;
                    var number = NumberValue(tok.Text);
                    return number as Unsupported ?? (Node)new Const(number);
                case Tok.String:
                    return ParseStrings();
                case Tok.Name:
                    _p++;
                    switch (tok.Text)
                    {
                        case "True": return new Const(true);
                        case "False": return new Const(false);
                        case "None": return new Const(null);
                    }
                    if (Keywords.Contains(tok.Text)) throw Syntax("invalid syntax");
                    return new Name(tok.Text);
                case Tok.Op:
                    switch (tok.Text)
                    {
                        case "(":
                        {
                            _p++;
                            if (IsOp(")")) { _p++; return new Unsupported("Tuple"); }
                            if (IsKw("yield")) { var y = ParseYield(); Expect(")"); return y; }
                            var first = ParseTestOrStar();
                            if (IsKw("for") || IsKw("async")) { SkipComprehension(); Expect(")"); return new Unsupported("GeneratorExp"); }
                            if (IsOp(","))
                            {
                                while (IsOp(",")) { _p++; if (IsOp(")")) break; ParseTestOrStar(); }
                                Expect(")");
                                return new Unsupported("Tuple");
                            }
                            Expect(")");
                            if (first is Unsupported { Kind: "Starred" }) throw Syntax("cannot use starred expression here");
                            return first;
                        }
                        case "[":
                        {
                            _p++;
                            if (IsOp("]")) { _p++; return new Unsupported("List"); }
                            ParseTestOrStar();
                            if (IsKw("for") || IsKw("async")) { SkipComprehension(); Expect("]"); return new Unsupported("ListComp"); }
                            while (IsOp(",")) { _p++; if (IsOp("]")) break; ParseTestOrStar(); }
                            Expect("]");
                            return new Unsupported("List");
                        }
                        case "{":
                        {
                            _p++;
                            if (IsOp("}")) { _p++; return new Unsupported("Dict"); }
                            bool dict;
                            if (IsOp("**")) { _p++; ParseOrExpr(); dict = true; }
                            else
                            {
                                ParseTestOrStar();
                                dict = IsOp(":");
                                if (dict) { _p++; ParseTest(); }
                            }
                            if (IsKw("for") || IsKw("async")) { SkipComprehension(); Expect("}"); return new Unsupported(dict ? "DictComp" : "SetComp"); }
                            while (IsOp(","))
                            {
                                _p++;
                                if (IsOp("}")) break;
                                if (dict)
                                {
                                    if (IsOp("**")) { _p++; ParseOrExpr(); }
                                    else { ParseTest(); Expect(":"); ParseTest(); }
                                }
                                else ParseTestOrStar();
                            }
                            Expect("}");
                            return new Unsupported(dict ? "Dict" : "Set");
                        }
                        case "...":
                            _p++;
                            return new Ellipsis();
                    }
                    break;
                case Tok.End:
                    throw Syntax("unexpected EOF while parsing");
            }
            throw Syntax("invalid syntax");
        }

        /// <summary>Adjacent string literals are one string (an f-string among them makes the whole a JoinedStr).</summary>
        Node ParseStrings()
        {
            var pieces = new List<StringPiece>();
            while (Peek.Kind == Tok.String) pieces.AddRange(Next().Pieces!);
            bool anyBytes = pieces.Any(p => p.Prefix.Contains('b')), allBytes = pieces.All(p => p.Prefix.Contains('b'));
            if (anyBytes && !allBytes) throw Syntax("cannot mix bytes and nonbytes literals");
            if (anyBytes) return new Unsupported("bytes");
            if (!pieces.Any(p => p.Prefix.Contains('f')))
                return new Const(string.Concat(pieces.Select(p => p.Prefix.Contains('r') ? p.Body : Unescape(p.Body))));
            var parts = new List<Node>();
            var lit = new StringBuilder();
            foreach (var p in pieces)
            {
                if (!p.Prefix.Contains('f')) { lit.Append(p.Prefix.Contains('r') ? p.Body : Unescape(p.Body)); continue; }
                ParseFString(p.Body, p.Prefix.Contains('r'), parts, lit);
            }
            if (lit.Length > 0) parts.Add(new Const(lit.ToString()));
            return new JoinedStr(parts);
        }

        static void ParseFString(string body, bool raw, List<Node> parts, StringBuilder lit)
        {
            int i = 0, n = body.Length;
            var chunk = new StringBuilder();
            void FlushChunk() { if (chunk.Length > 0) { lit.Append(raw ? chunk.ToString() : Unescape(chunk.ToString())); chunk.Clear(); } }
            while (i < n)
            {
                char c = body[i];
                if (c == '{')
                {
                    if (i + 1 < n && body[i + 1] == '{') { chunk.Append('{'); i += 2; continue; }
                    FlushChunk();
                    var (node, debugText) = ParseField(body, ref i, raw);
                    if (debugText is not null) lit.Append(debugText);
                    if (lit.Length > 0) { parts.Add(new Const(lit.ToString())); lit.Clear(); }
                    parts.Add(node);
                    continue;
                }
                if (c == '}')
                {
                    if (i + 1 < n && body[i + 1] == '}') { chunk.Append('}'); i += 2; continue; }
                    throw Syntax("f-string: single '}' is not allowed");
                }
                if (c == '\\' && !raw && i + 1 < n) { chunk.Append(c).Append(body[i + 1]); i += 2; continue; }
                chunk.Append(c);
                i++;
            }
            FlushChunk();
        }

        /// <summary>One <c>{expr[=][!conv][:spec]}</c> field starting at <paramref name="i"/> (on the '{').</summary>
        static (FormattedValue Node, string? DebugText) ParseField(string body, ref int i, bool raw)
        {
            int n = body.Length;
            int start = ++i;
            int depth = 0;
            char? quote = null;
            while (i < n)
            {
                char c = body[i];
                if (quote is { } q) { if (c == '\\') { i += 2; continue; } if (c == q) quote = null; i++; continue; }
                if (c is '\'' or '"') { quote = c; i++; continue; }
                if (c is '(' or '[' or '{') depth++;
                else if (c is ')' or ']' or '}') { if (depth == 0) break; depth--; }
                else if (depth == 0 && c == '!' && !(i + 1 < n && body[i + 1] == '=')) break;
                else if (depth == 0 && c == ':') break;
                else if (depth == 0 && c == '=' && !(i + 1 < n && body[i + 1] == '=') && !(i > start && body[i - 1] is '=' or '!' or '<' or '>')) break;
                i++;
            }
            if (i >= n) throw Syntax("f-string: expecting '}'");
            var exprText = body[start..i];
            if (PyText.Strip(exprText).Length == 0) throw Syntax("f-string: valid expression required before '}'");
            var expr = new Parser(exprText.Trim()).ParseFieldExpression();
            string? debug = null;
            if (body[i] == '=') { debug = exprText + "="; i++; }
            if (i < n && body[i] == '!')
            {
                if (i + 1 >= n || body[i + 1] is not ('r' or 's' or 'a')) throw Syntax("f-string: invalid conversion character: expected 's', 'r', or 'a'");
                i += 2; // the conversion: simpleeval ignores it (it formats the value itself)
            }
            JoinedStr? spec = null;
            if (i < n && body[i] == ':')
            {
                i++;
                var specParts = new List<Node>();
                var lit = new StringBuilder();
                while (i < n && body[i] != '}')
                {
                    if (body[i] == '{')
                    {
                        if (lit.Length > 0) { specParts.Add(new Const(lit.ToString())); lit.Clear(); }
                        var (inner, _) = ParseField(body, ref i, raw);
                        specParts.Add(inner);
                        continue;
                    }
                    lit.Append(body[i]);
                    i++;
                }
                if (lit.Length > 0) specParts.Add(new Const(raw ? lit.ToString() : Unescape(lit.ToString())));
                spec = new JoinedStr(specParts);
            }
            if (i >= n || body[i] != '}') throw Syntax("f-string: expecting '}'");
            i++;
            return (new FormattedValue(expr, spec), debug);
        }

        Node ParseFieldExpression()
        {
            var e = ParseTestList();
            if (Peek.Kind != Tok.End) throw Syntax("f-string: invalid syntax");
            return e;
        }
    }

    /// <summary>Python string escapes: \\ \' \" \a \b \f \n \r \t \v \ooo \xhh \uXXXX \UXXXXXXXX and a backslash-newline; unknown escapes stay as written.</summary>
    static string Unescape(string s)
    {
        if (!s.Contains('\\')) return s;
        var sb = new StringBuilder(s.Length);
        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            if (c != '\\' || i + 1 >= s.Length) { sb.Append(c); continue; }
            char e = s[++i];
            switch (e)
            {
                case '\n': break;
                case '\r': if (i + 1 < s.Length && s[i + 1] == '\n') i++; break;
                case '\\': sb.Append('\\'); break;
                case '\'': sb.Append('\''); break;
                case '"': sb.Append('"'); break;
                case 'a': sb.Append('\a'); break;
                case 'b': sb.Append('\b'); break;
                case 'f': sb.Append('\f'); break;
                case 'n': sb.Append('\n'); break;
                case 'r': sb.Append('\r'); break;
                case 't': sb.Append('\t'); break;
                case 'v': sb.Append('\v'); break;
                case >= '0' and <= '7':
                {
                    int v = e - '0';
                    for (int k = 0; k < 2 && i + 1 < s.Length && s[i + 1] is >= '0' and <= '7'; k++) v = v * 8 + (s[++i] - '0');
                    sb.Append(char.ConvertFromUtf32(v));
                    break;
                }
                case 'x' or 'u' or 'U':
                {
                    int len = e == 'x' ? 2 : e == 'u' ? 4 : 8;
                    if (i + len >= s.Length) throw Syntax($"(unicode error) truncated \\{e} escape");
                    var hex = s.Substring(i + 1, len);
                    if (!hex.All(char.IsAsciiHexDigit) || !int.TryParse(hex, NumberStyles.AllowHexSpecifier, CultureInfo.InvariantCulture, out var cp))
                        throw Syntax($"(unicode error) truncated \\{e} escape");
                    if (cp > 0x10FFFF) throw Syntax("(unicode error) illegal Unicode character");
                    sb.Append(cp is >= 0xD800 and <= 0xDFFF ? ((char)cp).ToString() : char.ConvertFromUtf32(cp));
                    i += len;
                    break;
                }
                case 'N': throw new NotSupportedException("\\N{...} (characters by Unicode name) is not supported in expressions here; use \\uXXXX.");
                default: sb.Append('\\').Append(e); break;
            }
        }
        return sb.ToString();
    }

    // ==== values ===================================================================================================

    /// <summary>A built-in function (ComfyUI's MATH_FUNCTIONS) or a bound method such as <c>x.is_integer</c>.</summary>
    sealed class PyFunction(string name, Func<List<object?>, List<(string Name, object? Value)>, object?> call)
    {
        public string Name { get; } = name;
        public object? Invoke(List<object?> args, List<(string, object?)> kwargs) => call(args, kwargs);
        public override string ToString() => $"<built-in function {Name}>";
    }

    sealed class PySlice(object? start, object? stop, object? step)
    {
        public object? Start = start, Stop = stop, Step = step;
    }

    static bool IsInt(object? v) => v is BigInteger or bool;
    static bool IsNumber(object? v) => v is BigInteger or bool or double;
    static BigInteger AsInt(object v) => v is bool b ? (b ? BigInteger.One : BigInteger.Zero) : (BigInteger)v;

    static double AsFloat(object? v)
    {
        if (v is double d) return d;
        try { return PyNumber.ToDouble(AsInt(v!)); }
        catch (OverflowException) { throw new PyException("OverflowError", "int too large to convert to float"); }
    }

    static string TypeName(object? v) => v switch
    {
        PyFunction => "builtin_function_or_method",
        PySlice => "slice",
        Ellipsis => "ellipsis",
        _ => PyFormat.TypeName(v),
    };

    static bool HasLen(object? v) => v is string or System.Collections.IList or PyTuple;
    static int Len(object? v) => v switch
    {
        string s => PyText.Length(s),
        System.Collections.IList l => l.Count,
        PyTuple t => t.Items.Length,
        _ => throw new PyException("TypeError", $"object of type '{TypeName(v)}' has no len()"),
    };

    public static bool Truthy(object? v) => v switch
    {
        null => false,
        bool b => b,
        BigInteger i => !i.IsZero,
        long l => l != 0,
        double d => d != 0,
        string s => s.Length > 0,
        System.Collections.IList l => l.Count > 0,
        PyTuple t => t.Items.Length > 0,
        _ => true,
    };

    // ==== evaluator ================================================================================================

    sealed class Evaluator(IReadOnlyDictionary<string, object?> names, string expr)
    {
        readonly Dictionary<string, PyFunction> _functions = MathFunctions();

        PyException Feature(string what) => new("FeatureNotAvailable", what);

        public object? Eval(Node node)
        {
            switch (node)
            {
                case Const c:
                    if (c.Value is string s && s.Length > MaxStringLength)
                        throw new PyException("IterableTooLong", $"Literal in statement is too long! ({s.Length}, when {MaxStringLength} is max)");
                    return c.Value;
                case Name n: return EvalName(n);
                case Unary u: return EvalUnary(u);
                case Binary b: return BinaryOp(b.Op, Eval(b.Left), Eval(b.Right));
                case BoolOp bo:
                {
                    object? r = false;
                    foreach (var v in bo.Values)
                    {
                        r = Eval(v);
                        if (bo.And ? !Truthy(r) : Truthy(r)) break;
                    }
                    return r;
                }
                case Compare cmp:
                {
                    var right = Eval(cmp.Left);
                    bool ok = true;
                    for (int i = 0; i < cmp.Ops.Count; i++)
                    {
                        if (!ok) break;
                        var left = right;
                        right = Eval(cmp.Comparators[i]);
                        ok = CompareOp(cmp.Ops[i], left, right);
                    }
                    return ok;
                }
                case IfExp ie: return Truthy(Eval(ie.Test)) ? Eval(ie.Body) : Eval(ie.OrElse);
                case Call call: return EvalCall(call);
                case Subscript sub: return Index(Eval(sub.Value), Eval(sub.Index));
                case SliceNode sl: return new PySlice(sl.Lower is null ? null : Eval(sl.Lower), sl.Upper is null ? null : Eval(sl.Upper), sl.Step is null ? null : Eval(sl.Step));
                case Attribute a: return EvalAttribute(a);
                case JoinedStr js:
                {
                    var sb = new StringBuilder();
                    foreach (var part in js.Values)
                    {
                        var val = PyText.Str(Eval(part));
                        if (val.Length + sb.Length > MaxStringLength) throw new PyException("IterableTooLong", "Sorry, I will not evaluate something this long.");
                        sb.Append(val);
                    }
                    return sb.ToString();
                }
                case FormattedValue fv:
                    if (fv.Spec is { } spec) return PyFormat.Format(Eval(fv.Value), (string)Eval(spec)!);
                    return Eval(fv.Value);
                case Ellipsis: throw new NotSupportedException("The Ellipsis constant (...) is not supported here.");
                case Unsupported us:
                    if (us.Kind == "import") throw Feature("Sorry, 'import' is not allowed.");
                    if (us.Kind == "complex number") throw new NotSupportedException("Complex numbers (1j) are not supported here.");
                    if (us.Kind == "bytes") throw new NotSupportedException("Bytes literals (b'...') are not supported here.");
                    if (us.Kind == "Lambda") throw Feature("Sorry, Lambda is not available in this evaluator");
                    throw Feature($"Sorry, {us.Kind} is not available in this evaluator");
            }
            throw Feature($"Sorry, {node.GetType().Name} is not available in this evaluator");
        }

        object? EvalName(Name n)
        {
            if (names.TryGetValue(n.Id, out var v)) return v;
            if (_functions.TryGetValue(n.Id, out var f)) return f;
            throw new PyException("NameNotDefined", $"'{n.Id}' is not defined for expression '{expr}'");
        }

        object? EvalUnary(Unary u)
        {
            var v = Eval(u.Operand);
            switch (u.Op)
            {
                case "not": return !Truthy(v);
                case "-":
                    if (v is double d) return -d;
                    if (IsInt(v)) return -AsInt(v!);
                    break;
                case "+":
                    if (v is double d2) return d2;
                    if (IsInt(v)) return AsInt(v!);
                    break;
                case "~":
                    if (IsInt(v)) return -AsInt(v!) - 1;
                    break;
            }
            throw new PyException("TypeError", $"bad operand type for unary {u.Op}: '{TypeName(v)}'");
        }

        object? EvalCall(Call call)
        {
            PyFunction func;
            if (call.Func is Attribute)
            {
                var f = Eval(call.Func);
                func = f as PyFunction ?? throw new PyException("TypeError", $"'{TypeName(f)}' object is not callable");
            }
            else if (call.Func is Name fn)
            {
                if (!_functions.TryGetValue(fn.Id, out func!)) throw new PyException("FunctionNotDefined", $"Function '{fn.Id}' not defined, for expression '{expr}'.");
            }
            else throw Feature("Lambda Functions not implemented");
            var args = call.Args.Select(Eval).ToList();
            var kwargs = new List<(string, object?)>();
            foreach (var (name, value) in call.Keywords)
            {
                var v = Eval(value);
                if (name is null) throw new PyException("TypeError", "keywords must be strings");
                kwargs.Add((name, v));
            }
            return func.Invoke(args, kwargs);
        }

        object? EvalAttribute(Attribute a)
        {
            foreach (var prefix in DisallowPrefixes)
                if (a.Attr.StartsWith(prefix, StringComparison.Ordinal))
                    throw Feature($"Sorry, access to __attributes  or func_ attributes is not available. ({a.Attr})");
            if (DisallowMethods.Contains(a.Attr)) throw Feature($"Sorry, this method is not available. ({a.Attr})");
            var v = Eval(a.Value);
            return GetAttribute(v, a.Attr) ?? throw new PyException("AttributeDoesNotExist", $"Attribute '{a.Attr}' does not exist in expression '{expr}'");
        }
    }

    // ==== operators ================================================================================================

    static PyException Unsupported2(string op, object? a, object? b) =>
        new("TypeError", $"unsupported operand type(s) for {op}: '{TypeName(a)}' and '{TypeName(b)}'");

    static object? BinaryOp(string op, object? a, object? b)
    {
        switch (op)
        {
            case "+": return Add(a, b);
            case "-":
                if (IsInt(a) && IsInt(b)) return AsInt(a!) - AsInt(b!);
                if (IsNumber(a) && IsNumber(b)) return AsFloat(a) - AsFloat(b);
                throw Unsupported2("-", a, b);
            case "*": return Mult(a, b);
            case "/":
                if (IsInt(a) && IsInt(b))
                {
                    if (AsInt(b!).IsZero) throw new PyException("ZeroDivisionError", "division by zero");
                    try { return PyNumber.TrueDivide(AsInt(a!), AsInt(b!)); }
                    catch (OverflowException e) { throw new PyException("OverflowError", e.Message); }
                }
                if (IsNumber(a) && IsNumber(b))
                {
                    double y = AsFloat(b);
                    if (y == 0) throw new PyException("ZeroDivisionError", "float division by zero");
                    return AsFloat(a) / y;
                }
                throw Unsupported2("/", a, b);
            case "//":
                if (IsInt(a) && IsInt(b))
                {
                    var y = AsInt(b!);
                    if (y.IsZero) throw new PyException("ZeroDivisionError", "integer division or modulo by zero");
                    return FloorDiv(AsInt(a!), y);
                }
                if (IsNumber(a) && IsNumber(b)) return FloatDivMod(AsFloat(a), AsFloat(b), "float floor division by zero").Div;
                throw Unsupported2("//", a, b);
            case "%":
                if (IsInt(a) && IsInt(b))
                {
                    var y = AsInt(b!);
                    if (y.IsZero) throw new PyException("ZeroDivisionError", "integer modulo by zero");
                    return AsInt(a!) - FloorDiv(AsInt(a!), y) * y;
                }
                if (IsNumber(a) && IsNumber(b)) return FloatDivMod(AsFloat(a), AsFloat(b), "float modulo").Mod;
                if (a is string) throw new NotSupportedException("printf-style string formatting ('...' % values) is not supported here.");
                throw Unsupported2("%", a, b);
            case "**": return SafePower(a, b);
            case "<<" or ">>":
            {
                // simpleeval's safe_lshift / safe_rshift limits
                if (Greater(Abs(b), MaxShift) || Greater(Abs(a), MaxShiftBase))
                    throw new PyException("NumberTooHigh", $"Sorry! I don't want to evaluate {PyText.Repr(a)} {op} {PyText.Repr(b)}");
                if (!IsInt(a) || !IsInt(b)) throw Unsupported2(op, a, b);
                var count = AsInt(b!);
                if (count.Sign < 0) throw new PyException("ValueError", "negative shift count");
                return op == "<<" ? AsInt(a!) << (int)count : AsInt(a!) >> (int)count;
            }
            case "&" or "|" or "^":
                if (a is bool ba && b is bool bb) return op == "&" ? ba & bb : op == "|" ? ba | bb : ba ^ bb;
                if (IsInt(a) && IsInt(b))
                {
                    BigInteger x = AsInt(a!), y = AsInt(b!);
                    return op == "&" ? x & y : op == "|" ? x | y : x ^ y;
                }
                throw Unsupported2(op, a, b);
            case "@":
                throw new PyException("OperatorNotDefined", "Operator '<ast.MatMult>' does not exist");
        }
        throw new PyException("OperatorNotDefined", $"Operator '{op}' does not exist");
    }

    static BigInteger FloorDiv(BigInteger a, BigInteger b)
    {
        var q = BigInteger.DivRem(a, b, out var r);
        if (!r.IsZero && (r.Sign < 0) != (b.Sign < 0)) q -= 1;
        return q;
    }

    /// <summary>CPython's float divmod (Objects/floatobject.c).</summary>
    static (double Div, double Mod) FloatDivMod(double vx, double wx, string zeroMessage)
    {
        if (wx == 0) throw new PyException("ZeroDivisionError", zeroMessage);
        double mod = vx % wx; // C's fmod: the sign of the dividend
        double div = (vx - mod) / wx;
        if (mod != 0)
        {
            if ((wx < 0) != (mod < 0)) { mod += wx; div -= 1.0; }
        }
        else mod = Math.CopySign(0.0, wx);
        double floordiv;
        if (div != 0)
        {
            floordiv = Math.Floor(div);
            if (div - floordiv > 0.5) floordiv += 1.0;
        }
        else floordiv = Math.CopySign(0.0, vx / wx);
        return (floordiv, mod);
    }

    static object? Add(object? a, object? b)
    {
        if (HasLen(a) && HasLen(b) && Len(a) + Len(b) > MaxStringLength)
            throw new PyException("IterableTooLong", "Sorry, adding those two together would make something too long.");
        if (IsInt(a) && IsInt(b)) return AsInt(a!) + AsInt(b!);
        if (IsNumber(a) && IsNumber(b)) return AsFloat(a) + AsFloat(b);
        if (a is string sa)
            return b is string sb ? sa + sb : throw new PyException("TypeError", $"can only concatenate str (not \"{TypeName(b)}\") to str");
        if (a is System.Collections.IList la)
        {
            if (b is System.Collections.IList lb) return la.Cast<object?>().Concat(lb.Cast<object?>()).ToList();
            throw new PyException("TypeError", $"can only concatenate list (not \"{TypeName(b)}\") to list");
        }
        if (a is PyTuple ta && b is PyTuple tb) return new PyTuple([.. ta.Items, .. tb.Items]);
        throw Unsupported2("+", a, b);
    }

    static object? Mult(object? a, object? b)
    {
        // simpleeval's safe_mult: refuse to build a sequence longer than MAX_STRING_LENGTH
        if (HasLen(a) && Greater(BinaryOp("*", b, new BigInteger(Len(a))), MaxStringLength)) throw new PyException("IterableTooLong", "Sorry, I will not evaluate something that long.");
        if (HasLen(b) && Greater(BinaryOp("*", a, new BigInteger(Len(b))), MaxStringLength)) throw new PyException("IterableTooLong", "Sorry, I will not evaluate something that long.");
        if (IsInt(a) && IsInt(b)) return AsInt(a!) * AsInt(b!);
        if (IsNumber(a) && IsNumber(b)) return AsFloat(a) * AsFloat(b);
        if (HasLen(a) && IsInt(b)) return Repeat(a, AsInt(b!));
        if (HasLen(b) && IsInt(a)) return Repeat(b, AsInt(a!));
        if (HasLen(a) && b is double || HasLen(b) && a is double)
            throw new PyException("TypeError", "can't multiply sequence by non-int of type 'float'");
        throw Unsupported2("*", a, b);
    }

    static object Repeat(object? seq, BigInteger times)
    {
        // safe_mult already refused results over MAX_STRING_LENGTH; an empty sequence stays empty however often it is repeated
        int n = times.Sign <= 0 || Len(seq) == 0 ? 0 : (int)times;
        switch (seq)
        {
            case string s: return string.Concat(Enumerable.Repeat(s, n));
            case PyTuple t: return new PyTuple(Enumerable.Range(0, n).SelectMany(_ => t.Items).ToArray());
            default:
                var list = ((System.Collections.IList)seq!).Cast<object?>().ToList();
                return Enumerable.Range(0, n).SelectMany(_ => list).ToList();
        }
    }

    /// <summary><c>a &gt; limit</c> for a number (Python comparison); TypeError for anything else, as Python's would be.</summary>
    static bool Greater(object? a, BigInteger limit)
    {
        if (a is double d) return PyNumber.CompareIntFloat(limit, d) is < 0;
        if (IsInt(a)) return AsInt(a!) > limit;
        throw new PyException("TypeError", $"'>' not supported between instances of '{TypeName(a)}' and 'int'");
    }

    static object? Abs(object? v)
    {
        if (v is double d) return Math.Abs(d);
        if (IsInt(v)) return BigInteger.Abs(AsInt(v!));
        throw new PyException("TypeError", $"bad operand type for abs(): '{TypeName(v)}'");
    }

    /// <summary>simpleeval's <c>safe_power</c> (both operands at most 4,000,000 in size), then Python's <c>**</c>.</summary>
    static object? SafePower(object? a, object? b)
    {
        if (Greater(Abs(a), MaxPower) || Greater(Abs(b), MaxPower))
            throw new PyException("NumberTooHigh", $"Sorry! I don't want to evaluate {PyText.Repr(a)} ** {PyText.Repr(b)}");
        return Power(a, b);
    }

    static object? Power(object? a, object? b)
    {
        if (IsInt(a) && IsInt(b))
        {
            BigInteger x = AsInt(a!), y = AsInt(b!);
            if (y.Sign >= 0) return BigInteger.Pow(x, (int)y);
            if (x.IsZero) throw new PyException("ZeroDivisionError", "0.0 cannot be raised to a negative power");
            return FloatPow(AsFloat(x), AsFloat(y));
        }
        if (IsNumber(a) && IsNumber(b)) return FloatPow(AsFloat(a), AsFloat(b));
        throw Unsupported2("** or pow()", a, b);
    }

    static bool IsOddInteger(double x) => Math.Abs(x % 2.0) == 1.0;

    /// <summary>CPython's float_pow special cases, then C pow.</summary>
    static double FloatPow(double iv, double iw)
    {
        if (iw == 0) return 1.0;
        if (double.IsNaN(iv)) return iv;
        if (double.IsNaN(iw)) return iv == 1.0 ? 1.0 : iw;
        if (double.IsInfinity(iw))
        {
            iv = Math.Abs(iv);
            if (iv == 1.0) return 1.0;
            return (iw > 0) == (iv > 1.0) ? Math.Abs(iw) : 0.0;
        }
        if (double.IsInfinity(iv))
        {
            bool odd = IsOddInteger(iw);
            return iw > 0 ? (odd ? iv : Math.Abs(iv)) : (odd ? Math.CopySign(0.0, iv) : 0.0);
        }
        if (iv == 0.0)
        {
            bool odd = IsOddInteger(iw);
            if (iw < 0.0) throw new PyException("ZeroDivisionError", "0.0 cannot be raised to a negative power");
            return odd ? iv : 0.0;
        }
        bool negate = false;
        if (iv < 0.0)
        {
            if (iw != Math.Floor(iw)) throw new NotSupportedException("A negative number to a fractional power is a complex number in Python; complex results are not supported here.");
            iv = -iv;
            negate = IsOddInteger(iw);
        }
        if (iv == 1.0) return negate ? -1.0 : 1.0;
        double ix = Math.Pow(iv, iw);
        if (double.IsInfinity(ix)) throw new PyException("OverflowError", "(34, 'Numerical result out of range')");
        return negate ? -ix : ix;
    }

    // ==== comparisons ==============================================================================================

    static bool CompareOp(string op, object? a, object? b) => op switch
    {
        "==" => PyEquals(a, b),
        "!=" => !PyEquals(a, b),
        "<" => Order(a, b, "<") is < 0,
        "<=" => Order(a, b, "<=") is <= 0,
        ">" => Order(a, b, ">") is > 0,
        ">=" => Order(a, b, ">=") is >= 0,
        "in" => Contains(b, a),
        "not in" => !Contains(b, a),
        "is" => Identical(a, b),
        "is not" => !Identical(a, b),
        _ => throw new PyException("OperatorNotDefined", $"Operator '{op}' does not exist"),
    };

    static bool Identical(object? a, object? b)
    {
        if (a is null || b is null) return a is null && b is null;
        if (a is bool x && b is bool y) return x == y;
        if (a is BigInteger i && b is BigInteger j) return i == j && i >= -5 && i <= 256; // CPython caches the small ints
        return ReferenceEquals(a, b);
    }

    public static bool PyEquals(object? a, object? b)
    {
        if (IsNumber(a) && IsNumber(b))
        {
            if (IsInt(a) && IsInt(b)) return AsInt(a!) == AsInt(b!);
            if (a is double da && b is double db) return da == db;
            return a is double d1 ? PyNumber.CompareIntFloat(AsInt(b!), d1) == 0 : PyNumber.CompareIntFloat(AsInt(a!), (double)b!) == 0;
        }
        switch (a)
        {
            case null: return b is null;
            case string sa: return b is string sb && sa == sb;
            case PyTuple ta: return b is PyTuple tb && SeqEquals(ta.Items, tb.Items);
            case System.Collections.IList la when b is System.Collections.IList lb && a is not string: return SeqEquals(la.Cast<object?>().ToList(), lb.Cast<object?>().ToList());
        }
        return ReferenceEquals(a, b);
    }

    static bool SeqEquals(IReadOnlyList<object?> a, IReadOnlyList<object?> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++) if (!ReferenceEquals(a[i], b[i]) && !PyEquals(a[i], b[i])) return false;
        return true;
    }

    /// <summary>The sign of a - b for Python's ordering operators; null when unordered (NaN). TypeError for types Python cannot order.</summary>
    static int? Order(object? a, object? b, string op)
    {
        if (IsNumber(a) && IsNumber(b))
        {
            if (IsInt(a) && IsInt(b)) return AsInt(a!).CompareTo(AsInt(b!));
            if (a is double da && b is double db) return double.IsNaN(da) || double.IsNaN(db) ? null : da.CompareTo(db);
            if (a is double d1) return -PyNumber.CompareIntFloat(AsInt(b!), d1);
            return PyNumber.CompareIntFloat(AsInt(a!), (double)b!);
        }
        if (a is string sa && b is string sb) return CompareCodePoints(sa, sb);
        if (a is System.Collections.IList la && b is System.Collections.IList lb && a is not string)
            return OrderSeq(la.Cast<object?>().ToList(), lb.Cast<object?>().ToList(), op);
        if (a is PyTuple ta && b is PyTuple tb) return OrderSeq(ta.Items, tb.Items, op);
        throw new PyException("TypeError", $"'{op}' not supported between instances of '{TypeName(a)}' and '{TypeName(b)}'");
    }

    static int? OrderSeq(IReadOnlyList<object?> a, IReadOnlyList<object?> b, string op)
    {
        int n = Math.Min(a.Count, b.Count);
        for (int i = 0; i < n; i++)
            if (!PyEquals(a[i], b[i])) return Order(a[i], b[i], op);
        return a.Count.CompareTo(b.Count);
    }

    static int CompareCodePoints(string a, string b)
    {
        var x = PyText.CodePoints(a);
        var y = PyText.CodePoints(b);
        int n = Math.Min(x.Length, y.Length);
        for (int i = 0; i < n; i++) if (x[i] != y[i]) return x[i].CompareTo(y[i]);
        return x.Length.CompareTo(y.Length);
    }

    static bool Contains(object? container, object? item)
    {
        switch (container)
        {
            case string s:
                if (item is not string sub) throw new PyException("TypeError", $"'in <string>' requires string as left operand, not {TypeName(item)}");
                return s.Contains(sub, StringComparison.Ordinal);
            case PyTuple t: return t.Items.Any(x => ReferenceEquals(x, item) || PyEquals(x, item));
            case System.Collections.IList l: return l.Cast<object?>().Any(x => ReferenceEquals(x, item) || PyEquals(x, item));
        }
        throw new PyException("TypeError", $"argument of type '{TypeName(container)}' is not iterable");
    }

    // ==== subscripts and attributes ================================================================================

    static long IndexValue(object? v, string what)
    {
        if (!IsInt(v)) throw new PyException("TypeError", what);
        var i = AsInt(v!);
        if (i < long.MinValue || i > long.MaxValue) throw new PyException("IndexError", "cannot fit 'int' into an index-sized integer");
        return (long)i;
    }

    static object? Index(object? container, object? key)
    {
        if (key is PySlice sl)
        {
            const string msg = "slice indices must be integers or None or have an __index__ method";
            long? start = sl.Start is null ? null : IndexValue(sl.Start, msg);
            long? stop = sl.Stop is null ? null : IndexValue(sl.Stop, msg);
            long? step = sl.Step is null ? null : IndexValue(sl.Step, msg);
            if (step == 0) throw new PyException("ValueError", "slice step cannot be zero");
            switch (container)
            {
                case string s: return PyText.FromCodePoints(PyText.SliceItems(PyText.CodePoints(s), start, stop, step));
                case PyTuple t: return new PyTuple(PyText.SliceItems(t.Items, start, stop, step).ToArray());
                case System.Collections.IList l: return PyText.SliceItems(l.Cast<object?>().ToList(), start, stop, step).ToList();
            }
            throw new PyException("TypeError", $"'{TypeName(container)}' object is not subscriptable");
        }
        switch (container)
        {
            case string s:
            {
                var cps = PyText.CodePoints(s);
                long i = IndexValue(key, $"string indices must be integers, not '{TypeName(key)}'");
                if (i < 0) i += cps.Length;
                if (i < 0 || i >= cps.Length) throw new PyException("IndexError", "string index out of range");
                return PyText.FromCodePoints([cps[i]]);
            }
            case PyTuple t:
            {
                long i = IndexValue(key, $"tuple indices must be integers or slices, not {TypeName(key)}");
                if (i < 0) i += t.Items.Length;
                if (i < 0 || i >= t.Items.Length) throw new PyException("IndexError", "tuple index out of range");
                return t.Items[i];
            }
            case System.Collections.IList l:
            {
                long i = IndexValue(key, $"list indices must be integers or slices, not {TypeName(key)}");
                if (i < 0) i += l.Count;
                if (i < 0 || i >= l.Count) throw new PyException("IndexError", "list index out of range");
                return l[(int)i];
            }
            case IEnumerable<KeyValuePair<string, object?>> d:
                foreach (var kv in d) if (key is string k && kv.Key == k) return kv.Value;
                throw new PyException("KeyError", PyText.Repr(key));
        }
        throw new PyException("TypeError", $"'{TypeName(container)}' object is not subscriptable");
    }

    static PyFunction Method(string name, Func<List<object?>, object?> body, int minArgs, int maxArgs) =>
        new(name, (args, kwargs) =>
        {
            if (kwargs.Count > 0) throw new PyException("TypeError", $"{name}() takes no keyword arguments");
            if (args.Count < minArgs || args.Count > maxArgs)
                throw new PyException("TypeError", minArgs == maxArgs ? $"{name}() takes exactly {minArgs} argument{(minArgs == 1 ? "" : "s")} ({args.Count} given)"
                    : $"{name}() takes at most {maxArgs} arguments ({args.Count} given)");
            return body(args);
        });

    static string StrArg(object? v, string fn) => v as string ?? throw new PyException("TypeError", $"{fn}() argument must be str, not {TypeName(v)}");

    /// <summary>
    /// The attributes a math expression is likely to use: numbers' real / imag / numerator / denominator / conjugate() / is_integer() / bit_length() /
    /// bit_count() / as_integer_ratio(), lists' count() / index(), and the common str methods. Null when the attribute does not exist.
    /// </summary>
    static object? GetAttribute(object? v, string attr)
    {
        if (IsInt(v))
        {
            var i = AsInt(v!);
            switch (attr)
            {
                case "real" or "numerator": return i;
                case "imag": return BigInteger.Zero;
                case "denominator": return BigInteger.One;
                case "conjugate": return Method("conjugate", _ => i, 0, 0);
                case "is_integer": return Method("is_integer", _ => true, 0, 0);
                case "bit_length": return Method("bit_length", _ => new BigInteger(BigInteger.Abs(i).IsZero ? 0 : (long)BigInteger.Abs(i).GetBitLength()), 0, 0);
                case "bit_count": return Method("bit_count", _ => { var a = BigInteger.Abs(i); long c = 0; while (!a.IsZero) { c += (long)(a & 1); a >>= 1; } return new BigInteger(c); }, 0, 0);
                case "as_integer_ratio": return Method("as_integer_ratio", _ => new PyTuple([i, BigInteger.One]), 0, 0);
            }
        }
        else if (v is double d)
        {
            switch (attr)
            {
                case "real": return d;
                case "imag": return 0.0;
                case "conjugate": return Method("conjugate", _ => d, 0, 0);
                case "is_integer": return Method("is_integer", _ => double.IsFinite(d) && Math.Floor(d) == d, 0, 0);
                case "as_integer_ratio":
                    return Method("as_integer_ratio", _ =>
                    {
                        if (double.IsNaN(d)) throw new PyException("ValueError", "cannot convert NaN to integer ratio");
                        if (double.IsInfinity(d)) throw new PyException("OverflowError", "cannot convert Infinity to integer ratio");
                        long bits = BitConverter.DoubleToInt64Bits(d);
                        int e = (int)((bits >> 52) & 0x7FF);
                        BigInteger m = bits & 0xFFFFFFFFFFFFFL;
                        if (e == 0) e = 1; else m |= BigInteger.One << 52;
                        e -= 1075;
                        if (d < 0) m = -m;
                        BigInteger num = m, den = BigInteger.One;
                        if (e > 0) num <<= e; else den <<= -e;
                        var g = BigInteger.GreatestCommonDivisor(num, den);
                        if (!g.IsZero) { num /= g; den /= g; }
                        return new PyTuple([num, den]);
                    }, 0, 0);
            }
        }
        else if (v is string s)
        {
            switch (attr)
            {
                case "upper": return Method("upper", _ => PyText.Upper(s), 0, 0);
                case "lower": return Method("lower", _ => PyText.Lower(s), 0, 0);
                case "title": return Method("title", _ => PyText.Title(s), 0, 0);
                case "capitalize": return Method("capitalize", _ => PyText.Capitalize(s), 0, 0);
                case "strip": return Method("strip", a => a.Count == 0 || a[0] is null ? PyText.Strip(s) : s.Trim(StrArg(a[0], "strip").ToCharArray()), 0, 1);
                case "lstrip": return Method("lstrip", a => a.Count == 0 || a[0] is null ? PyText.LStrip(s) : s.TrimStart(StrArg(a[0], "lstrip").ToCharArray()), 0, 1);
                case "rstrip": return Method("rstrip", a => a.Count == 0 || a[0] is null ? PyText.RStrip(s) : s.TrimEnd(StrArg(a[0], "rstrip").ToCharArray()), 0, 1);
                case "startswith": return Method("startswith", a => s.StartsWith(StrArg(a[0], "startswith"), StringComparison.Ordinal), 1, 1);
                case "endswith": return Method("endswith", a => s.EndsWith(StrArg(a[0], "endswith"), StringComparison.Ordinal), 1, 1);
                case "count": return Method("count", a => new BigInteger(CountOccurrences(s, StrArg(a[0], "count"))), 1, 1);
                case "find": return Method("find", a => new BigInteger(FindCodePoint(s, StrArg(a[0], "find"), last: false)), 1, 1);
                case "rfind": return Method("rfind", a => new BigInteger(FindCodePoint(s, StrArg(a[0], "rfind"), last: true)), 1, 1);
                case "index":
                    return Method("index", a => FindCodePoint(s, StrArg(a[0], "index"), last: false) is var r and >= 0 ? new BigInteger(r) : throw new PyException("ValueError", "substring not found"), 1, 1);
                case "replace": return Method("replace", a => PyText.Replace(s, StrArg(a[0], "replace"), StrArg(a[1], "replace")), 2, 2);
                case "zfill":
                    return Method("zfill", a =>
                    {
                        long w = IndexValue(a[0], $"'{TypeName(a[0])}' object cannot be interpreted as an integer");
                        int len = PyText.Length(s);
                        if (w <= len) return s;
                        int sign = s.Length > 0 && s[0] is '+' or '-' ? 1 : 0;
                        return s[..sign] + new string('0', (int)(w - len)) + s[sign..];
                    }, 1, 1);
                case "isdigit" or "isdecimal" or "isnumeric" or "isalpha" or "isalnum" or "isspace" or "islower" or "isupper":
                    return Method(attr, _ => StrPredicate(s, attr), 0, 0);
            }
        }
        else if (v is System.Collections.IList || v is PyTuple)
        {
            var items = v is PyTuple t ? t.Items.ToList() : ((System.Collections.IList)v).Cast<object?>().ToList();
            string kind = v is PyTuple ? "tuple" : "list";
            switch (attr)
            {
                case "count": return Method("count", a => new BigInteger(items.Count(x => ReferenceEquals(x, a[0]) || PyEquals(x, a[0]))), 1, 1);
                case "index":
                    return Method("index", a =>
                    {
                        int k = items.FindIndex(x => ReferenceEquals(x, a[0]) || PyEquals(x, a[0]));
                        return k >= 0 ? new BigInteger(k) : throw new PyException("ValueError", $"{PyText.Repr(a[0])} is not in {kind}");
                    }, 1, 1);
                case "copy" when kind == "list": return Method("copy", _ => items.ToList(), 0, 0);
            }
        }
        return null;
    }

    static int CountOccurrences(string s, string sub)
    {
        if (sub.Length == 0) return PyText.Length(s) + 1;
        int count = 0, i = 0;
        while ((i = s.IndexOf(sub, i, StringComparison.Ordinal)) >= 0) { count++; i += sub.Length; }
        return count;
    }

    /// <summary>str.find / rfind as a code point index.</summary>
    static int FindCodePoint(string s, string sub, bool last)
    {
        int i = last ? s.LastIndexOf(sub, StringComparison.Ordinal) : s.IndexOf(sub, StringComparison.Ordinal);
        if (last && sub.Length == 0) i = s.Length;
        return i < 0 ? -1 : PyText.Length(s[..i]);
    }

    static bool StrPredicate(string s, string what)
    {
        var cps = PyText.CodePoints(s);
        if (cps.Length == 0) return false;
        static UnicodeCategory Cat(int cp) => cp is >= 0xD800 and <= 0xDFFF ? UnicodeCategory.Surrogate : CharUnicodeInfo.GetUnicodeCategory(cp);
        static bool IsAlpha(int cp) => Cat(cp) is UnicodeCategory.UppercaseLetter or UnicodeCategory.LowercaseLetter or UnicodeCategory.TitlecaseLetter
            or UnicodeCategory.ModifierLetter or UnicodeCategory.OtherLetter;
        static bool IsDecimal(int cp) => Cat(cp) == UnicodeCategory.DecimalDigitNumber;
        static bool IsDigit(int cp) => IsDecimal(cp) || CharUnicodeInfo.GetDigitValue(PyText.FromCodePoints([cp]), 0) >= 0;
        static bool IsNumeric(int cp) => Cat(cp) is UnicodeCategory.DecimalDigitNumber or UnicodeCategory.LetterNumber or UnicodeCategory.OtherNumber
            || CharUnicodeInfo.GetNumericValue(PyText.FromCodePoints([cp]), 0) != -1;
        switch (what)
        {
            case "isdigit": return cps.All(IsDigit);
            case "isdecimal": return cps.All(IsDecimal);
            case "isnumeric": return cps.All(IsNumeric);
            case "isalpha": return cps.All(IsAlpha);
            case "isalnum": return cps.All(cp => IsAlpha(cp) || IsNumeric(cp));
            case "isspace": return cps.All(cp => cp < 0x10000 && PyNumber.IsSpace((char)cp));
            case "islower": return PyText.Lower(s) == s && PyText.Upper(s) != s;
            case "isupper": return PyText.Upper(s) == s && PyText.Lower(s) != s;
        }
        return false;
    }

    // ==== ComfyUI's MATH_FUNCTIONS =================================================================================

    static Dictionary<string, PyFunction> MathFunctions()
    {
        var f = new Dictionary<string, PyFunction>(StringComparer.Ordinal);
        void Add(string name, Func<List<object?>, List<(string Name, object? Value)>, object?> body) => f[name] = new PyFunction(name, body);

        static void NoKw(string name, List<(string Name, object? Value)> kw)
        {
            if (kw.Count > 0) throw new PyException("TypeError", $"{name}() takes no keyword arguments");
        }
        static void Arity(string name, List<object?> a, int min, int max)
        {
            if (a.Count < min || a.Count > max)
                throw new PyException("TypeError", min == max ? $"{name}() takes exactly {(min == 1 ? "one argument" : $"{min} arguments")} ({a.Count} given)"
                    : $"{name} expected at most {max} arguments, got {a.Count}");
        }
        static double Real(object? v, string name)
        {
            if (IsNumber(v)) return AsFloat(v);
            throw new PyException("TypeError", $"must be real number, not {TypeName(v)}");
        }

        Add("sum", (a, kw) =>
        {
            NoKw("_variadic_sum", kw);
            // nodes_math._variadic_sum: sum(values) or sum(a, b, c)
            IReadOnlyList<object?> items = a.Count == 1 && Iterable(a[0]) is { } it ? it : a;
            if (a.Count == 1 && a[0] is string) throw new PyException("TypeError", "sum() can't sum strings [use ''.join(seq) instead]");
            return Sum(items);
        });
        Add("min", (a, kw) => MinMax("min", a, kw, -1));
        Add("max", (a, kw) => MinMax("max", a, kw, 1));
        Add("abs", (a, kw) => { NoKw("abs", kw); Arity("abs", a, 1, 1); return Abs(a[0]); });
        Add("round", (a, kw) =>
        {
            object? number = a.Count > 0 ? a[0] : null, nd = a.Count > 1 ? a[1] : null;
            bool haveNumber = a.Count > 0, haveNd = a.Count > 1;
            foreach (var (k, v) in kw)
            {
                if (k == "number" && !haveNumber) { number = v; haveNumber = true; }
                else if (k == "ndigits" && !haveNd) { nd = v; haveNd = true; }
                else throw new PyException("TypeError", $"round() got an unexpected keyword argument '{k}'");
            }
            if (!haveNumber) throw new PyException("TypeError", "round() missing required argument 'number' (pos 1)");
            if (a.Count > 2) throw new PyException("TypeError", $"round() takes at most 2 arguments ({a.Count} given)");
            return Round(number, nd);
        });
        Add("pow", (a, kw) =>
        {
            // nodes_math._safe_pow(base, exp)
            object? b = a.Count > 0 ? a[0] : null, e = a.Count > 1 ? a[1] : null;
            int have = a.Count;
            if (a.Count > 2) throw new PyException("TypeError", $"_safe_pow() takes 2 positional arguments but {a.Count} were given");
            foreach (var (k, v) in kw)
            {
                if (k == "base" && a.Count < 1) { b = v; have++; }
                else if (k == "exp" && a.Count < 2) { e = v; have++; }
                else throw new PyException("TypeError", $"_safe_pow() got an unexpected keyword argument '{k}'");
            }
            if (have < 2) throw new PyException("TypeError", "_safe_pow() missing required positional argument");
            if (Greater(Abs(e), MathPowMaxExponent)) throw new PyException("ValueError", $"Exponent {PyText.Repr(e)} exceeds maximum allowed ({MathPowMaxExponent})");
            return Power(b, e);
        });
        Add("sqrt", (a, kw) =>
        {
            NoKw("sqrt", kw); Arity("sqrt", a, 1, 1);
            double x = Real(a[0], "sqrt");
            if (x < 0) throw new PyException("ValueError", "math domain error");
            return Math.Sqrt(x);
        });
        Add("ceil", (a, kw) => { NoKw("ceil", kw); Arity("ceil", a, 1, 1); return IsInt(a[0]) ? AsInt(a[0]!) : ToIntChecked(Math.Ceiling(Real(a[0], "ceil"))); });
        Add("floor", (a, kw) => { NoKw("floor", kw); Arity("floor", a, 1, 1); return IsInt(a[0]) ? AsInt(a[0]!) : ToIntChecked(Math.Floor(Real(a[0], "floor"))); });
        Add("log", (a, kw) =>
        {
            NoKw("log", kw);
            if (a.Count is < 1 or > 2) throw new PyException("TypeError", $"log expected at least 1 argument, got {a.Count}");
            double num = Log(a[0], Math.Log);
            if (a.Count == 1) return num;
            double den = Log(a[1], Math.Log);
            if (den == 0) throw new PyException("ZeroDivisionError", "float division by zero");
            return num / den;
        });
        Add("log2", (a, kw) => { NoKw("log2", kw); Arity("log2", a, 1, 1); return Log(a[0], Math.Log2); });
        Add("log10", (a, kw) => { NoKw("log10", kw); Arity("log10", a, 1, 1); return Log(a[0], Math.Log10); });
        Add("sin", (a, kw) => { NoKw("sin", kw); Arity("sin", a, 1, 1); return Trig(Real(a[0], "sin"), Math.Sin); });
        Add("cos", (a, kw) => { NoKw("cos", kw); Arity("cos", a, 1, 1); return Trig(Real(a[0], "cos"), Math.Cos); });
        Add("tan", (a, kw) => { NoKw("tan", kw); Arity("tan", a, 1, 1); return Trig(Real(a[0], "tan"), Math.Tan); });
        Add("int", (a, kw) => IntFunction(a, kw));
        Add("float", (a, kw) =>
        {
            NoKw("float", kw);
            if (a.Count > 1) throw new PyException("TypeError", $"float expected at most 1 argument, got {a.Count}");
            if (a.Count == 0) return 0.0;
            var v = a[0];
            if (v is string s) return PyNumber.TryParseFloat(s) ?? throw new PyException("ValueError", $"could not convert string to float: {PyText.Repr(s)}");
            if (IsNumber(v)) return AsFloat(v);
            throw new PyException("TypeError", $"float() argument must be a string or a real number, not '{TypeName(v)}'");
        });
        return f;
    }

    static IReadOnlyList<object?>? Iterable(object? v) => v switch
    {
        string s => PyText.CodePoints(s).Select(cp => (object?)PyText.FromCodePoints([cp])).ToList(),
        PyTuple t => t.Items,
        System.Collections.IList l => l.Cast<object?>().ToList(),
        _ => null,
    };

    /// <summary>Python 3.12+ <c>sum</c>: exact while the items are ints, then floats with Neumaier compensation.</summary>
    static object? Sum(IReadOnlyList<object?> items)
    {
        object? result = BigInteger.Zero;
        int i = 0;
        // int phase (bools count as ints)
        BigInteger iResult = BigInteger.Zero;
        while (i < items.Count && IsInt(items[i])) iResult += AsInt(items[i++]!);
        result = iResult;
        if (i == items.Count) return result;
        result = BinaryOp("+", result, items[i++]);
        if (result is not double) { for (; i < items.Count; i++) result = BinaryOp("+", result, items[i]); return result; }
        double f = (double)result, c = 0.0;
        for (; i < items.Count; i++)
        {
            var item = items[i];
            if (item is double x)
            {
                double t = f + x;
                if (Math.Abs(f) >= Math.Abs(x)) c += (f - t) + x; else c += (x - t) + f;
                f = t;
                continue;
            }
            if (IsInt(item) && AsInt(item!) is var iv && iv >= long.MinValue && iv <= long.MaxValue)
            {
                f += (double)(long)iv; // CPython adds a C long without compensation
                continue;
            }
            if (c != 0 && double.IsFinite(c)) f += c;
            c = 0;
            object? r = BinaryOp("+", f, item);
            for (i++; i < items.Count; i++) r = BinaryOp("+", r, items[i]);
            return r;
        }
        if (c != 0 && double.IsFinite(c)) f += c;
        return f;
    }

    static object? MinMax(string name, List<object?> a, List<(string Name, object? Value)> kw, int sign)
    {
        PyFunction? key = null;
        object? dflt = null;
        bool haveDefault = false;
        foreach (var (k, v) in kw)
        {
            if (k == "key") key = v is null ? null : v as PyFunction ?? throw new PyException("TypeError", $"'{TypeName(v)}' object is not callable");
            else if (k == "default") { dflt = v; haveDefault = true; }
            else throw new PyException("TypeError", $"'{k}' is an invalid keyword argument for {name}()");
        }
        if (a.Count == 0) throw new PyException("TypeError", $"{name} expected at least 1 argument, got 0");
        IReadOnlyList<object?> items;
        if (a.Count == 1)
        {
            items = Iterable(a[0]) ?? throw new PyException("TypeError", $"'{TypeName(a[0])}' object is not iterable");
            if (items.Count == 0)
                return haveDefault ? dflt : throw new PyException("ValueError", $"{name}() iterable argument is empty");
        }
        else
        {
            if (haveDefault) throw new PyException("TypeError", $"Cannot specify a default for {name}() with multiple positional arguments");
            items = a;
        }
        object? best = items[0];
        object? bestKey = key is null ? best : key.Invoke([best], []);
        for (int i = 1; i < items.Count; i++)
        {
            var k = key is null ? items[i] : key.Invoke([items[i]], []);
            // min keeps the first of equal items: replace only when strictly smaller (larger for max)
            var o = Order(k, bestKey, sign < 0 ? "<" : ">");
            if (o is { } ov && Math.Sign(ov) == sign) { best = items[i]; bestKey = k; }
        }
        return best;
    }

    static BigInteger ToIntChecked(double x)
    {
        if (double.IsNaN(x)) throw new PyException("ValueError", "cannot convert float NaN to integer");
        if (double.IsInfinity(x)) throw new PyException("OverflowError", "cannot convert float infinity to integer");
        return PyNumber.ToBigInteger(x);
    }

    static object? Round(object? number, object? ndigits)
    {
        if (ndigits is null)
        {
            if (number is double d)
            {
                if (double.IsNaN(d)) throw new PyException("ValueError", "cannot convert float NaN to integer");
                if (double.IsInfinity(d)) throw new PyException("OverflowError", "cannot convert float infinity to integer");
                return PyNumber.RoundToInt(d);
            }
            if (IsInt(number)) return AsInt(number!);
            throw new PyException("TypeError", $"type {TypeName(number)} doesn't define __round__ method");
        }
        if (!IsInt(ndigits)) throw new PyException("TypeError", $"'{TypeName(ndigits)}' object cannot be interpreted as an integer");
        var nd = AsInt(ndigits);
        if (number is double x)
        {
            long n = nd > 10000 ? 10000 : nd < -10000 ? -10000 : (long)nd;
            try { return PyNumber.Round(x, n); }
            catch (OverflowException e) { throw new PyException("OverflowError", e.Message); }
        }
        if (IsInt(number))
        {
            var v = AsInt(number!);
            if (nd.Sign >= 0) return v;
            // round half to even at 10^-ndigits
            var pow = BigInteger.Pow(10, (int)BigInteger.Min(-nd, 100000));
            var q = FloorDiv(v, pow);
            var r = v - q * pow;
            var twice = r * 2;
            if (twice > pow || (twice == pow && !q.IsEven)) q += 1;
            return q * pow;
        }
        throw new PyException("TypeError", $"type {TypeName(number)} doesn't define __round__ method");
    }

    static double Log(object? v, Func<double, double> log)
    {
        if (IsInt(v))
        {
            var i = AsInt(v!);
            if (i.Sign <= 0) throw new PyException("ValueError", "math domain error");
            // Python takes the log of a huge int exactly enough (no overflow to infinity): log(m * 2^e) = log(m) + e log(2)
            long bits = (long)i.GetBitLength();
            if (bits > 1000)
            {
                long e = bits - 60;
                double m = (double)(long)(i >> (int)e);
                return log(m) + e * log(2.0);
            }
            return log(PyNumber.ToDouble(i));
        }
        if (v is double d)
        {
            if (double.IsNaN(d)) return d;
            if (d <= 0) throw new PyException("ValueError", "math domain error");
            return log(d);
        }
        throw new PyException("TypeError", $"must be real number, not {TypeName(v)}");
    }

    static double Trig(double x, Func<double, double> f)
    {
        if (double.IsInfinity(x)) throw new PyException("ValueError", "math domain error");
        return f(x);
    }

    static object? IntFunction(List<object?> a, List<(string Name, object? Value)> kw)
    {
        object? x = a.Count > 0 ? a[0] : null;
        object? baseArg = a.Count > 1 ? a[1] : null;
        bool haveBase = a.Count > 1;
        if (a.Count > 2) throw new PyException("TypeError", $"int() takes at most 2 arguments ({a.Count} given)");
        foreach (var (k, v) in kw)
        {
            if (k == "base" && !haveBase) { baseArg = v; haveBase = true; }
            else throw new PyException("TypeError", $"'{k}' is an invalid keyword argument for int()");
        }
        if (a.Count == 0)
        {
            if (haveBase) throw new PyException("TypeError", "int() missing string argument");
            return BigInteger.Zero;
        }
        if (haveBase)
        {
            if (x is not string s) throw new PyException("TypeError", "int() can't convert non-string with explicit base");
            if (!IsInt(baseArg)) throw new PyException("TypeError", $"'{TypeName(baseArg)}' object cannot be interpreted as an integer");
            var b = AsInt(baseArg!);
            if (b != 0 && (b < 2 || b > 36)) throw new PyException("ValueError", "int() base must be >= 2 and <= 36, or 0");
            return PyNumber.TryParseInt(s, (int)b) ?? throw new PyException("ValueError", $"invalid literal for int() with base {b}: {PyText.Repr(s)}");
        }
        return x switch
        {
            bool or BigInteger => AsInt(x),
            double d => ToIntChecked(d),
            string s => PyNumber.TryParseInt(s, 10) ?? throw new PyException("ValueError", $"invalid literal for int() with base 10: {PyText.Repr(s)}"),
            _ => throw new PyException("TypeError", $"int() argument must be a string, a bytes-like object or a real number, not '{TypeName(x)}'"),
        };
    }
}
