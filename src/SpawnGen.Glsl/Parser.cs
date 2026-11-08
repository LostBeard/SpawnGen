using System.Globalization;

namespace SpawnGen.Glsl;

/// <summary>Recursive-descent parser for the GLSL ES 3.00 fragment-shader subset (no structs, arrays, matrices or interface blocks).</summary>
internal sealed class Parser
{
    readonly List<Token> _t;
    int _p;

    Parser(List<Token> tokens) => _t = tokens;

    public static ShaderAst Parse(string source)
    {
        var (tokens, directives) = Lexer.Tokenize(source);
        var parser = new Parser(tokens);
        var globals = new List<GlobalDecl>();
        var functions = new List<FunctionDecl>();
        while (parser.Peek.Kind != TokenKind.End) parser.TopLevel(globals, functions);
        return new ShaderAst(globals, functions, directives);
    }

    Token Peek => _t[_p];
    Token PeekAt(int k) => _t[Math.Min(_p + k, _t.Count - 1)];
    Token Next() => _t[_p++];

    Token Expect(string punct)
    {
        var t = Next();
        if (!t.Is(punct)) throw new GlslException($"{t} : syntax error, '{punct}' expected", t.Line);
        return t;
    }

    string ExpectIdent()
    {
        var t = Next();
        if (t.Kind != TokenKind.Ident) throw new GlslException($"{t} : syntax error, a name expected", t.Line);
        return t.Text;
    }

    bool Accept(string punct)
    {
        if (!Peek.Is(punct)) return false;
        _p++;
        return true;
    }

    static readonly HashSet<string> Precisions = ["highp", "mediump", "lowp"];

    /// <summary>A type name after its precision qualifier, if the next tokens are one.</summary>
    bool AtType(out string type, out int length)
    {
        int k = 0;
        while (PeekAt(k).Kind == TokenKind.Ident && Precisions.Contains(PeekAt(k).Text)) k++;
        var t = PeekAt(k);
        type = t.Text;
        length = k + 1;
        return t.Kind == TokenKind.Ident && GlslType.IsTypeName(t.Text);
    }

    string ParseType()
    {
        while (Peek.Kind == TokenKind.Ident && Precisions.Contains(Peek.Text)) _p++;
        var t = Next();
        if (t.Kind != TokenKind.Ident || !GlslType.IsTypeName(t.Text)) throw new GlslException($"{t} : syntax error, a type expected", t.Line);
        if (Peek.Is("[")) throw new GlslException("arrays are not supported", Peek.Line);
        return t.Text;
    }

    void TopLevel(List<GlobalDecl> globals, List<FunctionDecl> functions)
    {
        var start = Peek;
        if (Accept(";")) return;
        if (start.Kind == TokenKind.Ident && start.Text == "precision")
        {
            while (!Next().Is(";")) { }
            return;
        }
        if (start.Kind == TokenKind.Ident && start.Text == "struct") throw new GlslException("structs are not supported", start.Line);

        int? location = null;
        if (start.Kind == TokenKind.Ident && start.Text == "layout")
        {
            Next();
            Expect("(");
            while (!Peek.Is(")"))
            {
                var key = ExpectIdent();
                if (Accept("="))
                {
                    var v = Next();
                    if (key == "location") location = int.Parse(v.Text, CultureInfo.InvariantCulture);
                }
                Accept(",");
            }
            Expect(")");
        }
        string storage = "";
        while (Peek.Kind == TokenKind.Ident && Peek.Text is "uniform" or "in" or "out" or "const" or "flat" or "smooth" or "centroid" or "invariant")
        {
            var q = Next().Text;
            if (q is "uniform" or "in" or "out" or "const") storage = q;
        }
        var type = ParseType();
        var name = ExpectIdent();
        if (Peek.Is("("))
        {
            if (storage != "") throw new GlslException($"'{storage}' : qualifier not allowed on a function", start.Line);
            functions.Add(Function(type, name, start.Line));
            return;
        }
        var vars = Declarators(name, start.Line);
        globals.Add(new GlobalDecl(storage, location, type, vars, start.Line));
    }

    List<Declarator> Declarators(string firstName, int line)
    {
        var vars = new List<Declarator>();
        string name = firstName;
        while (true)
        {
            if (Peek.Is("[")) throw new GlslException("arrays are not supported", Peek.Line);
            Expr? init = Accept("=") ? Assignment() : null;
            vars.Add(new Declarator(name, init, line));
            if (!Accept(",")) break;
            name = ExpectIdent();
        }
        Expect(";");
        return vars;
    }

    FunctionDecl Function(string returnType, string name, int line)
    {
        Expect("(");
        var ps = new List<ParamDecl>();
        if (Peek.Kind == TokenKind.Ident && Peek.Text == "void" && PeekAt(1).Is(")")) Next(); // f(void)
        else if (!Peek.Is(")"))
        {
            do
            {
                string qualifier = "in";
                while (Peek.Kind == TokenKind.Ident && Peek.Text is "in" or "out" or "inout" or "const")
                {
                    var q = Next().Text;
                    if (q != "const") qualifier = q;
                }
                var pt = ParseType();
                var pn = Peek.Kind == TokenKind.Ident ? Next().Text : "";
                ps.Add(new ParamDecl(qualifier, pt, pn, line));
            } while (Accept(","));
        }
        Expect(")");
        if (Accept(";")) return new FunctionDecl(returnType, name, ps, null, line);
        return new FunctionDecl(returnType, name, ps, Block(), line);
    }

    BlockStmt Block()
    {
        var open = Expect("{");
        var body = new List<Stmt>();
        while (!Peek.Is("}"))
        {
            if (Peek.Kind == TokenKind.End) throw new GlslException("unexpected end of file, '}' expected", Peek.Line);
            body.Add(Statement());
        }
        Expect("}");
        return new BlockStmt(body, open.Line);
    }

    Stmt Statement()
    {
        var t = Peek;
        if (t.Is("{")) return Block();
        if (t.Is(";")) { Next(); return new EmptyStmt(t.Line); }
        if (t.Kind == TokenKind.Ident)
        {
            switch (t.Text)
            {
                case "if":
                {
                    Next(); Expect("(");
                    var cond = Expression();
                    Expect(")");
                    var then = Statement();
                    Stmt? els = null;
                    if (Peek.Kind == TokenKind.Ident && Peek.Text == "else") { Next(); els = Statement(); }
                    return new IfStmt(cond, then, els, t.Line);
                }
                case "for":
                {
                    Next(); Expect("(");
                    Stmt? init = null;
                    if (!Accept(";")) init = DeclarationOrExpression();
                    Expr? cond = Peek.Is(";") ? null : Expression();
                    Expect(";");
                    Expr? step = Peek.Is(")") ? null : Expression();
                    Expect(")");
                    return new ForStmt(init, cond, step, Statement(), t.Line);
                }
                case "while":
                {
                    Next(); Expect("(");
                    var cond = Expression();
                    Expect(")");
                    return new WhileStmt(cond, Statement(), t.Line);
                }
                case "do":
                {
                    Next();
                    var body = Statement();
                    if (!(Peek.Kind == TokenKind.Ident && Peek.Text == "while")) throw new GlslException($"{Peek} : 'while' expected", Peek.Line);
                    Next(); Expect("(");
                    var cond = Expression();
                    Expect(")"); Expect(";");
                    return new DoWhileStmt(body, cond, t.Line);
                }
                case "break" or "continue" or "discard":
                    Next(); Expect(";");
                    return new JumpStmt(t.Text, t.Line);
                case "return":
                {
                    Next();
                    Expr? value = Peek.Is(";") ? null : Expression();
                    Expect(";");
                    return new ReturnStmt(value, t.Line);
                }
                case "switch":
                    throw new GlslException("switch is not supported", t.Line);
            }
        }
        return DeclarationOrExpression();
    }

    /// <summary>A local declaration (<c>const</c>? type name [= init], ...;) or an expression statement.</summary>
    Stmt DeclarationOrExpression()
    {
        var t = Peek;
        bool isConst = false;
        if (t.Kind == TokenKind.Ident && t.Text == "const") { Next(); isConst = true; }
        if (isConst || (AtType(out _, out int len) && PeekAt(len).Kind == TokenKind.Ident))
        {
            var type = ParseType();
            var name = ExpectIdent();
            return new VarDeclStmt(type, isConst, Declarators(name, t.Line), t.Line);
        }
        var e = Expression();
        Expect(";");
        return new ExprStmt(e, t.Line);
    }

    Expr Expression()
    {
        var e = Assignment();
        if (Peek.Is(",")) throw new GlslException("the comma operator is not supported", Peek.Line);
        return e;
    }

    static readonly HashSet<string> AssignOps = ["=", "+=", "-=", "*=", "/=", "%=", "<<=", ">>=", "&=", "|=", "^="];

    Expr Assignment()
    {
        var left = Conditional();
        if (Peek.Kind == TokenKind.Punct && AssignOps.Contains(Peek.Text))
        {
            var op = Next();
            return new AssignExpr(op.Text, left, Assignment(), op.Line);
        }
        return left;
    }

    Expr Conditional()
    {
        var cond = Binary(0);
        if (!Peek.Is("?")) return cond;
        var q = Next();
        var then = Expression();
        Expect(":");
        var els = Assignment();
        return new TernaryExpr(cond, then, els, q.Line);
    }

    // GLSL's binary precedence, loosest first.
    static readonly string[][] Levels =
    [
        ["||"], ["^^"], ["&&"], ["|"], ["^"], ["&"], ["==", "!="], ["<", ">", "<=", ">="], ["<<", ">>"], ["+", "-"], ["*", "/", "%"],
    ];

    Expr Binary(int level)
    {
        if (level == Levels.Length) return Unary();
        var left = Binary(level + 1);
        while (Peek.Kind == TokenKind.Punct && Levels[level].Contains(Peek.Text))
        {
            var op = Next();
            left = new BinaryExpr(op.Text, left, Binary(level + 1), op.Line);
        }
        return left;
    }

    Expr Unary()
    {
        var t = Peek;
        if (t.Kind == TokenKind.Punct && t.Text is "-" or "+" or "!" or "~" or "++" or "--")
        {
            Next();
            return new UnaryExpr(t.Text, Unary(), t.Line);
        }
        return Postfix(Primary());
    }

    Expr Postfix(Expr e)
    {
        while (true)
        {
            var t = Peek;
            if (t.Is(".")) { Next(); e = new FieldExpr(e, ExpectIdent(), t.Line); }
            else if (t.Is("[")) { Next(); var index = Expression(); Expect("]"); e = new IndexExpr(e, index, t.Line); }
            else if (t.Is("++") || t.Is("--")) { Next(); e = new PostfixExpr(t.Text, e, t.Line); }
            else return e;
        }
    }

    Expr Primary()
    {
        var t = Next();
        switch (t.Kind)
        {
            case TokenKind.FloatLit:
                return new LiteralExpr(GlslType.Float, float.Parse(t.Text, CultureInfo.InvariantCulture), t.Line);
            case TokenKind.IntLit:
                return new LiteralExpr(GlslType.Int, unchecked((int)uint.Parse(t.Text, CultureInfo.InvariantCulture)), t.Line);
            case TokenKind.UIntLit:
                return new LiteralExpr(GlslType.UInt, uint.Parse(t.Text, CultureInfo.InvariantCulture), t.Line);
            case TokenKind.Ident when t.Text is "true" or "false":
                return new LiteralExpr(GlslType.Bool, t.Text == "true", t.Line);
            case TokenKind.Ident:
                if (Peek.Is("("))
                {
                    Next();
                    var args = new List<Expr>();
                    if (!Peek.Is(")"))
                    {
                        if (Peek.Kind == TokenKind.Ident && Peek.Text == "void" && PeekAt(1).Is(")")) Next();
                        else do args.Add(Assignment()); while (Accept(","));
                    }
                    Expect(")");
                    return new CallExpr(t.Text, args, t.Line);
                }
                return new NameExpr(t.Text, t.Line);
            case TokenKind.Punct when t.Text == "(":
            {
                var e = Expression();
                Expect(")");
                return e;
            }
            default:
                throw new GlslException($"{t} : syntax error", t.Line);
        }
    }
}
