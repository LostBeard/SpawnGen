# Generates tests/SpawnGen.Nodes.Tests/PythonVectors.g.cs: expected values computed by CPython itself
# (str methods, format, repr, re, json) and by simpleeval 1.0.8 with ComfyUI's MATH_FUNCTIONS.
# A development tool only (SpawnGen runs no Python). Usage, from the repository root, with the simpleeval 1.0.8 source unpacked somewhere:
#   python -I -X utf8 tools/python-vectors/gen_vectors.py tests/SpawnGen.Nodes.Tests/PythonVectors.g.cs <folder holding simpleeval.py>
import sys, re, json, math, string, platform
sys.path.insert(0, sys.argv[2])  # simpleeval source folder
from simpleeval import simple_eval

BS = chr(92)

def cs(s):
    if s is None:
        return 'null'
    out = ['"']
    for ch in s:
        o = ord(ch)
        if ch == '"': out.append(BS + '"')
        elif ch == BS: out.append(BS + BS)
        elif ch == '\n': out.append(BS + 'n')
        elif ch == '\r': out.append(BS + 'r')
        elif ch == '\t': out.append(BS + 't')
        elif 0x20 <= o < 0x7f: out.append(ch)
        elif o > 0xFFFF:
            o -= 0x10000
            out.append(BS + 'u%04X' % (0xD800 + (o >> 10)) + BS + 'u%04X' % (0xDC00 + (o & 0x3FF)))
        else: out.append(BS + 'u%04X' % o)
    out.append('"')
    return ''.join(out)

def val(v):
    # (kind, text): i int, f float (repr), s str, b bool, n None
    if v is None: return '"n", ""'
    if isinstance(v, bool): return '"b", ' + cs(str(v))
    if isinstance(v, int): return '"i", ' + cs(str(v))
    if isinstance(v, float): return '"f", ' + cs(repr(v))
    return '"s", ' + cs(v)

def err(e):
    return 'ERR:' + type(e).__name__

lines = []
def emit(s): lines.append(s)

# ---------------- format(value, spec) ----------------
fmt_cases = []
values = [0, 1, -1, 7, -42, 255, 1234, 1234567, -9876543210, 10**20, True, False,
          0.0, -0.0, 1.0, -1.5, 0.1, 0.5, 2.5, 1/3, 2/3, 3.14159265358979, 123.456, 1234.5678, 1e16, 1.5e-7, 1e-5, 1e-4, 0.00012345, 123456789.0,
          2.675, 0.125, 1e100, 5e-324, 1.7976931348623157e308, float('inf'), float('-inf'), float('nan'),
          'abc', '', 'héllo', None]
specs = ['', '5', '<6', '>6', '^7', '*^7', '0>5', '05', '010', '+', ' ', '-', '=+8', '08', '+08', ',', '_', '010,', '09,', '08,',
         'd', 'x', 'X', '#x', '#X', 'o', '#o', 'b', '#b', '_b', '_x', 'c', 'n', 'e', 'E', '.2e', '.0e', '#.0e', 'f', '.2f', '.0f', '#.0f', ',.2f', '_.3f',
         'g', 'G', '.3g', '.1g', '#g', '#.3g', '.10g', '%', '.1%', '.3', '.0', '.1', '.12', 'z.1f', 'z', 's', '.2s', '>5s', '=5', '#', '.2', 'r']
for v in values:
    for sp in specs:
        try: r = format(v, sp)
        except Exception as e: r = err(e)
        fmt_cases.append((v, sp, r))
emit('    public static readonly (string Kind, string Value, string Spec, string Expected)[] Format =')
emit('    [')
for v, sp, r in fmt_cases:
    emit('        (%s, %s, %s),' % (val(v), cs(sp), cs(r)))
emit('    ];')

# ---------------- str.format(**values) ----------------
sf_values = {'a': 3, 'b': 2.5, 'c': 'xyz', 'd': True, 'e': None, 'f': -0.25, 'g': 'hé'}
sf_templates = ['{a}', '{a} + {b} = {c}', '{a:05}', '{b:.3f}', '{c!r}', '{g!a}', '{c:>{a}}', '{{a}}', '{a.real}', '{c[1]}', '{b!s:>8}', '{d}', '{e}', '{d:d}',
                '{f:+.1%}', '{a:{c}}', '{x}', '{}', '{0}', '}', '{a', '{a!z}', '{a:%}', '{c:d}', '{e:>4}', '{a:,}', '{a:_b}', '{b:e}',
                '{b:{a}.{a}}', '{c[0]}{c[2]}', '{c[5]}', '{a[0]}', '{c!r:>8}', '{g!r}', '{d!r}', '{e!s}', '{b!r}', '{a:x<4}', '{f:z.0f}', '{c:^9}', 'x{{y}}z', '{a}}}',
                '{ a}', '{a.b}', '{c:.1}', '{a:c}', '{a:=^7}', '{b:010.2f}']
emit('    public static readonly (string Template, string Expected)[] StrFormat =')
emit('    [')
for t in sf_templates:
    try: r = t.format(**sf_values)
    except Exception as e: r = err(e)
    emit('        (%s, %s),' % (cs(t), cs(r)))
emit('    ];')

# ---------------- repr(float) ----------------
floats = [0.0, -0.0, 1.0, 0.1, 0.2, 0.3, 1/3, 1e16, 1e15, 9999999999999998.0, 1e-4, 1e-5, 1.5e-5, 123456789012345680.0, 1.7976931348623157e308, 5e-324,
          2.2250738585072014e-308, 3.141592653589793, 100.0, 1e22, 1e21, 0.5, 4.35, 1e-7, 12345.678]
emit('    public static readonly (string Bits, string Repr)[] FloatRepr =')
emit('    [')
for f in floats:
    emit('        ("%s", %s),' % (f.hex(), cs(repr(f))))
emit('    ];')

# ---------------- str methods (ComfyUI's text nodes) ----------------
texts = ['hello World', 'the quick BROWN fox\'s 3rd', 'ßa straße', 'ﬁx ﬂy', 'ǆemal', 'ΑΣ ΑΣΑ Σ', 'ODYΣΣEYΣ.',
         '  \t\n x y \x1c\x1f\x85\xa0　', 'İstanbul', 'i̇', 'ᾀ ᾳ', 'a\U0001F600b\U0001F601c', '', '123abc', 'mIxEd-CaSe_words', 'café NAÏVE',
         'ªa ʰb', 'o\'neil mc\'donald', 'xͅy']
emit('    public static readonly (string Text, string Upper, string Lower, string Capitalize, string Title, string Strip, string LStrip, string RStrip, int Length)[] StrMethods =')
emit('    [')
for t in texts:
    emit('        (%s, %s, %s, %s, %s, %s, %s, %s, %d),' % tuple([cs(t), cs(t.upper()), cs(t.lower()), cs(t.capitalize()), cs(t.title()), cs(t.strip()), cs(t.lstrip()), cs(t.rstrip()), len(t)]))
emit('    ];')

emit('    public static readonly (string Text, long Start, long End, string Expected)[] Substring =')
emit('    [')
for t in ['hello world', 'a\U0001F600b\U0001F601c', '']:
    for s, e in [(0, 5), (-5, 11), (3, 100), (4, 2), (-100, 2), (1, -1), (0, 0), (2, 3)]:
        emit('        (%s, %d, %d, %s),' % (cs(t), s, e, cs(t[s:e])))
emit('    ];')

emit('    public static readonly (string Text, string Find, string Replace, string Expected)[] Replace =')
emit('    [')
for t, f, r in [('a-b-c', '-', '+'), ('abc', '', '-'), ('a\U0001F600', '', '|'), ('', '', 'x'), ('aaa', 'aa', 'b'), ('x$1y', '$1', '\\1')]:
    emit('        (%s, %s, %s, %s),' % (cs(t), cs(f), cs(r), cs(t.replace(f, r))))
emit('    ];')

# ---------------- regex (ComfyUI's RegexMatch / RegexExtract / RegexReplace bodies) ----------------
def flags_of(ci, ml, da):
    fl = 0
    if ci: fl |= re.IGNORECASE
    if ml: fl |= re.MULTILINE
    if da: fl |= re.DOTALL
    return fl

def regex_match(string, pattern, ci, ml, da):
    try:
        return re.search(pattern, string, flags_of(ci, ml, da)) is not None
    except re.error:
        return False

def regex_extract(string, regex_pattern, mode, case_insensitive, multiline, dotall, group_index):
    join_delimiter = "\n"
    flags = flags_of(case_insensitive, multiline, dotall)
    try:
        if mode == "First Match":
            match = re.search(regex_pattern, string, flags)
            result = match.group(0) if match else ""
        elif mode == "All Matches":
            matches = re.findall(regex_pattern, string, flags)
            if matches:
                if isinstance(matches[0], tuple):
                    result = join_delimiter.join([m[0] for m in matches])
                else:
                    result = join_delimiter.join(matches)
            else:
                result = ""
        elif mode == "First Group":
            match = re.search(regex_pattern, string, flags)
            if match and len(match.groups()) >= group_index:
                result = match.group(group_index)
            else:
                result = ""
        elif mode == "All Groups":
            matches = re.finditer(regex_pattern, string, flags)
            results = []
            for match in matches:
                if match.groups() and len(match.groups()) >= group_index:
                    results.append(match.group(group_index))
            result = join_delimiter.join(results)
        else:
            result = ""
    except re.error:
        result = ""
    return result

match_cases = [
    ('abc123', r'\d+', True, False, False), ('ABC', 'abc', False, False, False), ('ABC', 'abc', True, False, False),
    ('a\nb', '^b$', False, True, False), ('a\nb', '^b$', False, False, False), ('a\nb', 'a.b', False, False, True), ('a\nb', 'a.b', False, False, False),
    ('abc\n', r'c\Z', False, False, False), ('abc\n', r'c$', False, False, False), ('abc', r'(?P<x>b)(?P=x)', False, False, False), ('abbc', r'(?P<x>b)(?P=x)', False, False, False),
    ('abc', '(', False, False, False), ('abc', r'(?<n>b)', False, False, False), ('aaa', r'a{,2}', False, False, False), ('a{,2}', r'x{,2}', False, False, False),
    ('aaab', r'a*+b', False, False, False), ('aaa', r'a*+a', False, False, False), ('x\x1cy', r'x\sy', False, False, False), ('x_y', r'^\w+$', False, False, False),
    ('été', r'^\w+$', False, False, False), ('été', r'(?a)^\w+$', False, False, False), ('A1', r'(?i)a\d', False, False, False),
    ('ab', r'a(?#comment)b', False, False, False), ('a b', r'(?x) a \  b # comment', False, False, False), ('tab\there', r'\t', False, False, False),
    ('a', r'\e', False, False, False), ('a', r'\p{L}', False, False, False), ('$5', r'\$\d', False, False, False), ('a]b', r'[]]', False, False, False),
    ('a-b', r'[a\-z]-', False, False, False), ('ß', r'SS', True, False, False), ('K', 'K', True, False, False), ('aA', r'(?-i:a)A', True, False, False),
    ('ab', r'(a)(?(1)b|c)', False, False, False), ('x', r'\101', False, False, False), ('A', r'\101', False, False, False), ('a\x00', r'a\0', False, False, False),
    ('A', r'\x41', False, False, False), ('a 1', r'[\d\s]{2}', False, False, False), ('_', r'[^\W_]', False, False, False), ('aaa', r'^a{2,3}?$', False, False, False),
    ('a\nb', r'a(?s:.)b', False, False, False), ('ab', r'a(?i)b', False, False, False), ('xxx', r'x{3,2}', False, False, False), ('abc', r'c\z', False, False, False),
    ('\U0001F600\U0001F600', '^\U0001F600+$', False, False, False), ('aa', r'(a)\1', False, False, False), ('aa', r'(?P<n>a)\g<n>', False, False, False),
    ('ab', r'(?P<n>a)?(?(n)b|c)', False, False, False), ('c', r'(?P<n>a)?(?(n)b|c)', False, False, False), ('cat dog', r'\bdog\b', False, False, False),
    ('xdog', r'(?<=x)dog', False, False, False), ('a', r'a{1', False, False, False), ('a{1', r'a{1', False, False, False), ('ab', r'a|*b', False, False, False),
    ('a', r'a**', False, False, False), ('aB', r'[a-z]B', True, False, False), ('É', r'é', True, False, False), ('x', r'[\U0001F600]', False, False, False),
    ('a.b', r'a\.b', False, False, False), ('a+b', r'a\+b', False, False, False), ('end\n', r'end\Z', False, False, False), ('end', r'end\Z', False, False, False),
    ('A\nB', r'(?m)^B', False, False, False), ('ab', r'(?:a)(?=b)', False, False, False), ('ab', r'a(?!b)', False, False, False), ('x9', r'\d\Z', False, False, False),
]
emit('    public static readonly (string Text, string Pattern, bool IgnoreCase, bool Multiline, bool DotAll, bool Expected)[] RegexMatch =')
emit('    [')
for s, p, ci, ml, da in match_cases:
    emit('        (%s, %s, %s, %s, %s, %s),' % (cs(s), cs(p), str(ci).lower(), str(ml).lower(), str(da).lower(), str(regex_match(s, p, ci, ml, da)).lower()))
emit('    ];')

extract_cases = [
    ('a1b22c333', r'\d+'), ('John Smith, Jane Doe', r'(\w+) (\w+)'), ('k=v; x=y', r'(?P<key>\w)=(?P<val>\w)'), ('abc', r'x*'), ('abc', r'(x)?b'),
    ('no digits', r'\d'), ('Price: $12.50 and $3', r'\$(\d+)(?:\.(\d+))?'), ('aaa', '('), ('line1\nline2', r'^line(\d)$'), ('Hello', r'hello'),
    ('ab12', r'(?P<letters>[a-z]+)(\d+)'), ('x=1,y=22', r'(\w)=(?P<v>\d+)'), ('aXbX', r'(?i)x'), ('', r'^'), ('a\U0001F600b', r'\U0001F600'),
]
emit('    public static readonly (string Text, string Pattern, string Mode, long GroupIndex, string Expected)[] RegexExtract =')
emit('    [')
for s, p in extract_cases:
    for mode in ["First Match", "All Matches", "First Group", "All Groups"]:
        for gi in ([0, 1, 2] if 'Group' in mode else [1]):
            try: r = regex_extract(s, p, mode, True, True if '\n' in s else False, False, gi)
            except Exception as e: r = err(e)
            if r is None: r = 'NONE'
            emit('        (%s, %s, %s, %d, %s),' % (cs(s), cs(p), cs(mode), gi, cs(r)))
emit('    ];')

sub_cases = [
    ('a1b22', r'\d+', '#', 0), ('a1b22', r'\d+', '#', 1), ('John Smith', r'(\w+) (\w+)', r'\2, \1', 0), ('John Smith', r'(?P<f>\w+) (?P<l>\w+)', r'\g<l> \g<f>', 0),
    ('abc', r'b', r'$0 $1 \g<0>', 0), ('abc', r'b', r'[\n]', 0), ('abc', r'b', r'\&', 0), ('abxd', r'x*', '-', 0), ('ab', r'(a)|b', r'[\1]', 0), ('abc', r'(b)', r'\2', 0),
    ('abc', r'b', r'\q', 0), ('aaa', r'a', r'\141', 0), ('abc', r'(b)', r'\10', 0), ('Hello hello', r'hello', 'X', 0), ('a.b.c', r'\.', '', 2), ('abc', r'', '-', 0),
    ('x', r'(?P<n>x)', r'\g<n>\g<1>', 0), ('price 5', r'(\d)', r'$\1', 0),
    ('ab', r'(?P<a>a)(b)', r'\2\1', 0), ('ab', r'(a)(?P<b>b)', r'\g<b>-\g<2>', 0), ('aaa', r'a', r'\0', 0), ('aaa', r'a', r'\g<x>', 0), ('a', r'a', '\\', 0),
    ('a', r'a', r'\g<-1>', 0), ('tab', r'a', r'\t', 0), ('a\nb', r'$', 'E', 0), ('a\nb\n', r'$', 'E', 0), ('aaa', r'a*?', '-', 0), ('abc', r'(?=b)', '^', 0),
    ('AbC', r'[a-z]', '*', 0), ('x y  z', r'\s+', '_', 0), ('2024-01-05', r'(\d+)-(\d+)-(\d+)', r'\3/\2/\1', 0),
]
emit('    public static readonly (string Text, string Pattern, string Replace, int Count, string Expected)[] RegexSub =')
emit('    [')
for s, p, r, c in sub_cases:
    try: out = re.sub(p, r, s, count=c, flags=re.IGNORECASE)
    except Exception as e: out = err(e)
    emit('        (%s, %s, %s, %d, %s),' % (cs(s), cs(p), cs(r), c, cs(out)))
emit('    ];')

# ---------------- json ----------------
def json_extract(json_string, key):
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", json_string):
        try:
            data, _ = decoder.raw_decode(json_string, match.start())
        except json.JSONDecodeError:
            continue
        value = data.get(key)
        return "" if value is None else str(value)
    return ""

json_cases = [
    ('{"a": "x", "b": 2}', 'a'), ('{"a": "x", "b": 2}', 'b'), ('{"a": 1.0}', 'a'), ('{"a": 1e400}', 'a'), ('{"a": [1, "two", null, true, 2.5]}', 'a'),
    ('{"a": {"k": "it\'s"}}', 'a'), ('Here: ```json\n{"name": "cat", "n": 3}\n```', 'name'), ('{bad} {"a": 5}', 'a'), ('{"a": null}', 'a'), ('{"a": 1}', 'missing'),
    ('{"a": 1, "a": 2}', 'a'), ('no json', 'a'), ('{"a": "\\u00e9\\ud83d\\ude00"}', 'a'), ('{"a": 12345678901234567890123}', 'a'), ('{"a": NaN, "b": -Infinity}', 'a'),
    ('{"a": -0.0}', 'a'), ('{"a": 0.1}', 'a'), ('{"outer": {"inner": 1}} trailing', 'outer'), ('{"a": "x\\ty"}', 'a'), ('{"a":\n "line\\nbreak"}', 'a'), ('{"a": [] , "b": {}}', 'b'),
]
emit('    public static readonly (string Json, string Key, string Expected)[] JsonExtract =')
emit('    [')
for j, k in json_cases:
    emit('        (%s, %s, %s),' % (cs(j), cs(k), cs(json_extract(j, k))))
emit('    ];')

dump_cases = ['{"a": 1, "b": [1, 2.5, "x"], "c": {"d": null, "e": true}}', '{}', '{"e": []}', '{"s": "quote\\" back\\\\ tab\\t nl\\n ctl\\u0001 \\u00e9 \\u2028"}',
              '{"f": 1e16, "g": 1e-7, "h": 100.0, "i": NaN, "j": Infinity}', '[1, [2, [3, []]], {}]']
emit('    public static readonly (string Json, int Indent, string Expected)[] JsonDump =')
emit('    [')
for d in dump_cases:
    data = json.loads(d)
    for ind in [0, 1, 2, 4]:
        emit('        (%s, %d, %s),' % (cs(d), ind, cs(json.dumps(data, ensure_ascii=False, indent=ind or None))))
emit('    ];')

save_cases = ['{"b": 1, "a": [1, 2]}', 'not json', '  [1, 2]  ', '{"x": 1} extra']
emit('    public static readonly (string Text, string Expected)[] SaveTextJson =')
emit('    [')
for t in save_cases:
    try:
        data = json.loads(t)
        out = json.dumps(data, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        out = t
    emit('        (%s, %s),' % (cs(t), cs(out)))
emit('    ];')

# ---------------- number convert ----------------
def number_convert(value):
    if isinstance(value, bool):
        float_val = 1.0 if value else 0.0
        int_val = 1 if value else 0
    elif isinstance(value, int):
        float_val = float(value)
        int_val = value
    elif isinstance(value, float):
        float_val = value
        int_val = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Cannot convert empty string to number.")
        try:
            float_val = float(text)
        except ValueError:
            raise ValueError(f"Cannot convert string to number: {value!r}") from None
        if not math.isfinite(float_val):
            raise ValueError(f"Cannot convert non-finite value to number: {float_val}")
        try:
            int_val = int(text)
        except ValueError:
            int_val = int(float_val)
    else:
        raise TypeError(f"Unsupported input type: {type(value).__name__}")
    if not math.isfinite(float_val):
        raise ValueError(f"Cannot convert non-finite value to number: {float_val}")
    return float_val, int_val

nc_values = [True, False, 0, 7, -3, 2**53 + 1, 1.0, 2.9, -2.9, 1e18, float('inf'), float('nan'), '42', '  3.75 ', '-0.5', '1e3', '1_000', '0x10', '', '  ', 'abc', 'inf', 'nan',
             '١٢', '+5', '.5', '5.', '1__0', '9' * 25, '12abc', '1.5e-3']
emit('    public static readonly (string Kind, string Value, string Float, string Int)[] NumberConvert =')
emit('    [')
for v in nc_values:
    try:
        f, i = number_convert(v)
        fr, ir = repr(f), str(i)
        if not (-2**63 <= i < 2**63): fr, ir = 'ERR:Overflow', 'ERR:Overflow'
    except Exception as e:
        fr, ir = err(e), err(e)
    emit('        (%s, %s, %s),' % (val(v), cs(fr), cs(ir)))
emit('    ];')

# ---------------- math expression (simpleeval + ComfyUI's MATH_FUNCTIONS) ----------------
MAX_EXPONENT = 4000
def _variadic_sum(*args):
    if len(args) == 1 and hasattr(args[0], "__iter__"):
        return sum(args[0])
    return sum(args)
def _safe_pow(base, exp):
    if abs(exp) > MAX_EXPONENT:
        raise ValueError(f"Exponent {exp} exceeds maximum allowed ({MAX_EXPONENT})")
    return pow(base, exp)
MATH_FUNCTIONS = {"sum": _variadic_sum, "min": min, "max": max, "abs": abs, "round": round, "pow": _safe_pow, "sqrt": math.sqrt, "ceil": math.ceil,
                  "floor": math.floor, "log": math.log, "log2": math.log2, "log10": math.log10, "sin": math.sin, "cos": math.cos, "tan": math.tan, "int": int, "float": float}

def math_expr(expression, values):
    if not expression.strip():
        raise ValueError("Expression cannot be empty.")
    context = dict(values)
    context["values"] = list(values.values())
    result = simple_eval(expression, names=context, functions=MATH_FUNCTIONS)
    if not isinstance(result, (int, float)):
        raise ValueError("not numeric")
    try:
        float_result = float(result)
    except OverflowError:
        raise ValueError("too large") from None
    if not math.isfinite(float_result):
        raise ValueError("non-finite")
    return float_result, int(result), bool(result)

V = {'a': 7, 'b': 2, 'c': 2.5, 'd': True, 'e': 0, 'f': -3.75}
exprs = ['a + b', 'a - b * c', 'a / b', 'a // b', '-a // b', 'a % b', '-a % b', 'a % -b', 'f // 1', 'f % 1', 'c // 0.5', '-7.5 % 2', 'a ** b', 'b ** -1', '2 ** 0.5',
         '2 ** 100', '(-8) ** (1/3)', '0 ** -1', '1 / 0', '1 // 0', '1 % 0', '1.0 / 0', 'a > b', 'a < b < c', 'b < a < c', 'a == 7.0', 'a != b', 'not e', 'e or c', 'a and e', 'e and a',
         'a if d else b', 'b if e else c', 'min(a, b, c)', 'max(values)', 'min(values)', 'sum(values)', 'sum(a, b, c)', 'sum(values[:2])', 'abs(f)', 'round(c)', 'round(3.5)', 'round(-2.5)',
         'round(f, 1)', 'round(2.675, 2)', 'round(1234, -2)', 'round(1250, -2)', 'round(a / b)', 'pow(2, 10)', 'pow(2, 5000)', 'pow(b, -2)', 'sqrt(16)', 'sqrt(-1)', 'ceil(c)', 'floor(f)',
         'ceil(f)', 'log(100, 10)', 'log(8, 2)', 'log2(8)', 'log10(1000)', 'log(0)', 'sin(0)', 'cos(0)', 'tan(0)', 'int(f)', 'int("12")', 'int("ff", 16)', 'float("1.5")', 'float("inf")',
         'int(c) + float(a)', 'values[0]', 'values[-1]', 'values[10]', 'values[1:3][0]', 'len(values)', 'd + d', '-d', '~a', 'a << 3', 'a >> 1', 'a & 3', 'a | 8', 'a ^ 1', 'd & True',
         'a.real', 'c.is_integer()', 'f.as_integer_ratio()[1]', 'a.bit_length()', '"abc".count("b")', 'float(f"{a}{b}")', 'float(f"{c:.1f}")', 'int("1_000")', 'x + 1', 'foo(1)',
         'a = 5', 'a; b', '(1, 2)', '[1, 2]', 'lambda: 1', 'values.__len__()', '"a" * 3', '"ab" + "cd"', '1e308 * 10', '10.0 ** 400', '5000000 ** 1', '2 ** 4000001', '1 << 10001',
         'True', 'None', '"5"', '', '   ', 'a +', '1 +\n 2', 'a +\\\n b', '# comment\na', 'max(values, key=abs)', 'min([])', 'max(f, e)', 'min(e, False)', 'abs(-d)', 'sum([])',
         'sum(1)', 'int(2.9) + int(-2.9)', 'float(10**20)', '10**20', '10**30', '0.1 + 0.2', 'a * 1.0', '7 // 2.0', '1 if a > 5 else 0', 'a in values', 'c not in values', 'e is 0',
         'round(0.5) + round(1.5)', 'max(1, 1.0)', 'min(1.0, 1)', '2 ** -1074', 'b ** 0.5 * b ** 0.5', 'sum(values) / len(values)', '(a + b) * (c - f) / 2',
         'float(f"{c:{a}.2f}")', 'int("abc"[::-1] == "cba")', 'sum(values[::2])', 'sum(values[1:])', 'sum(values[::-1][:2])', '1 < a <= 7 != b', 'a if b > 1 else c',
         'round(-0.4, 0)', 'max([1, 2])', 'sum(values, 1)', 'min("abc")', 'int(True)', 'float(d)', 'abs(-2**63)', '2**63', '-2**63', 'ceil(1e300)', '"%d" % a',
         'a == "7"', 'a < "7"', 'values.count(7)', 'values.index(2)', 'c.real', '(2.5).as_integer_ratio()[0]', '0x1F + 0o17 + 0b11', '1_000 * 2', '1e3', '.5 + 5.',
         '012', '0_0', '1__0', 'float("a" "1")', 'int("10", base=2)', 'round(number=2.5)', 'pow(exp=3, base=2)', 'log(8, base=2)', 'x := 1', 'a if a else',
         'int(" 42 ")', 'float(" -1.5e2 ")', 'int(3.99e2)', 'round(2.5) == 2', 'round(1e20)', 'round(c, -1)', 'round(25, -1)', 'round(35, -1)', 'round(-25, -1)',
         'a // -b', '-a // -b', '7.5 // -2', '-7.5 % -2', '5 % 2.5', '0.0 ** 0', 'f ** 2', 'f ** 0.5', 'b ** b ** 2', '-b ** 2', '(-b) ** 2', 'not a == b', 'a > b == True',
         'abs(f) // 1 + c % 1', 'min(values, default=5)', 'max(f, key=abs)', 'max(1, 2, key=lambda x: -x)', 'sqrt(2) ** 2', 'sin(3.141592653589793)', 'log(1e308 * 10)',
         'float("nan") == float("nan")', 'int("1e3")', 'int(float("1e3"))', 'int("0x1F", 16)', 'int("0x1F", 0)', 'int("08", 0)', 'int("z", 36)', 'f"{a}" == "7"',
         'float(f"{a!r}")', 'float(f"{c=}"[2:])', 'float(f"{b:>5}")', '"%s"', 'a.imag + c.imag', 'values[True]', 'values[1.0]', '"abc"[1]', 'values[-7]', 'values[5:1:-2][0]',
         'd + 0.5', 'True / 2', 'True // 2', '2 ** True', '~True', 'a and b and c', 'e or 0 or 0.0', 'None or 3']
emit('    public static readonly (string Expression, string Float, string Int, string Bool)[] MathExpression =')
emit('    [')
for ex in exprs:
    try:
        f, i, b = math_expr(ex, V)
        r = (repr(f), str(i), str(b))
        if not (-2**63 <= i < 2**63): r = ('ERR:Overflow',) * 3
    except Exception as e:
        r = (err(e),) * 3
    emit('        (%s, %s, %s, %s),' % (cs(ex), cs(r[0]), cs(r[1]), cs(r[2])))
emit('    ];')

header = [
    '// <auto-generated>',
    '// Expected values computed by CPython %s itself (str methods, format, repr, re, json) and by simpleeval 1.0.8 with ComfyUI\'s MATH_FUNCTIONS' % platform.python_version(),
    '// (comfy_extras/nodes_math.py), plus the bodies of ComfyUI\'s RegexExtract / JsonExtractString / NumberConvert execute methods.',
    '// Regenerate with tools/python-vectors/gen_vectors.py; do not edit by hand.',
    '// </auto-generated>',
    'namespace SpawnGen.Nodes.Tests;',
    '',
    'static class PythonVectors',
    '{',
]
open(sys.argv[1], 'w', encoding='utf-8', newline='\n').write('\n'.join(header + lines + ['}', '']))
print('cases:', len(lines))
