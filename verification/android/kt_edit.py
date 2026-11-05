"""Small helpers for editing / extracting top-level Kotlin functions by name (brace matching that understands strings,
string templates, char literals and comments)."""
import re

def _skip_string(s, i):
    """s[i] == '"' (not a raw string): return the index after the closing quote, handling escapes and ${...} templates."""
    assert s[i] == '"'
    i += 1
    while i < len(s):
        c = s[i]
        if c == '\\':
            i += 2
            continue
        if c == '"':
            return i + 1
        if c == '$' and s[i + 1:i + 2] == '{':
            i = _skip_braces(s, i + 1)
            continue
        i += 1
    raise ValueError("unterminated string")

def _skip_braces(s, i):
    """s[i] == '{': return the index after the matching '}'."""
    assert s[i] == '{'
    depth = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == '"':
            if s.startswith('"""', i):
                j = s.index('"""', i + 3)
                i = j + 3
            else:
                i = _skip_string(s, i)
            continue
        if c == "'":
            # char literal: 'x' or '\n' or 'A'
            j = i + 1
            if s[j] == '\\':
                j += 2
                while s[j] != "'":
                    j += 1
            else:
                j += 1
            i = j + 1
            continue
        if c == '/' and s.startswith('//', i):
            i = s.index('\n', i) if '\n' in s[i:] else n
            continue
        if c == '/' and s.startswith('/*', i):
            i = s.index('*/', i) + 2
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unbalanced braces")

def _skip_parens(s, i):
    assert s[i] == '('
    depth = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == '"':
            i = s.index('"""', i + 3) + 3 if s.startswith('"""', i) else _skip_string(s, i)
            continue
        if c == "'":
            j = i + 1
            if s[j] == '\\':
                j += 2
                while s[j] != "'":
                    j += 1
            else:
                j += 1
            i = j + 1
            continue
        if c == '/' and s.startswith('//', i):
            i = s.index('\n', i)
            continue
        if c == '/' and s.startswith('/*', i):
            i = s.index('*/', i) + 2
            continue
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unbalanced parens")

def find_func(src, name, private=True):
    """(start, end) of `[@Composable] [private] fun name(...) [: T] { ... }` including a preceding @Composable line."""
    pat = re.compile(r'(?m)^(?:(?P<ann>@Composable[ \t]*\n))?(?P<decl>(?:private |internal )?fun ' + re.escape(name) + r'\()')
    m = pat.search(src)
    if not m:
        raise KeyError(name)
    start = m.start()
    p = src.index('(', m.start('decl') + src[m.start('decl'):].index('fun ' + name))
    p_end = _skip_parens(src, p)
    # optional return type, then the body
    j = p_end
    while src[j] in ' \t\r\n':
        j += 1
    if src[j] == ':':
        while src[j] != '{' and src[j] != '=':
            j += 1
    if src[j] == '=':
        # expression body: up to the end of line (not used for the functions we edit)
        raise ValueError("expression body")
    end = _skip_braces(src, j)
    return start, end

def get_func(src, name):
    a, b = find_func(src, name)
    return src[a:b]

def replace_func(src, name, new_text):
    a, b = find_func(src, name)
    return src[:a] + new_text.rstrip('\n') + src[b:]
