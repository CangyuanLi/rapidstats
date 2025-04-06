import dataclasses
import enum
from typing import Optional, Union

import polars as pl

from ._utils import _PLUGIN_PATH


class _Enum:
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def values(cls):
        return [m.value for m in cls]


class Type(_Enum, enum.Enum):
    SCIENTIFIC = "e"
    FIXED_POINT = "f"
    PERCENTAGE = "%"
    BINARY = "b"
    CHARACTER = "c"
    DECIMAL = "d"
    OCTAL = "o"
    HEX = "x"


class Separator(_Enum, enum.Enum):
    UNDERSCORE = "_"
    COMMA = ","


class Sign(_Enum, enum.Enum):
    ALL = "+"
    NEGATIVE = "-"


@dataclasses.dataclass
class FormatSpec:
    type_: Optional[str] = None
    separator: Optional[str] = None
    sign: Optional[str] = None
    precision: Optional[int] = None


def _parse_format_string(format_string):
    parts = []
    formatter = []
    current_part = []
    is_formatter = []
    i = 0

    format_string_len = len(format_string)

    while i < format_string_len:
        if format_string[i] == "{":
            if i + 1 < format_string_len and format_string[i + 1] == "{":
                current_part.append("{")
                i += 2
            else:
                if len(current_part) > 0:
                    parts.append("".join(current_part))
                    is_formatter.append(False)

                current_part = []

                formatter = []
                while i < format_string_len and format_string[i] != "}":
                    formatter.append(format_string[i])
                    i += 1

                if len(formatter) > 0:
                    parts.append("".join(c for c in formatter if c not in ("{", ":")))
                    is_formatter.append(True)

                i += 1
        elif format_string[i] == "}":
            if i + 1 < format_string_len and format_string[i + 1] == "}":
                current_part.append("}")
                i += 2
            else:
                raise ValueError("Single '}' not allowed")
        else:
            current_part.append(format_string[i])

            i += 1
    # Add the last part
    if len(current_part) > 0:
        parts.append("".join(current_part))
        is_formatter.append(False)

    return parts, is_formatter


def _parse_formatter(s: str) -> FormatSpec:
    i = 0
    length = len(s)

    spec = FormatSpec()

    if i < length and s[i] in Sign.values():
        spec.sign = s[i]
        i += 1

    if i < length and s[i] in Separator.values():
        spec.separator = s[i]
        i += 1

    if i < length and s[i] == ".":
        i += 1  # Skip the dot
        precision_chars = []
        while i < length and s[i].isdigit():
            precision_chars.append(s[i])
            i += 1

        if len(precision_chars) > 0:
            spec.precision = int("".join(precision_chars))

    # Parse type
    if i < length and (s[i] in Type.values()):
        spec.type_ = s[i]

    return spec


def _format_strnum_with_sep(expr: pl.Expr, sep: str) -> pl.Expr:
    return pl.plugins.register_plugin_function(
        plugin_path=_PLUGIN_PATH,
        function_name="_pl_format_strnum_with_sep",
        args=[expr, pl.lit(sep)],
        is_elementwise=True,
    )


def _apply_formatter(expr: pl.Expr, format_spec: FormatSpec) -> pl.Expr:
    add_percent = False

    if format_spec.type_ == Type.FIXED_POINT.value:
        expr = expr.cast(pl.Float64)
    elif format_spec.type_ == Type.PERCENTAGE.value:
        expr = expr.mul(100).cast(pl.Float64)

        add_percent = True
    elif format_spec.type_ is None:
        pass
    else:
        raise NotImplementedError(f"Type {format_spec.type_} not implemented")

    if format_spec.precision is not None:
        expr = expr.round(format_spec.precision).cast(pl.String)

        parts = expr.str.split(".")
        int_part = parts.list.get(0)
        decimal_part = parts.list.get(-1).str.pad_end(format_spec.precision, "0")

        expr = pl.concat_str(int_part, pl.lit("."), decimal_part)
    else:
        expr = expr.cast(pl.String)

    if add_percent:
        expr = pl.concat_str(expr, pl.lit("%"))

    if format_spec.separator is not None:
        expr = expr.pipe(_format_strnum_with_sep, sep=format_spec.separator)

    if format_spec.sign == Sign.ALL.value:
        expr = pl.when(expr.str.head(1).ne("-")).then(pl.concat_str(pl.lit("+"), expr))

    return expr


def format(f_string: str, *args: Union[pl.Expr, str, float]) -> pl.Expr:
    """Format expressions as a string using Python f-string syntax.

    Parameters
    ----------
    f_string : str
        A string with placeholders, mimicing Python f-string syntax. For example,
        "{:.3f}". Currently, the only supported types are "f" and "%". Width, alignment,
        and fill are also not supported.
    args
        Expression(s) that fill the placeholders. Note that strings are NOT parsed as
        columns.

    Returns
    -------
    pl.Expr

    Raises
    ------
    ValueError
        If the number of placeholders does not match the number of expressions

    Examples
    --------
    ``` py
    import polars as pl
    import rapidstats.polars as rps

    df = pl.DataFrame({"x": 1123.09873, "y": "foo"})
    df.select(
        rps.format(
            "{:,.3f} is {} is {}", pl.col("x"), pl.col("y"), "bar"
        )
    )
    ```
    ``` title="output"
    shape: (1, 1)
    ┌─────────────────────────┐
    │ x                       │
    │ ---                     │
    │ str                     │
    ╞═════════════════════════╡
    │ 1,123.099 is foo is bar │
    └─────────────────────────┘
    ```

    Added in version 0.2.0
    ----------------------
    """
    parts = _parse_format_string(f_string)
    formatters = [
        _parse_formatter(p) for p, is_formatter in zip(*parts) if is_formatter
    ]

    len_formatters = len(formatters)
    len_args = len(args)
    if len_formatters != len(args):
        raise ValueError(
            f"Number of placeholders `{len_formatters}` does not match number of arguments `{len_args}`"
        )

    outputs = [
        _apply_formatter(s if isinstance(s, pl.Expr) else pl.lit(s), format_spec)
        for s, format_spec in zip(args, formatters)
    ]

    i = 0
    to_concat = []
    for p, is_formatter in zip(*parts):
        if is_formatter:
            to_concat.append(outputs[i])
            i += 1
        else:
            to_concat.append(pl.lit(p))

    return pl.concat_str(*to_concat)
