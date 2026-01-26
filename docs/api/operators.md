# Operators

The following symbolic operators are supported on layers:

| Symbol | Operator |
|--------|----------|
| + | add |
| - | subtract |
| * | multiply |
| / | division |
| // | floor division |
| % | mod |
| ^ | power |
| == | equal |
| != | not equal |
| < | less than |
| <= | less than or equal |
| > | greater than |
| >= | greater than or equal |
| & | logical and |
| \| | logical or |

On a layer you can also invoke the following operations using `layer.operator(...)` syntax:

| Operator |
|----------|
| abs |
| ceil |
| clip |
| conv2d |
| exp |
| exp2 |
| floor |
| isin |
| isnan |
| log |
| log10 |
| log2 |
| nan_to_num |

You can also call the following methods from `yirgacheffe`:

::: yirgacheffe
    handler: python
    options:
        members:
            - abs
            - all
            - any
            - ceil
            - exp
            - exp2
            - floor
            - isin
            - log
            - log10
            - log2
            - maximum
            - minimum
            - nan_to_num
            - round
            - sum
            - where
        show_root_heading: false
        show_source: false
