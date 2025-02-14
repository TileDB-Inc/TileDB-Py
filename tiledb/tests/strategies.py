from hypothesis import strategies as st
from hypothesis.strategies import composite

# Helpers for Hypothesis-Python based property tests
# (custom strategies, etc.)


@composite
def ranged_slices(draw, min_value=0, max_value=10):
    bdd = st.one_of(st.none(), st.integers(min_value=min_value, max_value=max_value))
    start = draw(bdd)
    stop = draw(bdd)
    step = draw(bdd)

    return slice(start, stop, step)
