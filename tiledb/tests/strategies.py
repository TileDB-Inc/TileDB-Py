from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import composite

# Helpers for Hypothesis-Python based property tests
# (custom strategies, etc.)


@composite
def bounded_ntuple(draw, *, length=1, min_value=0, max_value=10):
    """hypothesis composite strategy that returns a `length` tuple of integers
    within the range (min_value, max_value)
    """

    return draw(st.tuples(*[st.integers(min_value, max_value) for _ in range(length)]))


@composite
def ranged_slices(draw, min_value=0, max_value=10):
    bdd = st.one_of(st.none(), st.integers(min_value=min_value, max_value=max_value))
    start = draw(bdd)
    stop = draw(bdd)
    step = draw(bdd)

    return slice(start, stop, step)
