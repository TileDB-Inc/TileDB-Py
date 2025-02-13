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
def bounded_anytuple(draw, *, min_value=0, max_value=10):
    """hypothesis composite strategy that returns a `length` tuple of integers
    within the range (min_value, max_value)
    """

    n = draw(st.integers(min_value=1, max_value=100))
    return draw(st.tuples(*[st.integers(min_value, max_value) for _ in range(n)]))


@composite
def ranged_slices(draw, min_value=0, max_value=10):
    bdd = st.integers(min_value=min_value, max_value=max_value)
    start = draw(bdd.filter(lambda x: x is not None))
    stop = draw(bdd.filter(lambda x: x is not None))
    start, stop = sorted((start, stop))
    # step = draw(bdd)

    return slice(start, stop, None)
