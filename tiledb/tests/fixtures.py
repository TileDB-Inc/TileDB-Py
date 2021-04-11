import pytest


@pytest.fixture(scope="module", params=["hilbert", "row-major"])
def sparse_cell_order(request):
    yield request.param
