import pytest


@pytest.fixture(scope="function", autouse=True)
def no_output(capfd):
    yield

    # flush stdout
    import ctypes

    libc = ctypes.CDLL(None)
    libc.fflush(None)

    out, err = capfd.readouterr()
    if out or err:
        pytest.fail(f"Output captured: {out + err}")
