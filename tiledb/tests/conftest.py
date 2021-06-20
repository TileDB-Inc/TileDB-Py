import ctypes
import pytest
import sys

if sys.platform != "win32":

    @pytest.fixture(scope="function", autouse=True)
    def no_output(capfd):
        yield

        # flush stdout
        libc = ctypes.CDLL(None)
        libc.fflush(None)

        out, err = capfd.readouterr()
        if out or err:
            pytest.fail(f"Output captured: {out + err}")
