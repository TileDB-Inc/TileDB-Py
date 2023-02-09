import os
import subprocess
import sys
import xml

import pytest

import tiledb

from .common import DiskTestCase


# Wrapper to execute specific code in subprocess so that we can ensure the thread count
# init is correct. Necessary because multiprocess.get_context is only available in Python 3.4+,
# and the multiprocessing method may be set to fork by other tests (e.g. dask).
def init_test_wrapper(cfg=None):
    python_exe = sys.executable
    cmd = (
        f"from tiledb.tests.test_context_and_config import init_test_helper; "
        f"init_test_helper({cfg})"
    )
    test_path = os.path.dirname(os.path.abspath(__file__))

    sp_output = subprocess.check_output([python_exe, "-c", cmd], cwd=test_path)
    return int(sp_output.decode("UTF-8").strip())


def init_test_helper(cfg=None):
    tiledb.default_ctx(cfg)
    concurrency_level = tiledb.default_ctx().config()["sm.io_concurrency_level"]
    print(int(concurrency_level))


class ContextTest(DiskTestCase):
    def test_default_ctx(self):
        ctx = tiledb.default_ctx()
        self.assertIsInstance(ctx, tiledb.Ctx)
        assert isinstance(ctx.config(), tiledb.libtiledb.Config)

    def test_default_ctx_errors(self):
        config = tiledb.Config()
        ctx = tiledb.Ctx(config=config)

        with pytest.raises(ValueError) as excinfo:
            tiledb.default_ctx(ctx)
        assert (
            "default_ctx takes in `tiledb.Config` object or dictionary with "
            "config parameters."
        ) == str(excinfo.value)

    def test_scope_ctx(self):
        key = "sm.memory_budget"
        ctx0 = tiledb.default_ctx()
        new_config_dict = {key: 42}
        new_config = tiledb.Config({key: 78})
        new_ctx = tiledb.Ctx({key: 61})

        assert tiledb.default_ctx() is ctx0
        assert tiledb.default_ctx().config()[key] == "5368709120"

        with tiledb.scope_ctx(new_config_dict) as ctx1:
            assert tiledb.default_ctx() is ctx1
            assert tiledb.default_ctx().config()[key] == "42"
            with tiledb.scope_ctx(new_config) as ctx2:
                assert tiledb.default_ctx() is ctx2
                assert tiledb.default_ctx().config()[key] == "78"
                with tiledb.scope_ctx(new_ctx) as ctx3:
                    assert tiledb.default_ctx() is ctx3 is new_ctx
                    assert tiledb.default_ctx().config()[key] == "61"
                assert tiledb.default_ctx() is ctx2
                assert tiledb.default_ctx().config()[key] == "78"
            assert tiledb.default_ctx() is ctx1
            assert tiledb.default_ctx().config()[key] == "42"

        assert tiledb.default_ctx() is ctx0
        assert tiledb.default_ctx().config()[key] == "5368709120"

    def test_scope_ctx_error(self):
        with pytest.raises(ValueError) as excinfo:
            with tiledb.scope_ctx([]):
                pass
        assert (
            "scope_ctx takes in `tiledb.Ctx` object, `tiledb.Config` object, "
            "or dictionary with config parameters."
        ) == str(excinfo.value)

    @pytest.mark.skipif(
        "pytest.tiledb_vfs == 's3'", reason="Test not yet supported with S3"
    )
    @pytest.mark.filterwarnings(
        # As of 0.17.0, a warning is emitted for the aarch64 conda builds with
        # the messsage:
        #     <jemalloc>: MADV_DONTNEED does not work (memset will be used instead)
        #     <jemalloc>: (This is the expected behaviour if you are running under QEMU)
        # This can be ignored as this is being run in a Docker image / QEMU and
        # is therefore expected behavior
        "ignore:This is the expected behaviour if you are running under QEMU"
    )
    def test_init_config(self):
        self.assertEqual(
            int(tiledb.default_ctx().config()["sm.io_concurrency_level"]),
            init_test_wrapper(),
        )

        self.assertEqual(3, init_test_wrapper({"sm.io_concurrency_level": 3}))


@pytest.mark.skipif(
    "pytest.tiledb_vfs == 's3'", reason="Test not yet supported with S3"
)
class TestConfig(DiskTestCase):
    def test_config(self):
        config = tiledb.Config()
        config["sm.memory_budget"] = 103
        assert repr(config) is not None
        tiledb.Ctx(config)

    def test_ctx_config(self):
        ctx = tiledb.Ctx({"sm.memory_budget": 103})
        config = ctx.config()
        self.assertEqual(config["sm.memory_budget"], "103")

    def test_vfs_config(self):
        config = tiledb.Config()
        config["vfs.min_parallel_size"] = 1
        ctx = tiledb.Ctx()
        self.assertEqual(ctx.config()["vfs.min_parallel_size"], "10485760")
        vfs = tiledb.VFS(config, ctx=ctx)
        self.assertEqual(vfs.config()["vfs.min_parallel_size"], "1")

    def test_config_iter(self):
        config = tiledb.Config()
        k, v = [], []
        for p in config.items():
            k.append(p[0])
            v.append(p[1])
        self.assertTrue(len(k) > 0)

        k, v = [], []
        for p in config.items("vfs.s3."):
            k.append(p[0])
            v.append(p[1])
        self.assertTrue(len(k) > 0)

    def test_config_bad_param(self):
        config = tiledb.Config()
        config["sm.foo"] = "bar"
        ctx = tiledb.Ctx(config)
        self.assertEqual(ctx.config()["sm.foo"], "bar")

    def test_config_unset(self):
        config = tiledb.Config()
        config["sm.memory_budget"] = 103
        del config["sm.memory_budget"]
        # check that config parameter is default
        self.assertEqual(
            config["sm.memory_budget"], tiledb.Config()["sm.memory_budget"]
        )

    def test_config_from_file(self):
        # skip: beacuse Config.load doesn't support VFS-supported URIs?
        if pytest.tiledb_vfs == "s3":
            pytest.skip(
                "TODO need more plumbing to make pandas use TileDB VFS to read CSV files"
            )

        config_path = self.path("config")
        with tiledb.FileIO(self.vfs, config_path, "wb") as fh:
            fh.write("sm.memory_budget 100")
        config = tiledb.Config.load(config_path)
        self.assertEqual(config["sm.memory_budget"], "100")

    def test_ctx_config_from_file(self):
        config_path = self.path("config")
        vfs = tiledb.VFS()
        with tiledb.FileIO(vfs, config_path, "wb") as fh:
            fh.write("sm.memory_budget 100")
        ctx = tiledb.Ctx(config=tiledb.Config.load(config_path))
        config = ctx.config()
        self.assertEqual(config["sm.memory_budget"], "100")

    def test_ctx_config_dict(self):
        ctx = tiledb.Ctx(config={"sm.memory_budget": "100"})
        config = ctx.config()
        assert issubclass(type(config), tiledb.libtiledb.Config)
        self.assertEqual(config["sm.memory_budget"], "100")

    def test_config_repr_html(self):
        config = tiledb.Config()
        try:
            assert xml.etree.ElementTree.fromstring(config._repr_html_()) is not None
        except:
            pytest.fail(
                f"Could not parse config._repr_html_(). Saw {config._repr_html_()}"
            )
