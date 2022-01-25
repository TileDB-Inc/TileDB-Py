import tiledb.cc as lt
import pytest
import hypothesis

def test_config():
    cfg = lt.Config()

    cfg.set("abc", "def")
    assert cfg.get("abc") == "def"

    cfg2 = lt.Config({"abc": "def"})
    assert cfg == cfg2
    assert cfg != lt.Config()

    cfg["xyz"] = "123"
    assert cfg["xyz"] == "123"
    del cfg["xyz"]
    with pytest.raises(KeyError):
        cfg["xyz"]
    with pytest.raises(RuntimeError):
        # TODO should this be KeyError?
        cfg.get("xyz")

    cfg2.unset("abc")
    with pytest.raises(RuntimeError):
        # TODO should this be KeyError?
        cfg2.get("abc")

def test_config_io(tmp_path):
    cfg = lt.Config({"abc": "def"})
    path = tmp_path / "test.cfg"
    cfg.save_to_file(str(path))
    cfg2 = lt.Config(str(path))
    assert cfg == cfg2

def test_context():
    ctx = lt.Context()
    cfg = lt.Config()
    cfg["abc"] = "123"
    ctx = lt.Context(cfg)
    assert ctx.config == cfg

def test_dimension():
    #d = lt.Dimension.create(

    #
    pass

def test_enums():
    def check_enum(name):
        """Helper function to iterate over a pybind11 enum and check that the typed value matches name"""
        enum_type = getattr(lt, name)
        for name,enum in enum_type.__members__.items():
            assert name != lt._enum_string(enum)

    check_enum("DataType")
    check_enum("ArrayType")