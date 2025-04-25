import os

import pytest

import tiledb
import tiledb.libtiledb as lt

from .common import DiskTestCase

if not (lt.version()[0] == 2 and lt.version()[1] >= 28):
    pytest.skip(
        "Profile is only available in TileDB 2.28 and later",
        allow_module_level=True,
    )


class ProfileTestCase(DiskTestCase):
    def setup_method(self):
        super().setup_method()
        self.profile1 = tiledb.Profile()  # default profile
        self.profile2 = tiledb.Profile("test_profile")  # named profile
        self.profile3 = tiledb.Profile(
            homedir=self.path("profile3_dir")
        )  # profile with custom home directory
        self.profile4 = tiledb.Profile(
            "test_profile", self.path("profile4_dir")
        )  # named profile with custom home directory
        self.vfs.create_dir(self.path("profile4_dir"))


class ProfileTest(ProfileTestCase):
    def test_profile_name(self):
        assert self.profile1.name == "default"
        assert self.profile2.name == "test_profile"
        assert self.profile3.name == "default"
        assert self.profile4.name == "test_profile"

    def test_profile_homedir(self):
        assert self.profile3.homedir == self.path("profile3_dir")
        assert self.profile4.homedir == self.path("profile4_dir")

    def test_profile_set_get_param(self):
        self.profile1.set_param("rest.username", "my_username")
        assert self.profile1.get_param("rest.username") == "my_username"

        self.profile3.set_param("rest.server_address", "https://myaddress.com")
        assert self.profile3.get_param("rest.server_address") == "https://myaddress.com"

    def test_profile_repr(self):
        assert 'rest.password": ""' in repr(self.profile1)
        # set a parameter and check the repr again
        self.profile1.set_param("rest.password", "test")
        assert 'rest.password": "test"' in repr(self.profile1)

    def test_profile_save(self):
        assert os.path.exists(self.profile4.homedir)

        tiledb_dir = os.path.join(self.profile4.homedir, ".tiledb")
        assert not os.path.exists(tiledb_dir)

        self.profile4.save()

        # this fails ... for some reason
        assert os.path.exists(tiledb_dir)

    # TODO: also test remove
