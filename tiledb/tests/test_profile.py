from pathlib import Path

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


class ProfileTest(ProfileTestCase):
    def test_profile_name(self):
        assert self.profile1.name == "default"
        assert self.profile2.name == "test_profile"
        assert self.profile3.name == "default"
        assert self.profile4.name == "test_profile"

    def test_profile_homedir(self):
        assert Path(self.profile1.homedir) == Path.home()
        assert Path(self.profile2.homedir) == Path.home()
        assert Path(self.profile3.homedir) == Path(self.path("profile3_dir"))
        assert Path(self.profile4.homedir) == Path(self.path("profile4_dir"))

    def test_profile_set_get_param(self):
        self.profile1["rest.username"] = "my_username"
        assert self.profile1["rest.username"] == "my_username"

        self.profile3["rest.server_address"] = "https://myaddress.com"
        assert self.profile3["rest.server_address"] == "https://myaddress.com"

    def test_profile_repr(self):
        self.profile1["rest.password"] = "testing_the_password"
        self.profile1["rest.payer_namespace"] = "testing_the_namespace"
        self.profile1["rest.server_address"] = "https://testing_the_address.com"
        self.profile1["rest.token"] = "testing_the_token"
        self.profile1["rest.username"] = "testing_the_username"

        import json

        goal_dict = {
            "default": {
                "rest.password": "testing_the_password",
                "rest.payer_namespace": "testing_the_namespace",
                "rest.server_address": "https://testing_the_address.com",
                "rest.token": "testing_the_token",
                "rest.username": "testing_the_username",
            }
        }

        assert goal_dict == json.loads(repr(self.profile1))

    def test_profile_set_save_load_get(self):
        self.profile4["rest.token"] = "testing_the_token_for_profile4"
        self.profile4["rest.payer_namespace"] = "testing_the_namespace_for_profile4"

        # save the profile
        self.profile4.save()

        # load
        new_profile = tiledb.Profile.load("test_profile", self.path("profile4_dir"))
        assert new_profile.name == "test_profile"
        assert new_profile.homedir == self.path("profile4_dir")
        assert new_profile["rest.username"] == ""
        assert new_profile["rest.password"] == ""
        assert new_profile["rest.server_address"] == "https://api.tiledb.com"
        assert new_profile["rest.token"] == "testing_the_token_for_profile4"
        assert (
            new_profile["rest.payer_namespace"] == "testing_the_namespace_for_profile4"
        )
