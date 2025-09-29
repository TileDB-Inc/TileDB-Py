import json
from pathlib import Path

import pytest

import tiledb
import tiledb.libtiledb as lt

from .common import DiskTestCase

if lt.version() < (2, 28, 1):
    pytest.skip(
        "Profile is only available in TileDB 2.28.1 and later",
        allow_module_level=True,
    )

"""
Due to the nature of Profiles, they are touching the filesystem,
so we need to be careful and not affect the user's Profiles.
Thus we use DiskTestCase to create temporary directories.
"""


class ProfileTestCase(DiskTestCase):
    def setup_method(self):
        super().setup_method()
        self.profile1 = tiledb.Profile(
            dir=self.path("profile1_dir")
        )  # profile with custom directory
        self.profile2 = tiledb.Profile(
            "profile2_name", self.path("profile2_dir")
        )  # named profile with custom directory


class ProfileTest(ProfileTestCase):
    def test_profile_name(self):
        assert self.profile1.name == "default"
        assert self.profile2.name == "profile2_name"

    def test_profile_dir(self):
        assert Path(self.profile1.dir) == Path(self.path("profile1_dir"))
        assert Path(self.profile2.dir) == Path(self.path("profile2_dir"))

    def test_profile_set_get_param(self):
        username = "my_username"
        server_address = "https://my.address"

        self.profile1["rest.username"] = username
        assert self.profile1["rest.username"] == username

        self.profile1["rest.server_address"] = server_address
        assert self.profile1["rest.server_address"] == server_address

    def test_profile_repr(self):
        password = "testing_the_password"
        payer_namespace = "testing_the_namespace"
        server_address = "https://testing_the_address.com"
        token = "testing_the_token"
        username = "testing_the_username"

        self.profile1["rest.password"] = password
        self.profile1["rest.payer_namespace"] = payer_namespace
        self.profile1["rest.server_address"] = server_address
        self.profile1["rest.token"] = token
        self.profile1["rest.username"] = username
        with pytest.raises(KeyError):
            # This should raise KeyError because the profile does not have this parameter
            self.profile1["rest.non_existent_param"]

        goal_dict = {
            "default": {
                "rest.password": password,
                "rest.payer_namespace": payer_namespace,
                "rest.server_address": server_address,
                "rest.token": token,
                "rest.username": username,
            }
        }

        assert goal_dict == json.loads(repr(self.profile1))

    def test_profile_save_load_remove(self):
        token = "testing_the_token_for_profile2"
        payer_namespace = "testing_the_namespace_for_profile2"

        self.profile2["rest.token"] = token
        self.profile2["rest.payer_namespace"] = payer_namespace

        # save the profile
        self.profile2.save()

        # load the profile
        loaded_profile = tiledb.Profile.load("profile2_name", self.path("profile2_dir"))

        # check that the loaded profile has the same parameters
        assert loaded_profile.name == "profile2_name"
        assert Path(loaded_profile.dir) == Path(self.path("profile2_dir"))
        assert loaded_profile["rest.token"] == token
        assert loaded_profile["rest.payer_namespace"] == payer_namespace

        # remove the profile
        tiledb.Profile.remove("profile2_name", self.path("profile2_dir"))


class ConfigWithProfileTest(ProfileTestCase):
    def test_config_with_profile(self):
        username = "username_coming_from_profile"
        password = "password_coming_from_profile"
        server_address = "https://profile_address.com"

        # Create a profile and set some parameters
        profile = tiledb.Profile(dir=self.path("profile_with_config_dir"))
        profile["rest.username"] = username
        profile["rest.password"] = password
        profile["rest.server_address"] = server_address

        # Save the profile
        profile.save()

        # -----
        # The above is done only once, so we can use the same profile later
        # -----

        # Create a config and set the profile directory
        config = tiledb.Config()
        config["profile_dir"] = self.path("profile_with_config_dir")
        # Test that the config parameters are set correctly
        assert config["rest.username"] == username
        assert config["rest.password"] == password
        assert config["rest.server_address"] == server_address

        # Alternatively, we can set the profile details directly in the Config constructor
        config2 = tiledb.Config({"profile_dir": self.path("profile_with_config_dir")})
        # Test that the config parameters are set correctly
        assert config2["rest.username"] == username
        assert config2["rest.password"] == password
        assert config2["rest.server_address"] == server_address
