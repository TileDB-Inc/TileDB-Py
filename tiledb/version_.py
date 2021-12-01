from .version import version, version_tuple


class VersionHelper:
    def __getattr__(self, name):
        if name == "version":
            return version
        else:
            raise AttributeError

    def __call__(self):
        return version_tuple
