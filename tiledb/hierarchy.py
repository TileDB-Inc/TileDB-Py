from __future__ import absolute_import, print_function


class Group(object):

    def __init__(self, path=None):
        self._path = path

    @property
    def path(self):
        self._path

    @property
    def name(self):
        if self._path:
            name = self._path
            if name[0] != '/':
                name = '/'.join((name,))
            return name
        return '/'

    def __eq__(self, other):
        return isinstance(other, Group) and self._path == other.path

    def __repr__(self):
        return "<tiledb.Group '%s'>" % self.name


def group(path=None):
    if path is None:
        return False
    return True