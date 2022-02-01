import urllib

# from tiledb.tests.common import paths_equal


def paths_equal(path1, path2):
    p1 = urllib.parse.urlparse(path1)
    p2 = urllib.parse.urlparse(path2)

    if p1.scheme == p2.scheme:
        pass
    elif "file" in (p1.scheme, p2.scheme):
        # if
        scheme_eq = p1.scheme == "file" or p1.scheme == ""
        scheme_eq |= p2.scheme == "file" or p2.scheme == ""
        return p1.path == p2.path and scheme_eq

    return p1.schema == p2.schema and p1.path == p2.path and p1.netloc == p2.netloc
