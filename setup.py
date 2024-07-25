import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["install", "develop"])
    parser.add_argument("--tiledb", type=str, required=False)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("-v", action="store_true")
    args = parser.parse_args()

    os.getcwd()

    cmd = [
        "pip",
        "install",
    ]

    if args.command == "develop":
        cmd.append("-e")

    cmd.append(os.getcwd())

    if args.tiledb:
        cmd.append(f"-Cskbuild.cmake.define.TILEDB_PATH={args.tiledb}")

    if args.debug:
        cmd.append(f"-Cskbuild.cmake.build-type=Debug")

    if args.v:
        cmd.append("-v")

    print(
        "Note: 'setup.py' is deprecated in the Python ecosystem. Limited backward compatibility is currently provided for 'install' and 'develop' commands as passthrough to 'pip'."
    )
    print("    running: ", f"`{' '.join(cmd)}`")

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
