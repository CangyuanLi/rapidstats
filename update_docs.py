import subprocess

import toml


def main():
    with open("pyproject.toml", "r") as f:
        pyproject_toml = toml.load(f)

    version = pyproject_toml["project"]["version"]

    print("compiling docs")
    subprocess.run(
        ["mike", "deploy", "--push", "--update-aliases", f"{version}", "latest"]
    )
    subprocess.run(["mike", "set-default", "--push", "latest"])


if __name__ == "__main__":
    main()
