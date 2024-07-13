import subprocess
import time

import toml


def main():
    with open("pyproject.toml", "r") as f:
        pyproject_toml = toml.load(f)

    version = pyproject_toml["project"]["version"]

    with open("Cargo.toml", "r") as f:
        cargo_toml = toml.load(f)

    cargo_toml["package"]["version"] = version

    with open("Cargo.toml", "w") as f:
        toml.dump(cargo_toml, f)

    print("waiting for cargo.lock to update")
    time.sleep(2)

    print("committing version changes")
    subprocess.run(["git", "commit", "-am", version], check=True)

    print("pushing to remote")
    subprocess.run(["git", "push", "origin"], check=True)
    time.sleep(2)

    print("creating release")
    subprocess.run(
        ["gh", "release", "create", f"v{version}", "--generate-notes"], check=True
    )

    print("compiling docs")
    subprocess.run(
        ["mike", "deploy", "--push", "--update-aliases", f"{version}", "latest"]
    )
    subprocess.run(["mike", "set-default", "--push", "latest"])


if __name__ == "__main__":
    main()
