import subprocess
import sys
import time

import toml


def main():
    new_version = str(sys.argv[1])

    with open("Cargo.toml", "r") as f:
        cargo_toml = toml.load(f)

    cargo_toml["package"]["version"] = new_version

    with open("Cargo.toml", "w") as f:
        toml.dump(cargo_toml, f)

    print("waiting for cargo.lock to update")
    time.sleep(2)

    print("committing version changes")
    subprocess.run(["git", "commit", "-am", new_version], check=True)

    print("pushing to remote")
    subprocess.run(["git", "push", "origin"], check=True)
    time.sleep(2)

    print("creating release")
    subprocess.run(
        ["gh", "release", "create", f"v{new_version}", "--generate-notes"], check=True
    )


if __name__ == "__main__":
    main()
