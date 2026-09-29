import subprocess
import sys

SITE_MODULES = [
    "src.scrapers.kontan",
    "src.scrapers.bisnis",
    "src.scrapers.cnbc_indonesia",
]


def main():
    print("Spawning parallel scrapers across all target sites...")
    procs = [
        subprocess.Popen([sys.executable, "-m", module])
        for module in SITE_MODULES
    ]
    exit_codes = [p.wait() for p in procs]

    for module, code in zip(SITE_MODULES, exit_codes):
        status = "Completed successfully" if code == 0 else f"Failed (exit code {code})"
        print(f"[{module}]: {status}")

    if any(exit_codes):
        sys.exit(1)


if __name__ == "__main__":
    main()