import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Sync MINIMAX_API_KEY into OpenClaw auth profiles")
    parser.add_argument(
        "--path",
        help="Path to auth-profiles.json",
    )
    parser.add_argument(
        "--token",
        help="MINIMAX_API_KEY token",
    )
    args = parser.parse_args()

    key = args.token

    if not key:
        print("[sync_minimax_key] MINIMAX_API_KEY not set, skipping")
        sys.exit(0)

    try:
        with open(args.path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[sync_minimax_key] File not found: {args.path}, skipping")
        sys.exit(0)
    except json.JSONDecodeError as e:
        print(f"[sync_minimax_key] Invalid JSON in {args.path}: {e}", file=sys.stderr)
        sys.exit(1)

    profiles = data.setdefault("profiles", {})
    updated = False

    for name in ("minimax-global", "minimax:cn", "minimax:global"):
        profile = profiles.get(name)
        if isinstance(profile, dict) and profile.get("provider") == "minimax":
            profile["key"] = key
            updated = True
            print(f"[sync_minimax_key] Updated profile: {name}")

    if updated:
        with open(args.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"[sync_minimax_key] Saved to {args.path}")
    else:
        print("[sync_minimax_key] No minimax profiles found")

    sys.exit(0)


if __name__ == "__main__":
    main()
