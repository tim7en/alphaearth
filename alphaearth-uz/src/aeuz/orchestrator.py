import argparse, importlib, sys, json

MODULES = [
    "soil_moisture",
    "afforestation",
    "degradation",
    "riverbank",
    "protected_areas",
    "biodiversity",
    "urban_heat",
    "synthesis",
    "qa_module",
    "publish"
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run", default="all", help="module name or 'all'")
    args = ap.parse_args()

    if args.dry_run:
        print(json.dumps({"planned_modules": MODULES}, indent=2))
        return

    to_run = MODULES if args.run == "all" else [args.run]
    results = {}
    for name in to_run:
        if name not in MODULES:
            print(f"Unknown module: {name}", file=sys.stderr); sys.exit(2)
        mod = importlib.import_module(f"aeuz.{name}")
        print(f"=== Running {name} ===")
        results[name] = mod.run()

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
