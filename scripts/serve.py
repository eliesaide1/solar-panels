"""Start the SolarMap web UI at http://127.0.0.1:8000

    python scripts/serve.py
"""

import argparse

import _bootstrap  # noqa: F401

import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()

    print(f"SolarMap UI -> http://{args.host}:{args.port}")
    uvicorn.run(
        "solarmap.api.main:app", host=args.host, port=args.port, reload=args.reload
    )


if __name__ == "__main__":
    main()
