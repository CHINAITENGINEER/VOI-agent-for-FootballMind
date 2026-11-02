"""Run one label-free postread request using immutable, correctly paired local f/g.

The request contains acquired bodies and saved semantic reviews, never outcomes.
This CLI cannot send HTTP/API requests or modify the online service.
"""
import argparse
import json
from pathlib import Path
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_research.acquisition.postread_selector import PostreadSelector, run_request


def forbidden(*args, **kwargs):
    raise RuntimeError("Single-match postread research inference is offline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--f-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output; existing runs are immutable")
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = socket.getaddrinfo = forbidden
    selector = PostreadSelector(args.model_dir, args.f_model)
    result = run_request(selector, json.loads(args.request.read_text(encoding="utf-8")))
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k:result[k] for k in ("physical_match_id", "model_name", "p_before", "p_after",
                                          "f_adopted_body_ids", "stop_reason")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
