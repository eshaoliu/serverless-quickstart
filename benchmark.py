#!/usr/bin/env python3
"""Simple concurrent benchmark for the RunPod serverless endpoint.

Uses only `requests` (no aiohttp) to keep dependencies minimal.

Usage:
    export RUNPOD_API_KEY=your_key_here
    python3 benchmark.py --concurrency 10 --requests 100

RunPod serverless /run returns a job ID immediately; this script optionally
polls /status/{id} until each job completes.
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

ENDPOINT_URL = "https://api.runpod.ai/v2/34orgs5ae40zo7"


def get_auth_header() -> dict[str, str]:
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise RuntimeError("Please set the RUNPOD_API_KEY environment variable.")
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def make_payload(prompt: str, max_tokens: int) -> dict:
    return {
        "input": {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 1.0,
        }
    }


def submit_one(session: requests.Session, payload: dict, timeout: int) -> dict:
    start = time.monotonic()
    try:
        resp = session.post(f"{ENDPOINT_URL}/run", json=payload, timeout=timeout)
        return {
            "status": resp.status_code,
            "latency": time.monotonic() - start,
            "data": resp.json() if resp.text else None,
        }
    except Exception as exc:
        return {
            "status": None,
            "latency": time.monotonic() - start,
            "error": str(exc),
        }


def poll_status(session: requests.Session, job_id: str, timeout: int, poll_interval: float = 1.0) -> dict:
    url = f"{ENDPOINT_URL}/status/{job_id}"
    start = time.monotonic()
    while True:
        if time.monotonic() - start > timeout:
            return {"completed": False, "error": "polling timeout"}
        try:
            resp = session.get(url, timeout=30)
            data = resp.json()
            status = data.get("status")
            if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
                return {
                    "completed": True,
                    "status": status,
                    "total_time": time.monotonic() - start,
                    "data": data,
                }
        except Exception as exc:
            return {"completed": False, "error": str(exc)}
        time.sleep(poll_interval)


def run_one(
    headers: dict[str, str],
    payload: dict,
    poll: bool,
    submit_timeout: int,
    poll_timeout: int,
) -> dict:
    session = requests.Session()
    session.headers.update(headers)

    result = submit_one(session, payload, submit_timeout)

    if poll and result.get("status") == 200:
        job_id = result.get("data", {}).get("id")
        if job_id:
            result["poll"] = poll_status(session, job_id, poll_timeout)

    return result


def main(
    concurrency: int,
    total_requests: int,
    prompt: str,
    max_tokens: int,
    poll: bool,
    submit_timeout: int,
    poll_timeout: int,
) -> None:
    headers = get_auth_header()
    payload = make_payload(prompt, max_tokens)

    results: list[dict] = []
    start = time.monotonic()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(run_one, headers, payload, poll, submit_timeout, poll_timeout)
            for _ in range(total_requests)
        ]
        for future in as_completed(futures):
            results.append(future.result())

    total_time = time.monotonic() - start

    # Summary
    successes = sum(1 for r in results if r.get("status") == 200)
    failures = total_requests - successes
    submit_latencies = [r["latency"] for r in results if "latency" in r]

    print(f"Total requests: {total_requests}")
    print(f"Concurrency: {concurrency}")
    print(f"Successful /run submits: {successes}")
    print(f"Failed /run submits: {failures}")
    print(f"Total wall time: {total_time:.2f}s")
    print(f"Throughput: {total_requests / total_time:.2f} req/s")
    if submit_latencies:
        avg = sum(submit_latencies) / len(submit_latencies)
        print(f"Submit latency avg: {avg:.3f}s")
        print(f"Submit latency min: {min(submit_latencies):.3f}s")
        print(f"Submit latency max: {max(submit_latencies):.3f}s")

    if poll:
        completed_polls = [
            r["poll"]
            for r in results
            if isinstance(r.get("poll"), dict) and r["poll"].get("completed")
        ]
        print(f"Completed poll results: {len(completed_polls)}")
        if completed_polls:
            poll_times = [p["total_time"] for p in completed_polls]
            avg_poll = sum(poll_times) / len(poll_times)
            print(f"End-to-end time avg: {avg_poll:.3f}s")
            print(f"End-to-end time max: {max(poll_times):.3f}s")

    if failures:
        print("\nFirst failure:")
        for r in results:
            if r.get("status") != 200:
                print(r)
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RunPod serverless benchmark")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--prompt", type=str, default="Hello, how are you?")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--poll", action="store_true", help="Poll /status until each job completes")
    parser.add_argument("--submit-timeout", type=int, default=60)
    parser.add_argument("--poll-timeout", type=int, default=300)
    args = parser.parse_args()

    main(
        concurrency=args.concurrency,
        total_requests=args.requests,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        poll=args.poll,
        submit_timeout=args.submit_timeout,
        poll_timeout=args.poll_timeout,
    )
