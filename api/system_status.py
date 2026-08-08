"""
system_status.py — Hardware / system status for API + Triton.

Proxies Triton health, model statistics, and Prometheus metrics where available.
Adds API-host CPU/RAM/GPU utilization.
"""
from __future__ import annotations

import os
import asyncio
import re
import subprocess
from typing import Any, Optional

import httpx

from gpu_manager import discover_gpus, models_per_gpu

TRITON_HTTP_URL = os.getenv("TRITON_HTTP_URL", "http://triton-remote:8000")
TRITON_METRICS_URL = os.getenv("TRITON_METRICS_URL", "http://triton-remote:8002/metrics")


async def _get_json(client: httpx.AsyncClient, url: str) -> Optional[dict | list]:
    try:
        resp = await client.get(url, timeout=10.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


async def _get_text(client: httpx.AsyncClient, url: str) -> Optional[str]:
    try:
        resp = await client.get(url, timeout=15.0)
        if resp.status_code == 200:
            return resp.text
    except Exception:
        pass
    return None


def _host_memory() -> dict[str, Any]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {
            "total_mb": round(vm.total / (1024 * 1024)),
            "used_mb": round(vm.used / (1024 * 1024)),
            "available_mb": round(vm.available / (1024 * 1024)),
            "percent": vm.percent,
        }
    except Exception:
        return {}


def _host_cpu() -> dict[str, Any]:
    try:
        import psutil

        return {
            "percent": psutil.cpu_percent(interval=0.1),
            "count": psutil.cpu_count(logical=True) or 0,
        }
    except Exception:
        return {}


def _gpu_utilization() -> list[dict[str, Any]]:
    """nvidia-smi utilization + memory (complements discover_gpus)."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []

    rows: list[dict[str, Any]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        rows.append({
            "index": int(parts[0]),
            "name": parts[1],
            "memory_used_mb": int(parts[2]) if parts[2].isdigit() else None,
            "memory_total_mb": int(parts[3]) if parts[3].isdigit() else None,
            "gpu_util_percent": int(parts[4]) if parts[4].isdigit() else None,
            "memory_util_percent": int(parts[5]) if len(parts) > 5 and parts[5].isdigit() else None,
            "temperature_c": int(parts[6]) if len(parts) > 6 and parts[6].isdigit() else None,
        })
    return rows


def _parse_prometheus_summary(text: str) -> dict[str, Any]:
    """Extract high-level counters from Triton Prometheus metrics."""
    summary: dict[str, Any] = {
        "inference_success_total": 0,
        "inference_failure_total": 0,
        "models": {},
    }
    if not text:
        return summary

    success_re = re.compile(
        r'nv_inference_request_success\{([^}]*)\}\s+(\S+)'
    )
    failure_re = re.compile(
        r'nv_inference_request_failure\{([^}]*)\}\s+(\S+)'
    )

    def _labels(blob: str) -> dict[str, str]:
        return dict(re.findall(r'(\w+)="([^"]*)"', blob))

    for m in success_re.finditer(text):
        labels = _labels(m.group(1))
        val = int(float(m.group(2)))
        model = labels.get("model", "unknown")
        summary["inference_success_total"] += val
        summary["models"].setdefault(model, {"success": 0, "failure": 0})
        summary["models"][model]["success"] += val

    for m in failure_re.finditer(text):
        labels = _labels(m.group(1))
        val = int(float(m.group(2)))
        model = labels.get("model", "unknown")
        summary["inference_failure_total"] += val
        summary["models"].setdefault(model, {"success": 0, "failure": 0})
        summary["models"][model]["failure"] += val

    return summary


async def collect_system_status(
    api_health: dict[str, Any],
    model_repo: str,
) -> dict[str, Any]:
    """Aggregate API + Triton + host resource status."""
    async with httpx.AsyncClient() as client:
        triton_live = await _get_text(client, f"{TRITON_HTTP_URL.rstrip('/')}/v2/health/live")
        triton_ready = await _get_text(client, f"{TRITON_HTTP_URL.rstrip('/')}/v2/health/ready")
        triton_meta = await _get_json(client, f"{TRITON_HTTP_URL.rstrip('/')}/v2")
        model_stats = await _get_json(client, f"{TRITON_HTTP_URL.rstrip('/')}/v2/models/stats")
        metrics_text = await _get_text(client, TRITON_METRICS_URL)

    gpus_static = discover_gpus()
    gpus_live = _gpu_utilization()
    gpu_by_index = {g["index"]: g for g in gpus_live}
    gpus = []
    for g in gpus_static:
        merged = dict(g)
        if g["index"] in gpu_by_index:
            merged.update(gpu_by_index[g["index"]])
        gpus.append(merged)

    # Fetch triton container resource usage synchronously in threadpool
    triton_stats = await asyncio.to_thread(_get_triton_container_stats)

    return {
        "api": api_health,
        "triton": {
            "live": triton_live == "",
            "ready": triton_ready == "",
            "server": triton_meta,
            "http_url": TRITON_HTTP_URL,
            "metrics_url": TRITON_METRICS_URL,
            "container_stats": triton_stats,
        },
        "host": {
            "cpu": _host_cpu(),
            "memory": _host_memory(),
        },
        "gpus": gpus,
        "models_by_gpu": models_per_gpu(model_repo),
        "triton_model_stats": model_stats,
        "triton_metrics_summary": _parse_prometheus_summary(metrics_text or ""),
    }

def _get_triton_container_stats() -> dict[str, Any]:
    """Queries docker socket to find triton-remote container CPU/memory usage."""
    import socket
    import json
    if not os.path.exists("/var/run/docker.sock"):
        return {}
        
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.settimeout(2.0)
        s.connect("/var/run/docker.sock")
        
        # 1. Discover the Triton container ID dynamically
        s.sendall(b"GET /containers/json HTTP/1.0\r\nHost: localhost\r\n\r\n")
        response = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            response += chunk
        parts = response.split(b"\r\n\r\n", 1)
        if len(parts) < 2:
            return {}
        
        containers = json.loads(parts[1].decode('utf-8', errors='ignore'))
        container_id = None
        for c in containers:
            for name in c.get("Names", []):
                if "triton-remote" in name:
                    container_id = c["Id"]
                    break
            if container_id:
                break
                
        if not container_id:
            return {}
            
        # 2. Query stats for the container
        s.close()
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect("/var/run/docker.sock")
        
        request = f"GET /containers/{container_id}/stats?stream=false HTTP/1.0\r\nHost: localhost\r\n\r\n"
        s.sendall(request.encode('utf-8'))
        
        response = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            response += chunk
            
        parts = response.split(b"\r\n\r\n", 1)
        if len(parts) < 2:
            return {}
            
        stats = json.loads(parts[1].decode('utf-8', errors='ignore'))
        
        # Calculate CPU Usage
        cpu_percent = 0.0
        try:
            cpu_usage = stats["cpu_stats"]["cpu_usage"]["total_usage"]
            precpu_usage = stats["precpu_stats"]["cpu_usage"]["total_usage"]
            system_cpu = stats["cpu_stats"]["system_cpu_usage"]
            presystem_cpu = stats["precpu_stats"]["system_cpu_usage"]
            
            cpu_delta = cpu_usage - precpu_usage
            system_delta = system_cpu - presystem_cpu
            
            if system_delta > 0 and cpu_delta > 0:
                online_cpus = stats["cpu_stats"].get("online_cpus") or len(stats["cpu_stats"]["cpu_usage"].get("percpu_usage") or [1])
                cpu_percent = (cpu_delta / system_delta) * online_cpus * 100.0
        except KeyError:
            pass
            
        # Calculate Memory Usage
        mem_used_mb = 0.0
        mem_limit_mb = 0.0
        mem_percent = 0.0
        try:
            mem_stats = stats["memory_stats"]
            usage = mem_stats["usage"]
            cache = 0
            if "stats" in mem_stats:
                cache = mem_stats["stats"].get("inactive_file", 0) or mem_stats["stats"].get("cache", 0)
            
            active_usage = max(0, usage - cache)
            limit = mem_stats["limit"]
            
            mem_used_mb = active_usage / (1024 * 1024)
            mem_limit_mb = limit / (1024 * 1024)
            mem_percent = (active_usage / limit) * 100.0 if limit > 0 else 0.0
        except KeyError:
            pass
            
        return {
            "cpu_percent": round(cpu_percent, 1),
            "memory_used_mb": round(mem_used_mb, 1),
            "memory_limit_mb": round(mem_limit_mb, 1),
            "memory_percent": round(mem_percent, 1)
        }
    except Exception:
        return {}
    finally:
        s.close()


async def fetch_triton_metrics_raw() -> str:
    async with httpx.AsyncClient() as client:
        text = await _get_text(client, TRITON_METRICS_URL)
    return text or ""
