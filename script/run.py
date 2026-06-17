import math
import operator
import os
import signal
import socket
import subprocess
import sys
import time

import hydra
from omegaconf import OmegaConf

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CARLA_ROOT = "/home/codon/CARLA/CARLA_0.9.12"
os.environ.setdefault("PC_HMD_ROOT", ROOT_DIR)
os.environ.setdefault("CARLA_ROOT", DEFAULT_CARLA_ROOT)
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
sys.path.insert(0, ROOT_DIR)

OmegaConf.register_new_resolver("mul", operator.mul, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
OmegaConf.register_new_resolver("round_down", math.floor, replace=True)

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import env


def _port_is_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _tail_file(path, max_lines=40):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as stream:
            return "".join(stream.readlines()[-max_lines:])
    except OSError:
        return ""


def _find_carla_pids():
    pids = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/cmdline", "rb") as stream:
                command = stream.read().replace(b"\0", b" ").decode(
                    "utf-8", errors="replace"
                )
        except OSError:
            continue
        if "CarlaUE4-Linux-Shipping" in command:
            pids.append(int(name))
    return pids


def stop_existing_carla(port, timeout=10):
    pids = _find_carla_pids()
    if not pids and not _port_is_open(port):
        return
    if not pids:
        raise RuntimeError(
            f"Port {port} is occupied by a non-CARLA process; refusing to kill it."
        )

    print(f"[CARLA] Stopping previous CARLA processes: {pids}", flush=True)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _find_carla_pids() and not _port_is_open(port):
            print(f"[CARLA] Port {port} released.", flush=True)
            return
        time.sleep(0.2)

    remaining = _find_carla_pids()
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(0.5)
    if _port_is_open(port):
        raise RuntimeError(f"Port {port} is still occupied after stopping CARLA.")
    print(f"[CARLA] Port {port} released after force stop.", flush=True)


def start_carla(port, map_name, quality="Epic", offscreen=True):
    carla_root = os.environ.get("CARLA_ROOT", DEFAULT_CARLA_ROOT)
    candidates = (
        os.path.join(
            carla_root,
            "CarlaUE4",
            "Binaries",
            "Linux",
            "CarlaUE4-Linux-Shipping",
        ),
        os.path.join(carla_root, "CarlaUE4.sh"),
    )
    executable = next((path for path in candidates if os.path.isfile(path)), None)
    if executable is None:
        raise FileNotFoundError(f"CARLA executable not found under {carla_root}")
    stop_existing_carla(port)

    log_path = os.path.join(ROOT_DIR, "carla_server.log")
    cmd = [executable]
    if executable.endswith("CarlaUE4-Linux-Shipping"):
        cmd.append("CarlaUE4")
    cmd.extend(
        [
            f"/Game/Carla/Maps/{map_name}",
            f"-carla-rpc-port={port}",
            f"-quality-level={quality}",
            "-nosound",
        ]
    )
    if offscreen:
        cmd.append("-RenderOffScreen")

    print(
        f"[CARLA] Starting {map_name} on port {port} "
        f"(quality={quality}, offscreen={offscreen})...",
        flush=True,
    )
    log_file = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_file.close()
    return process, log_path


def wait_for_carla(process, port, map_name, timeout, log_path):
    import carla

    deadline = time.monotonic() + timeout
    last_error = None
    map_load_requested = False
    print(f"[CARLA] Waiting for {map_name}...", flush=True)

    while time.monotonic() < deadline:
        return_code = process.poll() if process is not None else None
        if process is not None and return_code is not None:
            log_tail = _tail_file(log_path)
            raise RuntimeError(
                f"CARLA exited during startup with code {return_code}.\n"
                f"Log: {log_path}\n{log_tail}"
            )
        if not _port_is_open(port):
            time.sleep(1)
            continue

        try:
            client = carla.Client("127.0.0.1", port)
            client.set_timeout(10.0)
            world = client.get_world()
            loaded_map = world.get_map().name.rsplit("/", 1)[-1]
            if loaded_map == map_name:
                print(f"[CARLA] Ready: {loaded_map}", flush=True)
                return
            if not map_load_requested:
                map_load_requested = True
                client.load_world(map_name)
                continue
            last_error = RuntimeError(
                f"CARLA loaded {loaded_map}, expected {map_name}"
            )
        except RuntimeError as error:
            last_error = error
        time.sleep(1)

    raise TimeoutError(
        f"CARLA was not ready after {timeout:.0f}s on port {port}. "
        f"Last error: {last_error}. Log: {log_path}"
    )


@hydra.main(
    version_base=None,
    config_path="../cfg",
    config_name="default",
)
def main(cfg: OmegaConf):
    visualize = bool(cfg.get("visualize", False))
    os.environ["CARLA_VISUALIZE"] = str(visualize)
    map_name = str(cfg.run.get("carla_map", "Town10HD_Opt"))
    carla_process, carla_log_path = start_carla(
        port=int(cfg.run.carla_port),
        map_name=map_name,
        quality=str(cfg.run.carla_quality),
        offscreen=not visualize,
    )
    try:
        wait_for_carla(
            process=carla_process,
            port=int(cfg.run.carla_port),
            map_name=map_name,
            timeout=float(cfg.run.carla_startup_seconds),
            log_path=carla_log_path,
        )
        OmegaConf.resolve(cfg)
        cls = hydra.utils.get_class(cfg._target_)
        agent = cls(cfg)
        agent.run()
    finally:
        if carla_process is not None:
            try:
                os.killpg(carla_process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                carla_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(carla_process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                carla_process.wait()


if __name__ == "__main__":
    main()
