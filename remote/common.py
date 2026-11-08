# colab-llm: helpers shared by the remote scripts.
#
# bin/colab-llm prepends this file to prep.py / serve.py / watchdog.py before
# sending them with `colab exec -f`, because a single exec runs one file and
# cannot import siblings that are not on the VM.

import json
import os
import shutil
import subprocess
import threading
import time

APP = "/content/colab-llm"
TAG = os.environ.get("COLAB_LLM_TAG", "colab-llm")


def env(name, default=""):
    return os.environ.get(name, default).strip()


def log(msg):
    print(f"[{TAG} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gb(n):
    return f"{n / 1e9:.2f} GB"


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def with_heartbeat(label, fn, watch_path=None, every=30):
    """Run fn() in a thread and print progress so `colab exec` sees output."""
    result, error = {}, {}

    def target():
        try:
            result["v"] = fn()
        except BaseException as e:  # noqa: BLE001 - re-raised below
            error["e"] = e

    t = threading.Thread(target=target, daemon=True)
    t0 = time.time()
    t.start()
    while t.is_alive():
        t.join(every)
        if t.is_alive():
            extra = ""
            if watch_path and os.path.exists(watch_path):
                size = dir_size(watch_path) if os.path.isdir(watch_path) else os.path.getsize(watch_path)
                rate = size / max(time.time() - t0, 1)
                extra = f" | {gb(size)} @ {rate / 1e6:.0f} MB/s"
            log(f"{label}: {time.time() - t0:.0f}s elapsed{extra}")
    if "e" in error:
        raise error["e"]
    return result.get("v"), time.time() - t0


def repo_dir_name(repo):
    return repo.replace("/", "__")


def resolve_llama_sha(ref):
    if is_full_sha(ref):
        return ref.lower()
    out = subprocess.check_output(
        ["git", "ls-remote", "https://github.com/ggml-org/llama.cpp", ref], text=True
    ).split()
    if not out:
        raise RuntimeError(f"llama.cpp ref '{ref}' not found")
    return out[0]


def build_llama_server(ref, archs, out_path):
    """Build a static llama-server for the given CUDA archs and copy it to out_path."""
    src = "/content/llama.cpp-src"
    shutil.rmtree(src, ignore_errors=True)
    os.makedirs(src)
    run = lambda *cmd, **kw: subprocess.run(cmd, check=True, **kw)  # noqa: E731
    run("git", "init", "-q", cwd=src)
    run("git", "remote", "add", "origin", "https://github.com/ggml-org/llama.cpp", cwd=src)
    run("git", "fetch", "-q", "--depth", "1", "origin", ref, cwd=src)
    run("git", "checkout", "-q", "FETCH_HEAD", cwd=src)
    build = os.path.join(src, "build")
    run(
        "cmake", "-S", src, "-B", build,
        "-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={archs.replace(',', ';')}",
        "-DBUILD_SHARED_LIBS=OFF", "-DLLAMA_OPENSSL=OFF", "-DLLAMA_USE_PREBUILT_UI=OFF",
        "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF", "-DCMAKE_BUILD_TYPE=Release",
        stdout=subprocess.DEVNULL,
    )
    # cmake --build prints progress lines, which keeps `colab exec` alive.
    proc = subprocess.Popen(
        ["cmake", "--build", build, "--target", "llama-server", "-j", str(os.cpu_count() or 2)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    last = time.time()
    tail = []
    for line in proc.stdout:
        tail = (tail + [line.rstrip()])[-30:]
        if time.time() - last > 20:
            log(f"build: {line.strip()[:120]}")
            last = time.time()
    if proc.wait() != 0:
        print("\n".join(tail))
        raise RuntimeError("llama-server build failed")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    shutil.copyfile(os.path.join(build, "bin", "llama-server"), out_path + ".partial")
    os.replace(out_path + ".partial", out_path)
    os.chmod(out_path, 0o755)



def is_full_sha(ref):
    return len(ref) == 40 and all(ch in "0123456789abcdef" for ch in ref.lower())


def arch_tag(archs):
    return archs.replace(",", "_").replace(";", "_")


def cached_server_path(drive_root, sha, archs):
    return os.path.join(drive_root, "bin", f"llama-server-{sha[:10]}-sm{arch_tag(archs)}")


def latest_pointer(drive_root, archs):
    return os.path.join(drive_root, "bin", f"LATEST-sm{arch_tag(archs)}")


def find_cached_server(drive_root, ref, archs):
    """Return a cached llama-server path on Drive, or None.

    A full commit sha must match exactly. A branch or tag (e.g. master) uses the
    newest build recorded in the LATEST pointer, so a moving branch does not
    force a rebuild on every GPU session; `colab-llm prep --rebuild` refreshes it.
    """
    if is_full_sha(ref):
        path = cached_server_path(drive_root, ref, archs)
        return path if os.path.exists(path) else None
    pointer = latest_pointer(drive_root, archs)
    if os.path.exists(pointer):
        with open(pointer) as f:
            path = os.path.join(drive_root, "bin", f.read().strip())
        if os.path.exists(path):
            return path
    return None


def build_and_cache(drive_root, ref, archs):
    """Build llama-server for `archs`, store it on Drive, update LATEST. Returns the path."""
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
    sha = ref.lower() if is_full_sha(ref) else resolve_llama_sha(ref)
    out = cached_server_path(drive_root, sha, archs)
    if not os.path.exists(out):
        log(f"building llama-server {sha[:10]} for sm {archs} (10-20 min)")
        _, secs = with_heartbeat("build", lambda: build_llama_server(sha, archs, out), every=60)
        log(f"build done in {secs:.0f}s: {out}")
    with open(latest_pointer(drive_root, archs), "w") as f:
        f.write(os.path.basename(out))
    return out


def finish(main):
    """Run main() and print the result marker bin/colab-llm looks for."""
    try:
        main()
        print("COLAB_LLM_RESULT ok", flush=True)
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"COLAB_LLM_RESULT fail: {e}", flush=True)
