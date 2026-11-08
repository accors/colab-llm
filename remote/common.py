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

# hf_xet already downloads each file over many connections; high-performance
# mode raises its concurrency and buffers (fine on Colab's 8+ vCPU / 50+ GB
# VMs). Must be set before huggingface_hub is imported. XET_HIGH_PERF=0 to opt out.
if os.environ.get("XET_HIGH_PERF", "1") == "1":
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
# HF token: bin/colab-llm copies it over ssh to APP/hf_token (never through
# `colab exec`, which logs the code it sends). With a token, downloads count
# against your own rate-limit bucket instead of the shared Colab IP. Without
# one, stop huggingface_hub from trying the Colab Secrets vault: that only
# works from the Colab web UI and just times out with a warning under the CLI.
if not os.environ.get("HF_TOKEN") and os.path.exists(f"{APP}/hf_token"):
    with open(f"{APP}/hf_token") as _f:
        os.environ["HF_TOKEN"] = _f.read().strip()
if not os.environ.get("HF_TOKEN"):
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
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

    def size_of(path):
        if not path or not os.path.exists(path):
            return 0
        return dir_size(path) if os.path.isdir(path) else os.path.getsize(path)

    # Count only bytes added since this call started: a staging directory may
    # still hold the previous file while it is being copied to Drive.
    base = size_of(watch_path)
    t = threading.Thread(target=target, daemon=True)
    t0 = time.time()
    t.start()
    while t.is_alive():
        t.join(every)
        if t.is_alive():
            extra = ""
            if watch_path:
                done = max(size_of(watch_path) - base, 0)
                rate = done / max(time.time() - t0, 1)
                extra = f" | {gb(done)} @ {rate / 1e6:.0f} MB/s"
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


def server_name(sha, archs):
    return f"llama-server-{sha[:10]}-sm{arch_tag(archs)}"


def pointer_name(archs):
    return f"LATEST-sm{arch_tag(archs)}"


class DriveBinStore:
    """llama-server builds under DRIVE_MODELS/bin (needs `colab drivemount`)."""

    def __init__(self, drive_root):
        self.dir = os.path.join(drive_root, "bin")
        self.label = f"Drive {self.dir}"

    def get(self, ref, archs):
        if not os.path.ismount("/content/drive"):
            raise RuntimeError("BIN_STORE=drive but Google Drive is not mounted")
        if is_full_sha(ref):
            name = server_name(ref, archs)
        else:
            pointer = os.path.join(self.dir, pointer_name(archs))
            if not os.path.exists(pointer):
                return None
            with open(pointer) as f:
                name = f.read().strip()
        path = os.path.join(self.dir, name)
        return path if os.path.exists(path) else None

    def put(self, local_path, sha, archs):
        os.makedirs(self.dir, exist_ok=True)
        name = server_name(sha, archs)
        shutil.copyfile(local_path, os.path.join(self.dir, name))
        with open(os.path.join(self.dir, pointer_name(archs)), "w") as f:
            f.write(name)
        flush_drive()  # the VM may be stopped right after this


class HfBinStore:
    """llama-server builds in a private Hugging Face model repo (needs HF_TOKEN)."""

    def __init__(self, repo):
        if not repo:
            raise RuntimeError("BIN_STORE=hf needs HF_BIN_REPO (e.g. you/colab-llm-bin) in config.env")
        if not os.environ.get("HF_TOKEN"):
            raise RuntimeError("BIN_STORE=hf needs an HF token (~/.config/colab-llm/hf_token on the NUC)")
        self.repo = repo
        self.label = f"hf://{repo}"
        ensure_hf_hub()

    def _download(self, name):
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

        try:
            return hf_hub_download(self.repo, name, local_dir=f"{APP}/bin-cache")
        except EntryNotFoundError:
            return None
        except RepositoryNotFoundError as e:
            raise RuntimeError(
                f"cannot read {self.repo}: create it as a private model repo and give the token access ({e})"
            ) from e

    def get(self, ref, archs):
        if is_full_sha(ref):
            return self._download(server_name(ref, archs))
        pointer = self._download(pointer_name(archs))
        if not pointer:
            return None
        with open(pointer) as f:
            return self._download(f.read().strip())

    def put(self, local_path, sha, archs):
        from huggingface_hub import CommitOperationAdd, HfApi

        name = server_name(sha, archs)
        HfApi().create_commit(
            self.repo,
            operations=[
                CommitOperationAdd(name, local_path),
                CommitOperationAdd(pointer_name(archs), name.encode()),
            ],
            commit_message=f"{name} (llama.cpp {sha[:10]})",
        )


def ensure_hf_hub():
    """huggingface_hub with hf_xet (parallel Xet downloads); installed once per VM."""
    try:
        import hf_xet  # noqa: F401
        import huggingface_hub  # noqa: F401
    except ImportError:
        subprocess.run(["pip", "install", "-q", "-U", "huggingface_hub[hf_xet]"],
                       check=True, stdout=subprocess.DEVNULL)


def bin_store():
    kind = env("BIN_STORE", "hf")
    if kind == "hf":
        return HfBinStore(env("HF_BIN_REPO"))
    if kind == "drive":
        return DriveBinStore(env("DRIVE_MODELS", "/content/drive/MyDrive/models"))
    raise RuntimeError(f"unknown BIN_STORE '{kind}' (hf or drive)")


def build_and_store(store, ref, archs):
    """Build llama-server for `archs`, upload it to `store`, return the local path."""
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
    sha = ref.lower() if is_full_sha(ref) else resolve_llama_sha(ref)
    out = f"{APP}/build-out/{server_name(sha, archs)}"
    if not os.path.exists(out):
        log(f"building llama-server {sha[:10]} for sm {archs} (10-20 min)")
        _, secs = with_heartbeat("build", lambda: build_llama_server(sha, archs, out), every=60)
        log(f"build done in {secs:.0f}s")
    _, secs = with_heartbeat(f"storing build in {store.label}", lambda: store.put(out, sha, archs))
    log(f"stored {server_name(sha, archs)} in {store.label} ({secs:.0f}s, {os.path.getsize(out) / 1e6:.0f} MB)")
    return out


def flush_drive():
    """Wait until everything written to /content/drive is uploaded, then unmount.

    drivefs uploads in the background: a finished copyfile() only means the
    data reached its local cache. Stopping the VM before this returns can
    leave missing or truncated files on Drive.
    """
    if not os.path.ismount("/content/drive"):
        return
    try:
        from google.colab import drive
    except ImportError:
        log("google.colab not available; cannot flush Drive")
        return
    _, secs = with_heartbeat("flushing Drive uploads", drive.flush_and_unmount, every=30)
    log(f"Drive flushed and unmounted ({secs:.0f}s)")


def finish(main):
    """Run main() and print the result marker bin/colab-llm looks for."""
    try:
        main()
        print("COLAB_LLM_RESULT ok", flush=True)
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"COLAB_LLM_RESULT fail: {e}", flush=True)
