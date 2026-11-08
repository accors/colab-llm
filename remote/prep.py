# colab-llm: prep step. Runs inside a Colab CPU session via `colab exec`.
#
# 1. Downloads the selected GGUF files from Hugging Face to local disk, then
#    copies them to Google Drive (one file at a time, so local disk never needs
#    the whole model at once).
# 2. If nvcc is available, builds llama-server (CUDA) and caches the binary on
#    Drive so the GPU session can skip the build.
#
# Configuration arrives as environment variables set by `colab exec --env`.
# The last line printed is always `COLAB_LLM_RESULT ok` or
# `COLAB_LLM_RESULT fail: <reason>`; bin/colab-llm looks for it.

def select_files(api, repo, quant, want_mmproj):
    info = api.model_info(repo, files_metadata=True)
    files = {}
    for f in info.siblings:
        name = f.rfilename
        if name.startswith(f"{quant}/") and name.endswith(".gguf"):
            files[name] = f.size
        elif want_mmproj and name.startswith("mmproj") and name.endswith(".gguf"):
            files[name] = f.size
    return info.sha, files


class DriveCopy(threading.Thread):
    """Copy one staged file to Drive in the background, then delete the staged copy."""

    def __init__(self, src, target, size):
        super().__init__(daemon=True)
        self.src, self.target, self.size = src, target, size
        self.tmp = target + ".partial"
        self.error = None

    def run(self):
        try:
            os.makedirs(os.path.dirname(self.target), exist_ok=True)
            t0 = time.time()
            shutil.copyfile(self.src, self.tmp)
            os.replace(self.tmp, self.target)
            secs = time.time() - t0
            log(f"copied to Drive: {os.path.basename(self.target)} in {secs:.0f}s "
                f"({self.size / max(secs, 1) / 1e6:.0f} MB/s)")
            os.remove(self.src)
        except BaseException as e:  # noqa: BLE001 - surfaced by check()
            self.error = e

    def check(self):
        if self.error:
            raise RuntimeError(f"copy to Drive failed for {self.target}: {self.error}")


def sync_repo(api, repo, quant, want_mmproj, drive_root, stage_root, extra_files=()):
    from huggingface_hub import hf_hub_download

    revision, files = select_files(api, repo, quant, want_mmproj)
    for name in extra_files:
        info = api.model_info(repo, files_metadata=True)
        sizes = {s.rfilename: s.size for s in info.siblings}
        if name not in sizes:
            raise RuntimeError(f"{repo} has no file {name}")
        files[name] = sizes[name]
    if not files:
        raise RuntimeError(f"no files matched quant '{quant}' in {repo}")

    dest = os.path.join(drive_root, repo_dir_name(repo))
    os.makedirs(dest, exist_ok=True)
    log(f"{repo} @ {revision[:10]}: {len(files)} files, {gb(sum(files.values()))}")

    todo = []
    for name, size in sorted(files.items()):
        target = os.path.join(dest, name)
        if os.path.exists(target) and os.path.getsize(target) == size:
            log(f"already on Drive: {name}")
        else:
            todo.append((name, size))

    need = sum(size for _, size in todo)
    if need:
        free = shutil.disk_usage(drive_root).free
        log(f"Drive free: {gb(free)}, still needed: {gb(need)}")
        if free < need:
            raise RuntimeError(f"not enough Drive space: need {gb(need)}, free {gb(free)}")

    # Pipeline: while file N is being copied to Drive, file N+1 downloads.
    # At most two files sit on local disk at once (~84 GB for IQ3_S).
    copier = None
    for name, size in todo:
        stage = os.path.join(stage_root, repo_dir_name(repo))
        os.makedirs(stage, exist_ok=True)
        local_path, secs = with_heartbeat(
            f"download {name}",
            lambda: hf_hub_download(repo, name, revision=revision, local_dir=stage),
            watch_path=stage,
        )
        log(f"downloaded {name}: {gb(size)} in {secs:.0f}s ({size / max(secs, 1) / 1e6:.0f} MB/s)")
        if copier:
            copier.join()
            copier.check()
        copier = DriveCopy(local_path, os.path.join(dest, name), size)
        copier.start()
    if copier:
        with_heartbeat("waiting for last Drive copy", copier.join, watch_path=copier.tmp)
        copier.check()

    manifest_path = os.path.join(dest, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    manifest.update({"repo": repo, "revision": revision})
    manifest.setdefault("files", {}).update(files)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1)
    log(f"manifest written: {manifest_path}")


def main():
    try:
        prep()
    finally:
        flush_drive()


def prep():
    repo = env("HF_REPO")
    quant = env("QUANT", "IQ3_S")
    drive_root = env("DRIVE_MODELS", "/content/drive/MyDrive/models")
    want_mmproj = env("MMPROJ", "0") == "1"
    mtp_repo, mtp_file = env("MTP_REPO"), env("MTP_FILE")
    build_mode = env("BUILD", "auto")  # auto | yes | no
    llama_ref = env("LLAMA_CPP_REF", "master")
    archs = env("CUDA_ARCHS", "80")

    if not repo:
        raise RuntimeError("HF_REPO is not set")
    if not os.path.ismount("/content/drive"):
        raise RuntimeError("Google Drive is not mounted (run `colab drivemount` first)")

    subprocess.run(
        ["pip", "install", "-q", "-U", "huggingface_hub[hf_xet]"],
        check=True, stdout=subprocess.DEVNULL,
    )
    from huggingface_hub import HfApi

    api = HfApi()
    stage_root = f"{APP}/stage"
    sync_repo(api, repo, quant, want_mmproj, drive_root, stage_root)
    if mtp_repo and mtp_file:
        sync_repo(api, mtp_repo, "__none__", False, drive_root, stage_root, extra_files=[mtp_file])

    if build_mode == "no":
        log("BUILD=no, skipping llama-server build")
        return
    if not shutil.which("nvcc") and not os.path.exists("/usr/local/cuda/bin/nvcc"):
        msg = "nvcc not found on this runtime"
        if build_mode == "yes":
            raise RuntimeError(msg)
        log(f"{msg}; llama-server will be built in the GPU session instead")
        return
    force = env("FORCE_BUILD", "0") == "1"
    cached = find_cached_server(drive_root, llama_ref, archs)
    if cached and not force:
        log(f"llama-server already cached: {cached} (use `colab-llm prep --rebuild` to refresh)")
        return
    try:
        build_and_cache(drive_root, llama_ref, archs)
    except Exception as e:  # noqa: BLE001
        if build_mode == "yes":
            raise
        log(f"build on CPU runtime failed ({e}); the GPU session will build instead")

finish(main)
