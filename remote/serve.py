# colab-llm: serve step. Runs inside the Colab GPU session via `colab exec`
# (bin/colab-llm prepends common.py).
#
# 1. Checks the GPU (refuses to continue below MIN_VRAM_GIB, so a 40 GB A100
#    does not quietly burn compute units).
# 2. Puts the model on local disk: copies it from Drive (SOURCE=drive) or
#    downloads it from Hugging Face (SOURCE=hf). Prints throughput either way.
# 3. Gets llama-server: Drive cache, otherwise builds it here and caches it.
# 4. Starts llama-server detached, bound to 127.0.0.1 only; it is reached
#    through `ssh -L` from the NUC, never from the internet.


def gpu_info():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()[0]
    name, mem_mib, cap = [x.strip() for x in out.split(",")]
    return name, int(mem_mib) / 1024, cap.replace(".", "")


def copy_if_needed(src, dst, size):
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        log(f"already on local disk: {os.path.basename(dst)}")
        return 0.0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".partial"
    _, secs = with_heartbeat(f"copy {os.path.basename(dst)}", lambda: shutil.copyfile(src, tmp), watch_path=tmp)
    os.replace(tmp, dst)
    log(f"copied {os.path.basename(dst)}: {gb(size)} in {secs:.0f}s ({size / max(secs, 1) / 1e6:.0f} MB/s)")
    return secs


def fetch_from_drive(drive_root, repo, local_root, names):
    """Copy files listed in the repo's Drive manifest. Returns {name: local_path}."""
    src_dir = os.path.join(drive_root, repo_dir_name(repo))
    manifest_path = os.path.join(src_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise RuntimeError(f"no manifest at {manifest_path}: run `colab-llm prep` first")
    with open(manifest_path) as f:
        manifest = json.load(f)
    out = {}
    for name in names(manifest["files"]):
        size = manifest["files"][name]
        src = os.path.join(src_dir, name)
        if not os.path.exists(src) or os.path.getsize(src) != size:
            raise RuntimeError(f"{src} missing or incomplete on Drive: rerun `colab-llm prep`")
        dst = os.path.join(local_root, repo_dir_name(repo), name)
        copy_if_needed(src, dst, size)
        out[name] = dst
    return out


def fetch_from_hf(repo, local_root, names):
    ensure_hf_hub()
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().model_info(repo, files_metadata=True)
    files = {s.rfilename: s.size for s in info.siblings}
    dest = os.path.join(local_root, repo_dir_name(repo))
    out = {}
    for name in names(files):
        dst = os.path.join(dest, name)
        if os.path.exists(dst) and os.path.getsize(dst) == files[name]:
            log(f"already on local disk: {name}")
        else:
            _, secs = with_heartbeat(
                f"download {name}",
                lambda: hf_hub_download(repo, name, revision=info.sha, local_dir=dest),
                watch_path=dest,
            )
            log(f"downloaded {name}: {gb(files[name])} in {secs:.0f}s ({files[name] / max(secs, 1) / 1e6:.0f} MB/s)")
        out[name] = dst
    return out


def model_files(quant, want_mmproj):
    def pick(files):
        chosen = [n for n in files if n.startswith(f"{quant}/") and n.endswith(".gguf")]
        if want_mmproj:
            chosen += [n for n in files if n.startswith("mmproj") and n.endswith(".gguf")]
        if not any(n.startswith(f"{quant}/") for n in chosen):
            raise RuntimeError(f"no '{quant}/*.gguf' files found")
        return sorted(chosen)

    return pick


def get_server(ref, archs, gpu_arch):
    target = f"{APP}/bin/llama-server"
    arch_list = [a.strip() for a in archs.replace(";", ",").split(",") if a.strip()]
    if gpu_arch not in arch_list:
        log(f"GPU is sm{gpu_arch}, CUDA_ARCHS={archs}: building for sm{gpu_arch} instead")
        archs = gpu_arch
    store = bin_store()
    cached = store.get(ref, archs)
    if cached:
        log(f"using llama-server from {store.label}: {os.path.basename(cached)}")
    else:
        log(f"no llama-server build in {store.label}; building on this GPU "
            "(cheaper next time: colab-llm prep --gpu T4)")
        cached = build_and_store(store, ref, archs)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copyfile(cached, target)
    os.chmod(target, 0o755)
    version = subprocess.run([target, "--version"], capture_output=True, text=True)
    log("llama-server " + (version.stdout + version.stderr).strip().splitlines()[0])
    return target


def get_chat_template(repo, filename, revision):
    """Download a Jinja chat template from HF; returns (path, commit sha)."""
    ensure_hf_hub()
    from huggingface_hub import HfApi, hf_hub_download

    sha = HfApi().model_info(repo, revision=revision or None).sha
    path = hf_hub_download(repo, filename, revision=sha, local_dir=f"{APP}/templates/{sha[:10]}")
    with open(path, errors="replace") as f:
        head = f.read(4000)
    version = ""
    for line in head.splitlines():
        if "template_version" in line and "=" in line:
            version = line.split("=", 1)[1].strip(" %}\"'")
            break
    log(f"chat template: {repo}/{filename} @ {sha[:10]}" + (f" ({version})" if version else "")
        + ("" if revision else " - not pinned; set CHAT_TEMPLATE_REV to this sha to pin it"))
    return path, sha


def stop_existing_server():
    # The bracket keeps pkill from matching its own command line.
    subprocess.run(["pkill", "-f", "[c]olab-llm/bin/llama-server"])
    for _ in range(30):
        if subprocess.run(["pgrep", "-f", "[c]olab-llm/bin/llama-server"], capture_output=True).returncode != 0:
            return
        time.sleep(1)
    subprocess.run(["pkill", "-9", "-f", "[c]olab-llm/bin/llama-server"])


def wait_healthy(proc, port, log_path, timeout_s):
    import urllib.request

    t0 = time.time()
    last = 0.0
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {proc.returncode}:\n" + tail(log_path, 40))
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:  # noqa: BLE001 - still loading
            pass
        if time.time() - last > 15:
            lines = tail(log_path, 1).strip()
            log(f"loading... {time.time() - t0:.0f}s | {lines[-110:]}")
            last = time.time()
        time.sleep(2)
    raise RuntimeError(f"llama-server not healthy after {timeout_s}s:\n" + tail(log_path, 40))


def tail(path, n):
    try:
        with open(path, errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


def main():
    import shlex

    repo = env("HF_REPO")
    quant = env("QUANT", "IQ3_S")
    source = env("SOURCE", "drive")
    drive_root = env("DRIVE_MODELS", "/content/drive/MyDrive/models")
    want_mmproj = env("MMPROJ", "0") == "1"
    mtp_repo, mtp_file = env("MTP_REPO"), env("MTP_FILE")
    min_vram = float(env("MIN_VRAM_GIB", "75"))
    port = int(env("REMOTE_PORT", "8000"))
    ctx = env("CTX", "262144")
    parallel = env("PARALLEL", "1")
    alias = env("ALIAS", "qwen3.8-flash-next")
    extra = shlex.split(env("EXTRA_ARGS"))
    load_timeout = int(env("LOAD_TIMEOUT", "1800"))
    key_file = f"{APP}/api_key"

    name, vram, arch = gpu_info()
    log(f"GPU: {name}, {vram:.1f} GiB, sm{arch}")
    if vram < min_vram:
        raise RuntimeError(
            f"got {name} with {vram:.1f} GiB < MIN_VRAM_GIB={min_vram:.0f}. "
            "For A100 80GB the session must be created with --high-mem."
        )
    if not os.path.exists(key_file):
        raise RuntimeError(f"{key_file} missing (bin/colab-llm uploads it before serve)")
    drive_ok = os.path.ismount("/content/drive")
    if source == "drive" and not drive_ok:
        raise RuntimeError("SOURCE=drive but Google Drive is not mounted")

    local_root = f"{APP}/models"
    t0 = time.time()
    pick = model_files(quant, want_mmproj)
    if source == "drive":
        files = fetch_from_drive(drive_root, repo, local_root, pick)
    elif source == "hf":
        files = fetch_from_hf(repo, local_root, pick)
    else:
        raise RuntimeError(f"unknown SOURCE '{source}' (drive or hf)")
    mtp_path = None
    if mtp_repo and mtp_file:
        only = lambda fs: [mtp_file]  # noqa: E731
        got = fetch_from_drive(drive_root, mtp_repo, local_root, only) if source == "drive" \
            else fetch_from_hf(mtp_repo, local_root, only)
        mtp_path = got[mtp_file]
    log(f"model files ready in {time.time() - t0:.0f}s (SOURCE={source})")

    main_shard = next(p for n, p in sorted(files.items()) if n.startswith(f"{quant}/"))
    mmproj = next((p for n, p in files.items() if n.startswith("mmproj")), None)

    server = get_server(env("LLAMA_CPP_REF", "master"), env("CUDA_ARCHS", "80"), arch)
    stop_existing_server()

    cmd = [
        server, "-m", main_shard, "--alias", alias, "--api-key-file", key_file,
        "-ngl", "99", "-c", ctx, "-np", parallel,
        # The n-gram table (shard 2) stays memory-mapped on local disk and is
        # read row by row on demand instead of occupying ~29 GB of VRAM/RAM.
        "-lm", "mmap", "--lazy-mode", "on",
        # --jinja must come before --chat-template-file
        "--jinja", "--metrics", "--host", "127.0.0.1", "--port", str(port),
    ]
    template_info = None
    if env("CHAT_TEMPLATE_REPO"):
        tpl, tpl_sha = get_chat_template(
            env("CHAT_TEMPLATE_REPO"), env("CHAT_TEMPLATE_FILE", "chat_template.jinja"), env("CHAT_TEMPLATE_REV")
        )
        cmd += ["--chat-template-file", tpl]
        template_info = {"repo": env("CHAT_TEMPLATE_REPO"), "sha": tpl_sha}
    if env("REASONING_FORMAT"):
        cmd += ["--reasoning-format", env("REASONING_FORMAT")]
    if mmproj:
        cmd += ["--mmproj", mmproj]
    if mtp_path:
        cmd += ["--spec-draft-model", mtp_path, "--spec-type", "draft-mtp"]
    cmd += extra

    os.makedirs(f"{APP}/logs", exist_ok=True)
    log_path = f"{APP}/logs/llama-server.log"
    log("starting: " + " ".join(shlex.quote(c) for c in cmd))
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    secs = wait_healthy(proc, port, log_path, load_timeout)
    used = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"], text=True
    ).strip()
    log(f"llama-server healthy after {secs:.0f}s, pid {proc.pid}, VRAM {used}")
    with open(f"{APP}/server.json", "w") as f:
        json.dump({"pid": proc.pid, "started": time.time(), "port": port, "alias": alias,
                   "chat_template": template_info, "cmd": cmd}, f)


finish(main)
