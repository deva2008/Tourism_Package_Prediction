import os, textwrap
from huggingface_hub import HfApi, upload_file

HF_USERNAME   = os.environ.get("HF_USERNAME")
HF_SPACE_REPO = os.environ.get("HF_SPACE_REPO")
TOKEN         = os.environ.get("HF_TOKEN")

assert HF_USERNAME and HF_SPACE_REPO, "Set HF_USERNAME and HF_SPACE_REPO in env"
SPACE_ID = f"{HF_USERNAME}/{HF_SPACE_REPO}"

def ensure_space(api: HfApi):
    """
    Try to create the Space as Streamlit. If the hub rejects the 'sdk' field or it already
    exists, continue. As a last resort, create a bare space (no sdk) and configure via README.
    """
    try:
        api.create_repo(repo_id=SPACE_ID, repo_type="space", space_sdk="streamlit",
                        exist_ok=True, private=False, token=TOKEN)
        print(f"[OK] Space ensured with SDK=streamlit: {SPACE_ID}")
        return
    except Exception as e:
        print(f"[WARN] create_repo with SDK failed: {e}")

    # If it exists already, proceed
    try:
        info = api.repo_info(repo_id=SPACE_ID, repo_type="space", token=TOKEN)
        print(f"[OK] Space exists: {SPACE_ID} (proceeding to upload files)")
        return
    except Exception:
        pass

    # Fallback: create minimal space (no sdk), we'll set sdk via README
    try:
        api.create_repo(repo_id=SPACE_ID, repo_type="space", exist_ok=True,
                        private=False, token=TOKEN)
        print(f"[OK] Space ensured (minimal): {SPACE_ID}")
    except Exception as e:
        print(f"[WARN] minimal create_repo failed too: {e} (continuing to upload files)")

def upload_app_files(api: HfApi):
    # Upload app.py
    upload_file(
        path_or_fileobj="app_streamlit/app.py",
        path_in_repo="app.py",
        repo_id=SPACE_ID, repo_type="space", token=TOKEN,
    )
    print("[OK] Uploaded app.py")

    # Upload requirements for Space
    upload_file(
        path_or_fileobj="space_requirements.txt",
        path_in_repo="requirements.txt",
        repo_id=SPACE_ID, repo_type="space", token=TOKEN,
    )
    print("[OK] Uploaded requirements.txt")

    # Ensure README sets sdk/app_file so the Space runs Streamlit
    readme = textwrap.dedent("""\
    ---
    title: Wellness Tourism Purchase Predictor
    emoji: 🧘
    colorFrom: indigo
    colorTo: green
    sdk: streamlit
    app_file: app.py
    pinned: false
    license: other
    ---
    # Wellness Tourism Purchase Predictor
    Loads the best model from HF Hub and predicts purchase probability for the Wellness package.
    """).strip()+"\n"

    with open("SPACE_README.md","w") as f: f.write(readme)
    upload_file(
        path_or_fileobj="SPACE_README.md",
        path_in_repo="README.md",
        repo_id=SPACE_ID, repo_type="space", token=TOKEN,
    )
    print("[OK] Uploaded README.md with sdk/app_file")

def main():
    api = HfApi(token=TOKEN)
    ensure_space(api)
    upload_app_files(api)
    print(f"[DONE] Space updated: https://huggingface.co/spaces/{SPACE_ID}")

if __name__ == "__main__":
    main()
