"""Lit la file d'attente de Bercail et prépare le notebook Kaggle (aucun secret dedans). Sort 0 avec RIEN=1 si rien à faire."""
import json, os, sys, urllib.request
url = os.environ.get("BERCAIL_URL", "https://app.bercail.online").rstrip("/")
req = urllib.request.Request(f"{url}/api/vision/pending?limit={os.environ.get('BV_LOT', '600')}",
                             headers={"Authorization": "Bearer " + os.environ["BERCAIL_VISION_TOKEN"], "User-Agent": "bercail-vision"})
m = json.load(urllib.request.urlopen(req, timeout=60))
print(f"photos en attente : {m['count']} (consigne {m['prompt_version']})")
gh = os.environ.get("GITHUB_OUTPUT")
if m["count"] == 0:
    if gh: open(gh, "a").write("rien=1\n")
    sys.exit(0)
os.makedirs("build", exist_ok=True)
src = open("kernel/analyse.py", encoding="utf-8").read().replace("__MANIFEST__", json.dumps(m, ensure_ascii=False).replace("'''", "'\\''"))
open("build/analyse.py", "w", encoding="utf-8").write(src)
user = os.environ["KAGGLE_USERNAME"]
json.dump({"id": f"{user}/bercail-vision", "title": "bercail-vision", "code_file": "analyse.py", "language": "python",
           "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
           "dataset_sources": [], "competition_sources": [], "kernel_sources": []},
          open("build/kernel-metadata.json", "w"), indent=1)
if gh: open(gh, "a").write("rien=0\n")
