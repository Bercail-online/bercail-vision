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
import base64
src = open("kernel/analyse.py", encoding="utf-8").read()
assert src.count("__MANIFEST_B64__") == 1, "le gabarit doit contenir une seule fois __MANIFEST_B64__"
src = src.replace("__MANIFEST_B64__", base64.b64encode(json.dumps(m, ensure_ascii=False).encode("utf-8")).decode("ascii"))
# Kaggle exécute un « script » comme un notebook : pas de __file__, donc impossible de se relancer sur chaque GPU.
# Le code envoyé est un petit lanceur qui dépose le vrai script sur le disque, le lance et relaie son journal.
LANCEUR = """# Bercail-vision : lanceur Kaggle (le vrai script est encodé plus bas ; aucun secret).
import base64, os, subprocess, sys
p = "/tmp/bercail_analyse.py"
open(p, "w", encoding="utf-8").write(base64.b64decode("%s").decode("utf-8"))
pr = subprocess.Popen([sys.executable, "-u", p], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for ligne in pr.stdout:
    print(ligne, end="", flush=True)
if pr.wait():
    raise SystemExit(f"analyse.py a échoué (code {pr.returncode})")
""" % base64.b64encode(src.encode("utf-8")).decode("ascii")
open("build/analyse.py", "w", encoding="utf-8").write(LANCEUR)
user = os.environ["KAGGLE_USERNAME"]
json.dump({"id": f"{user}/bercail-vision", "title": "bercail-vision", "code_file": "analyse.py", "language": "python",
           "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
           "dataset_sources": [], "competition_sources": [], "kernel_sources": []},
          open("build/kernel-metadata.json", "w"), indent=1)
if gh: open(gh, "a").write("rien=0\n")
