"""Renvoie les résultats du notebook à Bercail par lots de 100. N'affiche que des comptes (dépôt public)."""
import json, os, sys, urllib.request
url = os.environ.get("BERCAIL_URL", "https://app.bercail.online").rstrip("/")
f = sys.argv[1] if len(sys.argv) > 1 else "out/results.jsonl"
rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()] if os.path.exists(f) else []
ok = [r for r in rows if "traits" in r]; err = len(rows) - len(ok)
tot = {"enregistres": 0, "refuses": 0}
for i in range(0, len(ok), 100):
    body = json.dumps({"results": ok[i:i + 100]}).encode()
    req = urllib.request.Request(f"{url}/api/vision/results", data=body, method="POST",
                                 headers={"Authorization": "Bearer " + os.environ["BERCAIL_VISION_TOKEN"],
                                          "Content-Type": "application/json", "User-Agent": "bercail-vision"})
    r = json.load(urllib.request.urlopen(req, timeout=60))
    tot["enregistres"] += r.get("enregistres", 0); tot["refuses"] += r.get("refuses", 0)
print(f"résultats : {len(rows)} lignes, {err} erreurs d'analyse, {tot['enregistres']} enregistrés, {tot['refuses']} refusés")
if rows and tot["enregistres"] == 0: sys.exit(1)
