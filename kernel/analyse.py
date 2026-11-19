# Bercail : analyse des photos de chats (notebook Kaggle, GPU T4 ×2). Licence du modèle : Apache 2.0.
# GitHub Actions remplit la file d'attente (adresses de photos publiques seulement, encodées en base64 plus bas).
# Aucun secret. Chaque GPU traite sa part de la file ; les résultats vont dans /kaggle/working/results.jsonl.
#
# Version 2 (19 nov. 2026), après relecture de la première exécution :
#  - la seconde chance est vraiment différente (consigne plus stricte et réponse amorcée par « { ») ;
#  - la réponse est vérifiée contre le schéma reçu du serveur (clés manquantes : seconde chance ; valeur hors liste : incertain) ;
#  - l'empreinte SHA-256 est calculée sur les octets téléchargés, et une photo changée est signalée, pas analysée ;
#  - plus de « reprise » (Kaggle repart d'un dossier vide) : une photo en erreur reste en attente pour le prochain passage ;
#  - les deux T4 travaillent en parallèle, chacune avec sa copie du modèle ;
#  - transformers est épinglé à une plage connue.
import base64, hashlib, io, json, os, re, subprocess, sys, time

MANIFEST = json.loads(base64.b64decode("__MANIFEST_B64__").decode("utf-8"))
MODELE = os.environ.get("BV_MODELE", "Qwen/Qwen3-VL-4B-Instruct")
LIMITE_S = float(os.environ.get("BV_LIMITE_S", str(3 * 3600)))      # économise le quota GPU (30 h/semaine)
COTE = 768                                                         # côté max de l'image envoyée au modèle
DOSSIER = os.environ.get("BV_DOSSIER", "/kaggle/working" if os.path.exists("/kaggle") else ".")
SORTIE = os.path.join(DOSSIER, "results.jsonl")
T0 = float(os.environ.get("BV_T0", time.time()))
RANG = int(os.environ.get("BV_RANG", "-1"))                        # -1 : chef d'orchestre ; 0, 1… : un GPU


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[{'chef' if RANG < 0 else 'gpu' + str(RANG)}]", *a, flush=True)


# ---------------------------------------------------------------- chef d'orchestre : installe, partage, rassemble
SCRIPT = globals().get("__file__") or (sys.argv[0] if sys.argv and sys.argv[0] else "")
SEUL = RANG < 0 and not os.path.isfile(SCRIPT)                     # pas de fichier à relancer : un seul travailleur, ici
if SEUL:
    RANG = 0
if RANG < 0:
    if os.path.exists("/kaggle"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.57.1,<4.60", "accelerate>=1.0"], check=False)
    import torch
    n_gpu = torch.cuda.device_count()
    n = max(1, min(n_gpu, 2, len(MANIFEST["items"])))
    log(f"{len(MANIFEST['items'])} photos ; {n_gpu} GPU ; {n} travailleur(s)")
    procs = []
    for r in range(n):
        env = dict(os.environ, BV_RANG=str(r), BV_NB=str(n), BV_T0=str(T0), CUDA_VISIBLE_DEVICES=str(r) if n_gpu else "")
        procs.append(subprocess.Popen([sys.executable, os.path.abspath(SCRIPT)], env=env))
    codes = [p.wait() for p in procs]
    rows = []
    for r in range(n):
        part = os.path.join(DOSSIER, f"results.part{r}.jsonl")
        if os.path.exists(part):
            rows += [l for l in open(part, encoding="utf-8") if l.strip()]
            os.remove(part)
    with open(SORTIE, "w", encoding="utf-8") as f:
        f.writelines(rows)
    ok = sum(1 for l in rows if '"traits"' in l)
    log(f"fini : {ok} analysées, {len(rows) - ok} en erreur, codes {codes}, {time.time() - T0:.0f} s")
    sys.exit(0)

# ---------------------------------------------------------------- travailleur : un GPU, sa part de la file
import requests, torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

NB = 1 if SEUL else int(os.environ.get("BV_NB", "1"))
if SEUL and os.path.exists("/kaggle"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.57.1,<4.60", "accelerate>=1.0"], check=False)
cuda = torch.cuda.is_available()
dtype = torch.float16 if cuda else torch.bfloat16                  # le T4 ne gère pas bfloat16
proc = AutoProcessor.from_pretrained(MODELE)
model = AutoModelForImageTextToText.from_pretrained(MODELE, dtype=dtype, device_map="cuda:0" if cuda else "cpu").eval()
CONSIGNE = MANIFEST["consigne"]
VERSION = MANIFEST["prompt_version"]
SCHEMA = MANIFEST.get("schema") or {}
CLES = SCHEMA.get("keys") or []
FOURNISSEUR = "kaggle:" + MODELE.split("/")[-1]
STRICTE = (CONSIGNE + "\n\nIMPORTANT : ta réponse précédente n'était pas utilisable. Réponds par UN SEUL objet JSON, sans texte, "
           "sans ``` ni commentaire, avec exactement ces clés : " + ", ".join(CLES) + ". Pour toute valeur non visible : incertain.")
log(f"prêt en {time.time() - T0:.0f} s ; {'GPU ' + torch.cuda.get_device_name(0) if cuda else 'CPU'}")


def telecharger(it):
    r = requests.get(it["url"], timeout=30, headers={"User-Agent": "Bercail-vision/2 (+https://bercail.online/agents.html)"})
    r.raise_for_status()
    h = hashlib.sha256(r.content).hexdigest()
    if it.get("sha256") and h != it["sha256"]:
        raise ValueError("photo changée sur le serveur depuis la mise en file : non analysée")
    im = Image.open(io.BytesIO(r.content)).convert("RGB")
    im.thumbnail((COTE, COTE))
    return im, h


def json_de(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        raise ValueError("pas de JSON dans la réponse")
    return json.loads(m.group(0))


def verifier(t):
    """Clés manquantes : ValueError (on tente la seconde chance). Valeur hors liste : ramenée à « incertain », comptée."""
    if not isinstance(t, dict):
        raise ValueError("la réponse n'est pas un objet")
    manque = [k for k in CLES if k not in t]
    if manque:
        raise ValueError("clés manquantes : " + ", ".join(manque[:5]))
    hors = 0
    for k, permis in (SCHEMA.get("enums") or {}).items():
        if str(t.get(k, "")).strip().lower() not in permis:
            t[k] = "incertain" if "incertain" in permis else t.get(k); hors += 1
    for k, permis in (SCHEMA.get("lists") or {}).items():
        v = t.get(k)
        v = v if isinstance(v, list) else [v]
        bons = [str(x).strip().lower() for x in v if str(x).strip().lower() in permis]
        hors += len(v) - len(bons)
        t[k] = bons or ["incertain"]
    return t, hors


@torch.no_grad()
def generer(im, consigne, amorce=""):
    msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": consigne}]}]
    texte = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + amorce
    x = proc(text=[texte], images=[im], return_tensors="pt").to(model.device)
    out = model.generate(**x, max_new_tokens=400, do_sample=False)
    return amorce + proc.batch_decode(out[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0]


part = MANIFEST["items"][RANG::NB]
sortie = os.path.join(DOSSIER, f"results.part{RANG}.jsonl")
n_ok = n_err = n_2e = n_hors = 0
with open(sortie, "w", encoding="utf-8") as f:
    for i, it in enumerate(part):
        if time.time() - T0 > LIMITE_S:
            log("limite de temps atteinte : le reste attendra le prochain passage")
            break
        rec = {"photo_id": it["photo_id"], "sha256": it.get("sha256"), "provider": FOURNISSEUR, "prompt_version": VERSION}
        t = time.time()
        try:
            im, rec["sha256"] = telecharger(it)
            try:
                traits, hors = verifier(json_de(generer(im, CONSIGNE)))
            except Exception:                         # seconde chance : consigne plus stricte, réponse amorcée par « { »
                traits, hors = verifier(json_de(generer(im, STRICTE, amorce="{")))
                n_2e += 1
            rec["traits"] = traits
            n_hors += hors
            n_ok += 1
        except Exception as e:
            rec["erreur"] = f"{type(e).__name__}: {str(e)[:200]}"
            n_err += 1
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        if i < 3 or i % 25 == 0:
            log(f"{i + 1}/{len(part)} {it['photo_id']} {time.time() - t:.1f} s")
log(f"fini : {n_ok} analysées ({n_2e} en seconde chance, {n_hors} valeurs hors liste ramenées à incertain), {n_err} erreurs")
if SEUL:
    os.replace(sortie, SORTIE)
