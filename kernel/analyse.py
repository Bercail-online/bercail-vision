# Bercail : analyse des photos de chats (notebook Kaggle, GPU T4). Licence du modèle : Apache 2.0.
# Ce fichier est rempli par GitHub Actions : la file d'attente (adresses de photos publiques seulement) remplace __MANIFEST__.
# Il n'a besoin d'aucun secret. Il écrit /kaggle/working/results.jsonl au fil de l'eau (une coupure ne perd qu'une photo).
import io, json, os, re, subprocess, sys, time

MANIFEST = json.loads(r'''__MANIFEST__''')
MODELE = os.environ.get("BV_MODELE", "Qwen/Qwen3-VL-4B-Instruct")
LIMITE_S = float(os.environ.get("BV_LIMITE_S", str(3 * 3600)))      # économise le quota GPU (30 h/semaine)
COTE = 768                                                         # côté max de l'image envoyée au modèle
SORTIE = os.environ.get("BV_SORTIE", "/kaggle/working/results.jsonl")
t0 = time.time()

def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

if os.path.exists("/kaggle"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.57", "accelerate"], check=False)
import requests, torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

cuda = torch.cuda.is_available()
dtype = torch.float16 if cuda else torch.bfloat16                  # le T4 ne gère pas bfloat16
log(f"modèle {MODELE} ; {'GPU ' + torch.cuda.get_device_name(0) if cuda else 'CPU'} ; {len(MANIFEST['items'])} photos")
proc = AutoProcessor.from_pretrained(MODELE)
model = AutoModelForImageTextToText.from_pretrained(MODELE, dtype=dtype, device_map="cuda:0" if cuda else "cpu").eval()
CONSIGNE = MANIFEST["consigne"]; VERSION = MANIFEST["prompt_version"]
FOURNISSEUR = "kaggle:" + MODELE.split("/")[-1]

def image(url):
    r = requests.get(url, timeout=30, headers={"User-Agent": "Bercail-vision/1 (+https://bercail.online)"}); r.raise_for_status()
    im = Image.open(io.BytesIO(r.content)).convert("RGB"); im.thumbnail((COTE, COTE)); return im

def json_de(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m: raise ValueError("pas de JSON dans la réponse")
    return json.loads(m.group(0))

@torch.no_grad()
def analyser(im):
    msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": CONSIGNE}]}]
    x = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
    out = model.generate(**x, max_new_tokens=350, do_sample=False)
    return proc.batch_decode(out[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0]

faits = set()
if os.path.exists(SORTIE):
    faits = {json.loads(l)["photo_id"] for l in open(SORTIE) if l.strip()}
n_ok = n_err = 0
with open(SORTIE, "a") as f:
    for i, it in enumerate(MANIFEST["items"]):
        if it["photo_id"] in faits: continue
        if time.time() - t0 > LIMITE_S: log("limite de temps atteinte : le reste attendra le prochain passage"); break
        rec = {"photo_id": it["photo_id"], "sha256": it.get("sha256"), "provider": FOURNISSEUR, "prompt_version": VERSION}
        t = time.time()
        try:
            txt = analyser(image(it["url"]))
            try: rec["traits"] = json_de(txt)
            except Exception:                                        # une seconde chance, plus courte et plus stricte
                rec["traits"] = json_de(analyser(image(it["url"])))
            n_ok += 1
        except Exception as e:
            rec["erreur"] = f"{type(e).__name__}: {str(e)[:200]}"; n_err += 1
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
        if i < 3 or i % 25 == 0: log(f"{i + 1}/{len(MANIFEST['items'])} {it['photo_id']} {time.time() - t:.1f} s")
log(f"fini : {n_ok} analysées, {n_err} erreurs, {time.time() - t0:.0f} s")
