# bercail-vision

Analyse automatique des photos de chats de [Bercail](https://bercail.online), 4 fois par jour, sans frais.

1. **GitHub Actions** lit la file d'attente de Bercail (`/api/vision/pending` : adresses de photos publiques seulement).
2. Il lance un **notebook Kaggle** (GPU T4 gratuit, 30 h/semaine) avec [`Qwen/Qwen3-VL-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) (Apache 2.0), en float16.
3. Le notebook décrit chaque chat en JSON (couleurs, motif, poil, zones blanches, yeux, truffe, collier, qualité, éclairage, confiance), avec « incertain » quand ce n'est pas visible.
4. Les résultats reviennent à Bercail (`/api/vision/results`), qui les nettoie et les rend prudents (flash, plusieurs animaux).

La description est **un indice, jamais une identification**. Elle ne retire ni ne baisse jamais une fiche.
Si Kaggle échoue, le serveur de Bercail prend le relais avec des API gratuites (Cloudflare Workers AI, etc.).

## Réglages (Settings → Secrets and variables → Actions)
- Secrets : `BERCAIL_VISION_TOKEN`, `KAGGLE_USERNAME`, `KAGGLE_KEY` (Kaggle → Settings → API → Create New Token ; téléphone vérifié pour le GPU).
- Variable facultative : `BERCAIL_URL` (par défaut `https://app.bercail.online`).

Le dépôt est public (minutes GitHub Actions gratuites et illimitées). Les journaux n'affichent que des comptes.
GitHub met en pause les tâches planifiées d'un dépôt public sans activité depuis 60 jours : un commit par mois suffit.
