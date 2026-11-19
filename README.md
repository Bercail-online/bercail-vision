# bercail-vision

Analyse automatique des photos de chats de [Bercail](https://bercail.online), 4 fois par jour, sans frais.

1. **GitHub Actions** lit la file d'attente de Bercail (`/api/vision/pending` : adresses de photos publiques seulement).
2. Il lance un **notebook Kaggle** (GPU T4 gratuit, 30 h/semaine) avec [`Qwen/Qwen3-VL-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) (Apache 2.0), en float16.
3. Le notebook décrit chaque chat en JSON (couleurs, motif, poil, zones blanches, yeux, truffe, collier, qualité, éclairage, confiance), avec « incertain » quand ce n'est pas visible.
4. Les résultats reviennent à Bercail (`/api/vision/results`), qui les nettoie et les rend prudents (flash, plusieurs animaux).

La description est **un indice, jamais une identification**. Elle ne retire ni ne baisse jamais une fiche.
Si Kaggle échoue, le serveur de Bercail prend le relais avec des API gratuites (Cloudflare Workers AI, etc.).

## Notebook, version 2
- Les deux GPU T4 travaillent en parallèle (une copie du modèle chacune, la file partagée en deux).
- La réponse du modèle est vérifiée contre le schéma envoyé par Bercail ; s'il manque une clé, une seconde chance part avec une consigne plus stricte et une réponse amorcée par `{`.
- L'empreinte SHA-256 est recalculée sur la photo téléchargée : une photo changée depuis la mise en file n'est pas analysée.
- Une photo en erreur n'est jamais marquée faite : elle revient au passage suivant.

Notebook, version 2.1
Kaggle exécute un « script » comme un notebook (pas de __file__) : le code envoyé est un petit lanceur qui dépose le vrai script dans /tmp, le lance et relaie son journal. Sans lui, la v2 relançait ipykernel et restait bloquée.
Schéma p1 intégré si le serveur n'envoie pas « schema » ; le chef télécharge le modèle une fois avant les deux GPU.
GitHub Actions attend de voir le noyau en file ou en cours avant d'accepter « terminé » (sinon il lisait la version précédente).

## Réglages (Settings → Secrets and variables → Actions)
- Secrets : `BERCAIL_VISION_TOKEN`, et `KAGGLE_KEY` (fichier kaggle.json) ou `KAGGLE_API_TOKEN` (jeton KGAT_) ; variable `KAGGLE_USERNAME`. Téléphone vérifié sur Kaggle pour le GPU.
- Variable facultative : `BERCAIL_URL` (par défaut `https://app.bercail.online`).

Le dépôt est public (minutes GitHub Actions gratuites et illimitées). Les journaux n'affichent que des comptes.
GitHub met en pause les tâches planifiées d'un dépôt public sans activité depuis 60 jours : un commit par mois suffit.
