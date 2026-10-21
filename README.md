# Lemma

Bite-size, Duolingo-style lessons on the math behind machine learning, with Anki-style spaced repetition so it sticks. It works on phones and installs to the home screen.

**Course (20 lessons, 92 flashcards)**

1. **Matrices**: dot products, matrix × vector (row and column views), matrix × matrix entry by entry, the data matrix and XᵀX, NumPy shapes, axes and broadcasting.
2. **PCA**: max-variance directions, components and scree plots, PCA in NumPy with every shape.
3. **SVD**: rotate–stretch–rotate, rank-k compression, `np.linalg.svd` shapes and how it gives PCA.
4. **L2 regularization**: overfitting, ridge geometry, the gradient and closed form in NumPy.
5. **Weight decay**: decay vs L2 vs AdamW, and the shapes inside a PyTorch `nn.Linear`.
6. **Dropout**: masks, inverted scaling, and dropout in NumPy.

Each lesson mixes interactive widgets (drag vectors, tap matrix entries, collapse axes) with questions, including "predict the shape" questions where you type a NumPy shape. Finishing a lesson adds its flashcards to **Review**. Cards are scheduled with Anki's SM-2 rules (Again / Hard / Good / Easy, learning steps of 1 and 10 minutes, then growing day intervals).

## Run it

Requires Node.js 20.6 or newer. There are no npm dependencies.

```bash
cp .env.example .env        # then set SEED_USERNAME and SEED_PASSWORD
npm run seed                # creates that user (or resets their password)
npm start                   # http://127.0.0.1:3000
```

Set `PORT`, `HOST`, `DATA_FILE` or `TRUST_PROXY=1` to change where it listens, where data is stored, and whether to trust a reverse proxy's client address. **[DEPLOY.md](DEPLOY.md)** covers putting it on a domain with HTTPS and installing it on a phone.

## Project layout

| Path | What it is |
| --- | --- |
| `public/index.html` | The page shell. |
| `public/js/app.js` | Lesson engine, review sessions, sign-in, syncing. |
| `public/js/srs.js` | The spaced-repetition scheduler (pure, shared with tests). |
| `public/js/widgets.js` | Interactive widgets. |
| `public/js/content/` | One file per unit: lessons, questions and flashcards. See **[CONTENT.md](CONTENT.md)**. |
| `public/sw.js`, `manifest.webmanifest`, `icons/` | What makes it an installable, offline-capable app. |
| `server.js` | Node HTTP server: static files plus the JSON API. |
| `lib/` | Storage, password hashing, and validation and merging for progress and reviews. |
| `scripts/verify_numpy.py` | Runs every NumPy claim the lessons make. |

## API

| Method | Path | Notes |
| --- | --- | --- |
| `POST` | `/api/login` | `{ username, password }`. Sets an `HttpOnly`, `SameSite=Lax` cookie that lasts 30 days. |
| `POST` | `/api/logout` | Ends the session. |
| `GET` | `/api/me` | The signed-in username, or 401. |
| `GET` / `PUT` | `/api/progress` | `{ done, xp, streak, last, muted }`. PUT merges: lessons are unioned and XP keeps the max. |
| `GET` / `PUT` | `/api/reviews` | `{ cards: { [cardId]: state } }`. PUT merges per card; the most recent review wins. |
| `DELETE` | `/api/progress` | Resets lessons and reviews (keeps the sound setting). |

Reviews are written to an outbox in the browser before they are sent, so grading works offline and syncs later.

## Security notes

- Passwords are stored only as salted scrypt hashes. Session tokens are stored only as SHA-256 hashes.
- Writes must be `application/json`, which, with `SameSite=Lax` cookies, blocks cross-site form posts.
- Pages are served with a Content-Security-Policy that only allows scripts from this server.
- After 10 failed sign-ins from one address, sign-in is locked for 15 minutes.
- `.env` and `data/` are git-ignored. Set `NODE_ENV=production` behind HTTPS to mark cookies `Secure`.

## Tests

```bash
npm test                 # server, scheduler, service worker, content checks
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
npm run verify:py        # 65 NumPy checks against the lesson text
```
