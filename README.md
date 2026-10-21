# Lemma

Bite-size, Duolingo-style lessons on five ideas from linear algebra and deep learning:

- **PCA**: rotate a line through a point cloud to find the direction of maximum variance, then pick components from a scree plot.
- **SVD**: watch a matrix act as rotate, stretch, rotate, and compress an image with a rank-k approximation.
- **L2 regularization**: slide λ on an overfit polynomial to find the test-error sweet spot, and see ridge shrink the flat directions first.
- **Weight decay**: train with and without decay, and see why AdamW decouples it from Adam.
- **Dropout**: drop neurons at random, switch between train and test mode, and set the inverted-dropout scale.

Each lesson has hearts, XP and a daily streak. Progress is saved to your account on the server, so it follows you between browsers.

## Run it

Requires Node.js 20.6 or newer. There are no npm dependencies.

```bash
cp .env.example .env        # then set SEED_USERNAME and SEED_PASSWORD
npm run seed                # creates that user (or resets their password)
npm start                   # http://127.0.0.1:3000
```

Set `PORT`, `HOST` or `DATA_FILE` to change where it listens or where data is stored.

## How it works

| Piece | What it does |
| --- | --- |
| `public/index.html` | The whole app: lessons, interactive widgets, sign-in screen. Plain HTML, CSS and JS. |
| `server.js` | A small Node HTTP server that serves the page and the JSON API below. |
| `lib/store.js` | JSON-file storage in `data/db.json`, written atomically (temp file plus rename). |
| `lib/auth.js` | scrypt password hashing and session tokens. |
| `lib/progress.js` | Validates progress and merges it so a stale tab can't erase finished lessons. |

### API

| Method | Path | Notes |
| --- | --- | --- |
| `POST` | `/api/login` | `{ username, password }`. Sets an `HttpOnly`, `SameSite=Lax` session cookie that lasts 30 days. |
| `POST` | `/api/logout` | Ends the session. |
| `GET` | `/api/me` | The signed-in username, or 401. |
| `GET` | `/api/progress` | `{ done, xp, streak, last, muted }` |
| `PUT` | `/api/progress` | Merges with what's saved: lessons are unioned, XP keeps the max, the streak comes from the most recent day. |
| `DELETE` | `/api/progress` | Resets progress (keeps the sound setting). |

### Security notes

- Passwords are stored only as salted scrypt hashes. Session tokens are stored only as SHA-256 hashes.
- Writes must be `application/json`, which, together with `SameSite=Lax` cookies, blocks cross-site form posts.
- After 10 failed sign-ins from one address, sign-in is locked for 15 minutes.
- `.env` and `data/` are git-ignored. Set `NODE_ENV=production` behind HTTPS to mark cookies `Secure`.

## Tests

```bash
npm test
```
