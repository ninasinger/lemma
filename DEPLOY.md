# Putting Lemma on your own domain

Lemma is an installable web app (a PWA). Once it is served over **HTTPS** from your domain, you can add it to your phone's home screen, where it opens full-screen with its own icon, like a native app. No app store is involved.

There are two ways to host it. **Netlify** needs no server to look after. A server of your own (further down) suits you if you'd rather keep everything on one machine.

## Netlify

On Netlify, the pages in `public/` are served from Netlify's CDN, and the API runs as a Netlify Function (`netlify/functions/api.mjs`) that stores accounts, progress and reviews in **Netlify Blobs**. `netlify.toml` already holds every setting.

1. **Create the site.** At [app.netlify.com](https://app.netlify.com), choose **Add new project → Import an existing project → GitHub**. Give Netlify access to the `lemma` repository only, and pick it. The build settings fill in from `netlify.toml` (publish `public`, build command `npm test`, so a failing test stops a bad deploy).
2. **Set your account.** Before the first deploy, or right after, open **Project configuration → Environment variables** and add `SEED_USERNAME` and `SEED_PASSWORD`. Mark the password as a secret. The first request after deploying creates that account if it doesn't exist yet. It never changes an existing account, so you can delete both variables once you've signed in.
3. **Deploy.** Netlify builds and gives you an address like `lemma-1234.netlify.app` with HTTPS. Sign in there to check.
4. **Add your domain.** Under **Domain management → Add a domain**, enter the domain you bought. If you bought it through Netlify there's nothing else to do. Otherwise either switch the domain's nameservers to the ones Netlify lists, or add the DNS records it shows at your registrar. Netlify issues the HTTPS certificate automatically once DNS resolves.

Every push to `main` redeploys. Your data lives in the project's Blobs store named `lemma` and is kept across deploys.

**Your local data doesn't move automatically.** The Netlify account starts fresh. Lessons are quick to redo; ask if you want a one-off import of `data/db.json`.

## A server of your own

Any small Linux VPS works (1 vCPU, 512 MB RAM is plenty). This setup stores everything in one JSON file, so the server needs a persistent disk.

### Option A: Docker

```bash
docker build -t lemma .
docker run -d --name lemma --restart unless-stopped -p 127.0.0.1:3000:3000 -v lemma-data:/data lemma
docker exec -e SEED_USERNAME=you -e SEED_PASSWORD='a long passphrase' lemma node scripts/seed.js
```

### Option B: Node directly (systemd)

```ini
# /etc/systemd/system/lemma.service
[Unit]
Description=Lemma
After=network.target

[Service]
WorkingDirectory=/opt/lemma
ExecStart=/usr/bin/node server.js
Environment=NODE_ENV=production
Environment=TRUST_PROXY=1
Environment=PORT=3000
Environment=DATA_FILE=/var/lib/lemma/db.json
User=lemma
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Then create a user with `SEED_USERNAME=you SEED_PASSWORD='…' DATA_FILE=/var/lib/lemma/db.json node scripts/seed.js`.

## 2. Point the domain at it and add HTTPS

Create an `A` record for your domain (or a subdomain like `learn.example.com`) pointing at the server's IP. [Caddy](https://caddyserver.com) then gets and renews the certificate automatically:

```
# /etc/caddy/Caddyfile
learn.example.com {
    reverse_proxy 127.0.0.1:3000
}
```

Behind a proxy, keep `TRUST_PROXY=1` (the Dockerfile sets it) so sign-in lockouts apply to each visitor's real address instead of the proxy's. `NODE_ENV=production` marks the session cookie `Secure`.

## 3. Install it on your phone

- **iPhone or iPad (Safari):** open the site, tap **Share**, then **Add to Home Screen**.
- **Android (Chrome):** open the site, tap the **⋮** menu, then **Install app** (or **Add to Home screen**).

The installed app works offline for anything already loaded. Reviews graded offline are kept on the phone and sync the next time it's online.

## Trying it on your phone before you have a domain

On the same Wi-Fi, run `HOST=0.0.0.0 npm start` on your Mac and open `http://<your-mac's-IP>:3000` on the phone. Everything works in the browser, but phones only offer **Install** over HTTPS, so home-screen install has to wait for the real domain. A free tunnel such as `cloudflared tunnel --url http://localhost:3000` gives a temporary HTTPS address if you want to try installing sooner.

## Later: a real app-store-style package

If you ever want an `.apk` to sideload on Android, [Bubblewrap](https://github.com/GoogleChromeLabs/bubblewrap) wraps this PWA as a Trusted Web Activity. It needs your domain to serve `/.well-known/assetlinks.json`. For iOS, [Capacitor](https://capacitorjs.com) can wrap it, which requires an Apple developer account for TestFlight. Neither requires changes to the app itself.
