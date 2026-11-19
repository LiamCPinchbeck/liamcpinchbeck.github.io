# Blog comments: Cloudflare setup

Comments are stored in a free Cloudflare D1 database and served by a free Cloudflare Worker.
Readers don't sign in. Jekyll ignores this `_cloudflare/` folder, so nothing here is published.

Dashboard labels below may differ slightly; look for the closest match.

## 1. Account
Sign up at https://dash.cloudflare.com (free plan; no domain needed).

## 2. Database
1. **Storage & Databases → D1 → Create database**, named `blog-comments`.
2. Open it → **Console**, paste the contents of `schema.sql`, and run it.

## 3. Worker
1. **Workers & Pages → Create → Worker** (start from "Hello World"), named `blog-comments`. Click **Deploy**.
2. **Edit code**: delete the sample, paste the whole of `comments-worker.js`, then **Deploy**.
3. Note the Worker URL, e.g. `https://blog-comments.<your-subdomain>.workers.dev`.

## 4. Connect the database
Worker → **Settings → Bindings → Add → D1 database**: variable name `DB`, database `blog-comments`.

## 5. Settings (Worker → Settings → Variables and Secrets)
| Name | Type | Value |
|---|---|---|
| `ALLOWED_ORIGINS` | Text | `https://liamcpinchbeck.github.io,http://localhost:4000,http://127.0.0.1:4000` |
| `ADMIN_KEY` | Secret | a long random string (your moderation password) |
| `IP_SALT` | Secret | another random string |
| `TURNSTILE_SECRET` | Secret | from step 6 (optional) |
| `BLOCK_WORDS` | Text | optional; comma-separated words that send a comment to review |
| `HOLD_LINKS` | Text | optional; `false` to publish comments with links instantly (default: hold) |
| `HOLD_ALL` | Text | optional; `true` to hold every comment for approval |

Generate random strings with: `openssl rand -base64 32`

## 6. Turnstile spam check (optional, recommended)
**Turnstile → Add widget**: hostnames `liamcpinchbeck.github.io` and `localhost`, mode "Managed".
Copy the **site key** into `_config.yml` (`comments: selfhosted: turnstile_sitekey`) and the
**secret key** into the Worker's `TURNSTILE_SECRET`.

## 7. Site config
In `_config.yml`, set `comments: selfhosted: worker_url` to your Worker URL (no trailing slash).
Restart `bundle exec jekyll serve`, open a post, and leave a test comment.

## Moderating
Open `https://<your-worker-url>/admin` (works on your phone) and log in with `ADMIN_KEY`.
- Comments with links or blocked words are **held** until you approve them.
- Everything else appears instantly; you can hide or delete anything.

## What blocks spam
Turnstile check · hidden honeypot field · 3-second minimum fill time ·
3 comments per 10 minutes per connection · links and blocked words held for review ·
comments shown as plain text (links aren't clickable, so spam earns no SEO value).
