// Blog comments Worker (Cloudflare Workers + D1). No sign-in for readers.
//
// Bindings / settings (Cloudflare dashboard > Worker > Settings):
//   DB                 D1 database binding (required)
//   ALLOWED_ORIGINS    text, comma-separated, e.g. "https://liamcpinchbeck.github.io,http://localhost:4000,http://127.0.0.1:4000"
//   ADMIN_KEY          secret, long random string; used to log in at /admin
//   IP_SALT            secret, random string; IPs are only stored as salted hashes for rate limiting
//   TURNSTILE_SECRET   secret, optional; Cloudflare Turnstile secret key
//   BLOCK_WORDS        text, optional; comma-separated words that send a comment to review
//   HOLD_LINKS         text, optional; "false" to publish comments with links instantly (default: hold them)
//   HOLD_ALL           text, optional; "true" to hold every comment for approval
//
// Endpoints:
//   GET  /comments?page=/path/      public, approved comments for a page
//   POST /comments                  public, submit a comment (JSON)
//   GET  /admin                     moderation page (log in with ADMIN_KEY)
//   /admin/api/...                  moderation API (Authorization: Bearer ADMIN_KEY)

const MAX_NAME = 60;
const MAX_MSG = 4000;
const MIN_FILL_MS = 3000;       // reject forms submitted faster than this
const RATE_WINDOW_MIN = 10;     // rate-limit window
const RATE_MAX = 3;             // max comments per window per connection
const DEFAULT_BLOCK_WORDS =
  "casino,viagra,cialis,crypto,bitcoin,forex,loan,payday,essay writing,backlink,seo service,escort,porn,betting";

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const cors = corsHeaders(request.headers.get("Origin") || "", env);

    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: cors });

    try {
      if (url.pathname === "/comments" && request.method === "GET") return await listComments(url, env, cors);
      if (url.pathname === "/comments" && request.method === "POST") return await postComment(request, env, cors);
      if (url.pathname === "/admin" && request.method === "GET") return adminPage();
      if (url.pathname.startsWith("/admin/api/")) return await adminApi(request, url, env);
      return json({ error: "Not found" }, 404, cors);
    } catch (err) {
      console.error(err);
      return json({ error: "Server error. Please try again later." }, 500, cors);
    }
  },
};

// ---------- helpers ----------

function corsHeaders(origin, env) {
  const allowed = (env.ALLOWED_ORIGINS || "").split(",").map((s) => s.trim()).filter(Boolean);
  const h = {
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Vary": "Origin",
  };
  if (allowed.includes(origin)) h["Access-Control-Allow-Origin"] = origin;
  return h;
}

function json(obj, status = 200, headers = {}) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8", ...headers },
  });
}

function cleanPage(p) {
  return String(p || "").trim().slice(0, 300);
}

async function sha256(s) {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function verifyTurnstile(token, ip, secret) {
  if (!token) return false;
  const form = new FormData();
  form.append("secret", secret);
  form.append("response", token);
  if (ip) form.append("remoteip", ip);
  const r = await fetch("https://challenges.cloudflare.com/turnstile/v0/siteverify", { method: "POST", body: form });
  const data = await r.json();
  return Boolean(data.success);
}

function needsReview(message, name, env) {
  if (env.HOLD_ALL === "true") return true;
  const text = (name + " " + message).toLowerCase();
  const hasLink = /https?:\/\/|www\.|\b[a-z0-9-]+\.(com|net|org|io|xyz|ru|top|info|biz)\b/i.test(text);
  if (hasLink && env.HOLD_LINKS !== "false") return true;
  const words = (env.BLOCK_WORDS || DEFAULT_BLOCK_WORDS)
    .split(",").map((w) => w.trim().toLowerCase()).filter(Boolean);
  return words.some((w) => text.includes(w));
}

// ---------- public API ----------

async function listComments(url, env, cors) {
  const page = cleanPage(url.searchParams.get("page"));
  if (!page) return json({ error: "Missing page" }, 400, cors);
  const { results } = await env.DB.prepare(
    "SELECT id, name, message, created_at FROM comments WHERE page = ? AND status = 'approved' ORDER BY created_at ASC"
  ).bind(page).all();
  return json({ comments: results || [] }, 200, { ...cors, "Cache-Control": "no-store" });
}

async function postComment(request, env, cors) {
  if (!cors["Access-Control-Allow-Origin"]) return json({ error: "Origin not allowed" }, 403, cors);

  let body;
  try {
    body = await request.json();
  } catch {
    return json({ error: "Bad request" }, 400, cors);
  }

  const name = String(body.name || "").trim().slice(0, MAX_NAME) || "Anonymous";
  const message = String(body.message || "").trim();
  const page = cleanPage(body.page);

  // Honeypot: humans never see or fill this field. Pretend success, store nothing.
  if (body.website) return json({ ok: true, status: "approved" }, 200, cors);

  if (!page || !message) return json({ error: "Please write a comment first." }, 400, cors);
  if (message.length > MAX_MSG) return json({ error: `Comments are limited to ${MAX_MSG} characters.` }, 400, cors);
  if (Date.now() - Number(body.started || 0) < MIN_FILL_MS) {
    return json({ error: "That was quick! Please wait a moment and try again." }, 400, cors);
  }

  const ip = request.headers.get("CF-Connecting-IP") || "";
  if (env.TURNSTILE_SECRET) {
    const ok = await verifyTurnstile(body.turnstile, ip, env.TURNSTILE_SECRET);
    if (!ok) return json({ error: "Spam check failed. Please refresh the page and try again." }, 400, cors);
  }

  const ipHash = await sha256((env.IP_SALT || "change-me") + ip);
  const since = new Date(Date.now() - RATE_WINDOW_MIN * 60000).toISOString();
  const recent = await env.DB.prepare(
    "SELECT COUNT(*) AS n FROM comments WHERE ip_hash = ? AND created_at > ?"
  ).bind(ipHash, since).first();
  if (recent && recent.n >= RATE_MAX) {
    return json({ error: "You've commented a few times already. Please wait a few minutes." }, 429, cors);
  }

  const status = needsReview(message, name, env) ? "pending" : "approved";
  const createdAt = new Date().toISOString();
  const res = await env.DB.prepare(
    "INSERT INTO comments (page, name, message, created_at, ip_hash, status) VALUES (?, ?, ?, ?, ?, ?)"
  ).bind(page, name, message, createdAt, ipHash, status).run();

  return json(
    { ok: true, status, comment: { id: res.meta.last_row_id, name, message, created_at: createdAt } },
    200,
    cors
  );
}

// ---------- admin ----------

async function adminApi(request, url, env) {
  const auth = request.headers.get("Authorization") || "";
  if (!env.ADMIN_KEY || auth !== "Bearer " + env.ADMIN_KEY) return json({ error: "Unauthorized" }, 401);

  if (url.pathname === "/admin/api/comments" && request.method === "GET") {
    const { results } = await env.DB.prepare(
      "SELECT id, page, name, message, created_at, status FROM comments ORDER BY (status = 'pending') DESC, created_at DESC LIMIT 300"
    ).all();
    return json({ comments: results || [] });
  }

  const m = url.pathname.match(/^\/admin\/api\/comments\/(\d+)$/);
  if (m && request.method === "DELETE") {
    await env.DB.prepare("DELETE FROM comments WHERE id = ?").bind(Number(m[1])).run();
    return json({ ok: true });
  }
  if (m && request.method === "POST") {
    const { status } = await request.json();
    if (!["approved", "pending"].includes(status)) return json({ error: "Bad status" }, 400);
    await env.DB.prepare("UPDATE comments SET status = ? WHERE id = ?").bind(status, Number(m[1])).run();
    return json({ ok: true });
  }
  return json({ error: "Not found" }, 404);
}

function adminPage() {
  return new Response(ADMIN_HTML, {
    headers: { "Content-Type": "text/html; charset=utf-8", "X-Robots-Tag": "noindex", "Cache-Control": "no-store" },
  });
}

const ADMIN_HTML = `<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex"><title>Comment moderation</title>
<style>
  body{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:760px;margin:0 auto;padding:16px;color:#1f2328;background:#f7f6f2}
  @media (prefers-color-scheme:dark){body{background:#16181b;color:#e8e6e1}.c{background:#1f2226!important;border-color:#2e3238!important}}
  h1{font-size:20px} input{padding:8px;width:100%;box-sizing:border-box;font-size:15px}
  button{padding:6px 12px;margin-right:6px;border-radius:6px;border:1px solid #999;background:transparent;color:inherit;font-size:14px}
  .c{background:#fff;border:1px solid #e4e1d8;border-radius:10px;padding:12px;margin:10px 0}
  .meta{font-size:13px;opacity:.75;word-break:break-all} .msg{white-space:pre-wrap;margin:8px 0}
  .pending{border-left:4px solid #c98a12} .tag{font-size:12px;font-weight:700;color:#c98a12}
  #filter button.on{background:#2f6f6a;color:#fff;border-color:#2f6f6a}
</style></head><body>
<h1>Comment moderation</h1>
<div id="login"><p>Admin key:</p><input id="key" type="password" autocomplete="current-password">
<p><button id="go">Log in</button></p></div>
<div id="app" hidden>
  <p id="filter"><button data-f="pending" class="on">Held for review</button><button data-f="all">All</button>
  <button id="out">Log out</button></p>
  <div id="list"></div>
</div>
<script>
(function(){
  var key = '', filter = 'pending', all = [];
  try { key = localStorage.getItem('cm_key') || ''; } catch (e) {}
  var $ = function(id){ return document.getElementById(id); };
  function api(path, opts){
    opts = opts || {}; opts.headers = opts.headers || {};
    opts.headers['Authorization'] = 'Bearer ' + key;
    if (opts.body) opts.headers['Content-Type'] = 'application/json';
    return fetch('/admin/api' + path, opts).then(function(r){
      if (r.status === 401) { logout(); throw new Error('Wrong key'); }
      return r.json();
    });
  }
  function el(tag, cls, text){ var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  function render(){
    var list = $('list'); list.innerHTML = '';
    var items = all.filter(function(c){ return filter === 'all' || c.status === 'pending'; });
    if (!items.length) { list.appendChild(el('p', null, filter === 'pending' ? 'Nothing waiting for review.' : 'No comments yet.')); return; }
    items.forEach(function(c){
      var box = el('div', 'c' + (c.status === 'pending' ? ' pending' : ''));
      if (c.status === 'pending') box.appendChild(el('div', 'tag', 'HELD FOR REVIEW'));
      box.appendChild(el('div', 'meta', c.name + ' · ' + new Date(c.created_at).toLocaleString() + ' · ' + c.page));
      box.appendChild(el('div', 'msg', c.message));
      var act = el('div');
      var toggle = el('button', null, c.status === 'pending' ? 'Approve' : 'Hide');
      toggle.onclick = function(){ api('/comments/' + c.id, { method: 'POST', body: JSON.stringify({ status: c.status === 'pending' ? 'approved' : 'pending' }) }).then(load); };
      var del = el('button', null, 'Delete');
      del.onclick = function(){ if (confirm('Delete this comment permanently?')) api('/comments/' + c.id, { method: 'DELETE' }).then(load); };
      act.appendChild(toggle); act.appendChild(del); box.appendChild(act);
      list.appendChild(box);
    });
  }
  function load(){ return api('/comments').then(function(d){ all = d.comments || []; $('login').hidden = true; $('app').hidden = false; render(); }); }
  function logout(){ key = ''; try { localStorage.removeItem('cm_key'); } catch (e) {} $('app').hidden = true; $('login').hidden = false; }
  $('go').onclick = function(){ key = $('key').value.trim(); try { localStorage.setItem('cm_key', key); } catch (e) {} load().catch(function(e){ alert(e.message); }); };
  $('out').onclick = logout;
  Array.prototype.forEach.call(document.querySelectorAll('#filter [data-f]'), function(b){
    b.onclick = function(){ filter = b.getAttribute('data-f');
      Array.prototype.forEach.call(document.querySelectorAll('#filter [data-f]'), function(x){ x.className = x === b ? 'on' : ''; });
      render(); };
  });
  if (key) load().catch(function(){});
})();
</script></body></html>`;
