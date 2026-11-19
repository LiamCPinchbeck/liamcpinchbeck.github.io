-- Run once in the D1 console (Cloudflare dashboard > D1 > blog-comments > Console).
CREATE TABLE IF NOT EXISTS comments (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  page       TEXT NOT NULL,                     -- post path, e.g. /posts/2026/02/2026-02-03-SliceGibbs/
  name       TEXT NOT NULL,
  message    TEXT NOT NULL,
  created_at TEXT NOT NULL,                     -- ISO 8601, UTC
  ip_hash    TEXT,                              -- salted hash, only used for rate limiting
  status     TEXT NOT NULL DEFAULT 'approved'   -- 'approved' (public) or 'pending' (held)
);
CREATE INDEX IF NOT EXISTS idx_comments_page ON comments (page, status, created_at);
CREATE INDEX IF NOT EXISTS idx_comments_ip   ON comments (ip_hash, created_at);
