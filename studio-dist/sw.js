/* ══════════════════════════════════════════════════════════════
   K'UHUL STUDIO — service worker
   App-shell is cache-first so the studio opens instantly and works
   offline. The model fabric (/v1/* on the llama-server) and every
   non-GET request are NEVER cached — inference must always be live.
   ══════════════════════════════════════════════════════════════ */
const CACHE = 'kuhul-studio-v2';
const SHELL = [
  './',
  './index.html',
  './kuhul-studio.webmanifest',
  './kuhul-icon.svg'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE)
      .then((c) => c.addAll(SHELL))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;

  // Never intercept model calls or any mutation — inference stays live.
  if (req.method !== 'GET') return;

  const url = new URL(req.url);
  const sameOrigin = url.origin === self.location.origin;
  const isFontCdn = /(^|\.)fonts\.(googleapis|gstatic)\.com$/.test(url.hostname);

  // Explicitly bypass the fabric endpoints even if same-origin hosted.
  if (url.pathname.startsWith('/v1/') ||
      url.pathname.startsWith('/health') ||
      url.pathname.startsWith('/props') ||
      url.pathname.startsWith('/cors-proxy')) {
    return; // default network handling
  }

  if (sameOrigin) {
    // App shell: cache-first, refresh cache in the background, offline fallback.
    event.respondWith(
      caches.match(req).then((hit) => {
        if (hit) return hit;
        return fetch(req).then((res) => {
          if (res && res.ok && res.type === 'basic') {
            const copy = res.clone();
            caches.open(CACHE).then((c) => c.put(req, copy));
          }
          return res;
        }).catch(() => caches.match('./index.html') || caches.match('./'));
      })
    );
  } else if (isFontCdn) {
    // Google Fonts: stale-while-revalidate.
    event.respondWith(
      caches.open(CACHE).then(async (c) => {
        const hit = await c.match(req);
        const net = fetch(req).then((res) => {
          if (res && res.ok) c.put(req, res.clone());
          return res;
        }).catch(() => hit);
        return hit || net;
      })
    );
  }
  // Any other cross-origin GET (e.g. the model on :8085) → untouched network.
});
