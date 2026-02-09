const CACHE = "seydyaar-v0.2.0";
const CORE = [
  "./",
  "./index.html",
  "./app.html",
  "./styles.css",
  "./home.js",
  "./app.js",
  "./manifest.json",
  "./assets/logo.png",
  "./latest/meta_index.json",
  "./latest/preview.png"
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(CORE)));
  self.skipWaiting();
});

self.addEventListener("activate", (e) => {
  e.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.map(k => (k === CACHE) ? null : caches.delete(k)));
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  const url = new URL(req.url);
  // Cache-first for same-origin assets + latest data; network-first for tiles
  if (url.origin === location.origin) {
    e.respondWith((async () => {
      const cache = await caches.open(CACHE);
      const cached = await cache.match(req);
      if (cached) return cached;
      try {
        const fresh = await fetch(req);
        if (fresh && fresh.status === 200 && (req.method === "GET")) {
          cache.put(req, fresh.clone());
        }
        return fresh;
      } catch (err) {
        return cached || new Response("Offline", {status: 503});
      }
    })());
  }
});
