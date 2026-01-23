/**
 * Service Worker for Unit Converter PWA
 * Enables offline functionality
 */

const CACHE_NAME = 'unit-converter-v1.0.0';
const urlsToCache = [
  '/unit-converter-app/',
  '/unit-converter-app/index.html',
  '/unit-converter-app/styles.css',
  '/unit-converter-app/app.js',
  '/unit-converter-app/converter.js',
  '/unit-converter-app/manifest.json'
];

// Install event - cache files
self.addEventListener('install', event => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then(cache => {
        return cache.addAll(urlsToCache);
      })
      .then(() => {
        // Activate immediately
        return self.skipWaiting();
      })
  );
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', event => {
  event.respondWith(
    caches
      .match(event.request)
      .then(response => {
        // Cache hit - return response
        if (response) {
          return response;
        }

        // Clone the request
        const fetchRequest = event.request.clone();

        return fetch(fetchRequest).then(response => {
          // Check if valid response
          if (!response || response.status !== 200 || response.type !== 'basic') {
            return response;
          }

          // Clone the response
          const responseToCache = response.clone();

          caches.open(CACHE_NAME).then(cache => {
            cache.put(event.request, responseToCache);
          });

          return response;
        });
      })
      .catch(() => {
        // If both cache and network fail, show offline page
        return caches.match('/unit-converter-app/index.html');
      })
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
  const cacheWhitelist = [CACHE_NAME];

  event.waitUntil(
    caches
      .keys()
      .then(cacheNames => {
        return Promise.all(
          cacheNames.map(cacheName => {
            if (!cacheWhitelist.includes(cacheName)) {
              return caches.delete(cacheName);
            }
          })
        );
      })
      .then(() => {
        // Claim all clients
        return self.clients.claim();
      })
  );
});

// Background sync for future features
self.addEventListener('sync', event => {
  if (event.tag === 'sync-conversions') {
    event.waitUntil(syncConversions());
  }
});

async function syncConversions() {
  // Future: sync conversion history to cloud
  return Promise.resolve();
}
