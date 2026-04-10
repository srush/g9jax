/*! coi-service-worker v0.1.7 - Guido Zuidhof, licensed under MIT */
/* https://github.com/niccokunzmann/coi-serviceworker/blob/master/coi-serviceworker.js */
if (typeof window === "undefined") {
  self.addEventListener("install", () => self.skipWaiting());
  self.addEventListener("activate", (e) =>
    e.waitUntil(self.clients.claim()),
  );
  self.addEventListener("message", (ev) => {
    if (ev.data && ev.data.type === "deregister") {
      self.registration
        .unregister()
        .then(() => self.clients.matchAll())
        .then((clients) =>
          clients.forEach((client) => client.navigate(client.url)),
        );
    }
  });
  self.addEventListener("fetch", function (e) {
    if (
      e.request.cache === "only-if-cached" &&
      e.request.mode !== "same-origin"
    )
      return;
    e.respondWith(
      fetch(e.request).then((response) => {
        if (response.status === 0) return response;
        const newHeaders = new Headers(response.headers);
        newHeaders.set("Cross-Origin-Embedder-Policy", "credentialless");
        newHeaders.set("Cross-Origin-Opener-Policy", "same-origin");
        return new Response(response.body, {
          status: response.status,
          statusText: response.statusText,
          headers: newHeaders,
        });
      }),
    );
  });
} else {
  (async function () {
    const coop = window.crossOriginIsolated;
    if (coop) {
      console.log("Cross-origin isolated, SharedArrayBuffer available");
      return;
    }
    if (window.isSecureContext && !coop) {
      const reg = await navigator.serviceWorker.register(
        window.document.currentScript.src,
        { scope: new URL(".", window.document.currentScript.src).pathname },
      );
      if (reg.active && !navigator.serviceWorker.controller) {
        window.location.reload();
      } else if (!reg.active) {
        reg.addEventListener("updatefound", () => {
          reg.installing.addEventListener("statechange", () => {
            if (reg.installing?.state === "activated") {
              window.location.reload();
            }
          });
        });
      }
    }
  })();
}
