---
issue: 6041
summary: "Harden Rate Web Playwright Trusted apt/dpkg lock waiting and retry on transient contention"
---

- Wait for `/var/lib/dpkg/lock-frontend` alongside existing apt/dpkg locks and retry `npx --no-install playwright install --with-deps chromium` up to 3 times if background apt processes acquire locks mid-install (#6041).
