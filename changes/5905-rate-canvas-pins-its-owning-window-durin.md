---
issue: 5905
summary: "Rate canvas pins its owning window during paintEvent so cyclic GC in a pending-draw flush cannot delete it mid-paint and segfault Qt"
---
