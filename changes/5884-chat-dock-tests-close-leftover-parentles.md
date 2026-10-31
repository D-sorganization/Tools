---
issue: 5884
summary: "Chat dock tests close leftover parentless docks after each test so a deferred scroll cannot fire on a GC-freed scroll bar and segfault the xdist worker"
dl_state: "in_review"
next_step: "Merge after acceptance review"
branch: "brick/ci-toolchain/5884-chat-scroll-segfault"
---
