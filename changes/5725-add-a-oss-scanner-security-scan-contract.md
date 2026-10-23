---
issue: 5725
summary: "Add a .oss-scanner security-scan contract (project.yaml, offline audit Dockerfile, threat model) for the fleet scanner pilot."
title: "Security-Scan Contract for the Fleet Scanner"
branch: "claude/oss-scanner-contract"
paths: ".oss-scanner/project.yaml,.oss-scanner/Dockerfile,.oss-scanner/threat_model.md"
---

Part of Runner_Dashboard#2566. Contract is validated by Repository_Management shared_scripts/oss_scanner_contract.py; not a required check. No scans run until the executor has a VM scan host.
