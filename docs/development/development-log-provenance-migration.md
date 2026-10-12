# Development Log Provenance Migration

`shared_scripts/development_log_migration.py` prepares a reviewed repair of an
inherited development log when ordinary change fragments cannot reconcile
duplicate or legacy IDs. It never changes the source file. The result is a
candidate log and a JSON receipt containing the complete original bytes,
original SHA-256, every changed record's original bytes and SHA-256, the
decisions, and the candidate SHA-256. Review the diff and receipt before
installing the candidate through a governed repository PR.

The plan is JSON with `schema_version: 1`, `source_sha256` (the exact original
file), and a `decisions` array. Each decision has the original one-based
`source_line`, `source_sha256` of the complete entry block, `action`, and a
nonempty `rationale`. `amend` may set a unique `entry_id` of the form `DL-#N`
and `fields` such as `Issue` or `Owner`. Each field change needs a `value` and
an `evidence` explanation. State and PR cannot be changed. To replace `Last
verified`, supply `verification` with an ISO `date`, a locally resolvable Git
`commit`, `scope: "log_metadata_only"`, a repository-relative `receipt_path`,
and its `receipt_sha256`. The receipt file must contain that commit and the
statement "Log metadata only; feature acceptance unverified". The generated
field states that scope visibly. A reviewer must confirm the receipt actually
supports the metadata review; this migration makes no runtime or scientific
qualification claim. A source commit or collation commit alone is insufficient.
Only nonterminal records may receive a new metadata-only verification. A
terminal row with an unusable historical verification field must be
quarantined and preserved as unresolved history.

`quarantine` is available only for terminal historical records that lack
adequate verification. It removes the row from the candidate state table but
preserves its complete bytes and disposition in the receipt. A plan using it
must set `quarantine_receipt` to the repository-relative path where the receipt
will be committed. The candidate lists each unresolved original ID and line
under **Quarantined Historical Records** and points to that receipt. It does
not approve the feature, change issue status, or hide active work. An active or
parked record needs a reviewed amendment.

From the target repository root, after preparing and reviewing the plan and
verification files:

```sh
python -m shared_scripts.development_log_migration \
  --repo-root . \
  --source docs/development/DEVELOPMENT_LOG.md \
  --plan docs/development/provenance/tools-devlog-migration-plan.json \
  --bundle-dir /tmp/tools-devlog-migration-bundle-v1
```

The new bundle contains `DEVELOPMENT_LOG.candidate.md` and
`migration-receipt.json`; it appears only after both files have been written.
The command refuses
stale hashes, unknown records, unsupported decisions, missing evidence, and any
candidate that still fails the strict development-log validator. The target
repository's normal PR process then reviews and installs the candidate and
receipt together. No feature issue closes merely because a log row is repaired.

## Tools Sync Provenance

This Tools copy of `shared_scripts/development_log_migration.py` is byte-identical
to Repository_Management protected merge `db97f51573889f2a82b33562785735b3f801c24f`
(SHA-256 `c7bef7aa9a45c5b3c81689230667707ff39733360446e0b8db35ad86382627fa`).
The module command above uses the Tools repository's own shared validator.
