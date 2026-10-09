# Native State Transport Publication Staging

Tools issue #5470 requires a separately versioned opaque native restart envelope,
strict native clock and ordered-input identity, immutable payload verification,
and capability blockers. Existing numeric replay semantics remain authoritative.

The implementation is retained on local branch `feat/native-state-artifact-5470`.
Its native/schema subset passes77 tests, the full mocap suite231, and manual
QA/projection39. These are local prerequisite results; this initial draft does
not yet contain or qualify that implementation. The verified pinned renderer
produced the expanded canonical manual. Actual native decoding and production
model matching remain UpstreamDrift responsibilities under #11921/#11923.

The publishing guards require a real PR key for SPEC freshness before a source
push. This documentation-only draft establishes that key; merge the tested
implementation into this branch, add its actual PR row and rerun every required
gate. Do not bypass checks or invent a PR number. The ultimate muscle-driven
capture match and independent all-model replay epic remains active.
