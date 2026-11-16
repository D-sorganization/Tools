# Change fragment: remove the retired vendored GAAI tree

- Deletes `.gaai/`, `start-gaai-daemon.sh`, and GAAI slash-commands.
- Drops them from the root allowlist and module inventory; guard tests fail if they return.
- Refs #2725. Refs #2727.
