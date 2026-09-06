# Optimizer GUI (legacy shim)

`src/optimizer_gui` is a compatibility shim. The standalone PyQt6 application
that used to live in this directory was consolidated into
[`src/movement_optimizer`](../movement_optimizer) (Tools #3983); the drifted
vendored copy of the swing/chain models was removed.

## What remains

- `gui_registration.py` — hidden catalog registration (`catalog_visible: False`)
  that redirects to the canonical `movement_optimizer` PyQt6 app.
- `launch_pyqt6.py` — compatibility launcher that starts the canonical
  Movement Optimizer application:

  ```bash
  python launch_pyqt6.py
  ```

New development happens in `src/movement_optimizer`. Do not add features here;
old launch paths must keep redirecting to the canonical app.