"""Authentication module for P1AM HMI Control System.

Exposes AuthManager with support for Operator and Admin roles.

Admin credential configuration
------------------------------
There is **no built-in default Admin password**. The Admin gate fails closed
until a credential is configured via one of the following (checked in order):

1. ``ADMIN_PASSWORD_HASH`` — a salted PBKDF2 hash in the portable string form
   produced by :func:`hash_admin_password`
   (``pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>``). This is the
   recommended production setting: the plaintext never lives in the environment.
2. ``ADMIN_PASSWORD`` — plaintext password (convenience for bench/dev use). It is
   hashed with the same KDF at verification time. Prefer ``ADMIN_PASSWORD_HASH``.

If neither is set, :meth:`AuthManager.verify_admin_password` returns ``False`` and
``admin_credential_configured()`` reports ``False`` so the UI can surface a
"no admin password configured" message and route the operator to setup.

Use :func:`hash_admin_password` (or ``python -m p1am_control_system.desktop.auth``)
to generate an ``ADMIN_PASSWORD_HASH`` value for deployment.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enum import StrEnum
else:
    from compatibility import StrEnum

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# PBKDF2-HMAC-SHA256 parameters. 600k iterations follows current OWASP guidance.
_PBKDF2_ALGO = "pbkdf2_sha256"
_PBKDF2_ITERATIONS = 600_000
_PBKDF2_SALT_BYTES = 16
_PBKDF2_DKLEN = 32


def _pbkdf2(password: str, salt: bytes, iterations: int) -> bytes:
    """Derive a key from a password using PBKDF2-HMAC-SHA256."""
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
        dklen=_PBKDF2_DKLEN,
    )


def hash_admin_password(
    password: str,
    *,
    iterations: int = _PBKDF2_ITERATIONS,
    salt: bytes | None = None,
) -> str:
    """Hash a plaintext password into a portable, salted PBKDF2 string.

    Args:
        password: The plaintext password to hash. Must be non-empty.
        iterations: PBKDF2 iteration count (default: OWASP-recommended 600k).
        salt: Optional explicit salt (mainly for tests). A cryptographically
            random salt is generated when not provided.

    Returns:
        A string of the form
        ``pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>`` suitable for storing
        in the ``ADMIN_PASSWORD_HASH`` environment variable or a config file.

    Raises:
        ValueError: If ``password`` is empty.
    """
    if not password:
        raise ValueError("password must be a non-empty string")
    if salt is None:
        salt = secrets.token_bytes(_PBKDF2_SALT_BYTES)
    derived = _pbkdf2(password, salt, iterations)
    return f"{_PBKDF2_ALGO}${iterations}${salt.hex()}${derived.hex()}"


def _verify_against_encoded(password: str, encoded: str) -> bool:
    """Constant-time verify ``password`` against an encoded PBKDF2 string."""
    try:
        algo, iter_str, salt_hex, hash_hex = encoded.split("$")
    except ValueError:
        return False
    if algo != _PBKDF2_ALGO:
        return False
    try:
        iterations = int(iter_str)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except ValueError:
        return False
    if iterations <= 0:
        return False
    derived = _pbkdf2(password, salt, iterations)
    return hmac.compare_digest(derived, expected)


def admin_credential_configured() -> bool:
    """Return True if an Admin credential is configured via the environment."""
    return bool(
        os.environ.get("ADMIN_PASSWORD_HASH") or os.environ.get("ADMIN_PASSWORD")
    )


class Role(StrEnum):
    """Available user roles in the system."""

    OPERATOR = "Operator"
    ADMIN = "Admin"


class AuthManager:
    """Manages authentication and role-based permissions for the HMI application."""

    def __init__(self) -> None:
        """Initialize the AuthManager."""
        self._current_role: Role | None = None

    @property
    def current_role(self) -> Role | None:
        """Get the currently authenticated role."""
        return self._current_role

    def login(self, role: Role | str, password: str = "") -> bool:
        """Authenticate and log in a user with a specific role.

        Args:
            role: The target role (Operator or Admin).
            password: Password required for the Admin role.

        Returns:
            True if login was successful, False otherwise.
        """
        if not role:
            return False

        try:
            role_enum = Role(role)
        except ValueError:
            return False

        if role_enum == Role.OPERATOR:
            # Operator does not require a password by default
            self._current_role = Role.OPERATOR
            return True

        if self.verify_admin_password(password):
            self._current_role = Role.ADMIN
            return True

        return False

    def logout(self) -> None:
        """Log out the current user."""
        self._current_role = None

    def is_authenticated(self) -> bool:
        """Check if a user is currently logged in."""
        return self._current_role is not None

    def is_admin(self) -> bool:
        """Check if the currently logged in user is an Admin."""
        return self._current_role == Role.ADMIN

    @staticmethod
    def admin_credential_configured() -> bool:
        """Return True if an Admin credential is configured (see module docs)."""
        return admin_credential_configured()

    def verify_admin_password(self, password: str) -> bool:
        """Verify the provided password against the configured Admin credential.

        Uses a salted PBKDF2-HMAC-SHA256 key derivation and constant-time
        comparison. There is **no hardcoded default password**: if no credential
        is configured (neither ``ADMIN_PASSWORD_HASH`` nor ``ADMIN_PASSWORD``),
        this method fails closed and returns ``False``.

        Args:
            password: The password string to verify.

        Returns:
            True if valid, False otherwise.
        """
        if not password:
            return False

        # Preferred: a stored salted hash, so the plaintext is never in the env.
        encoded = os.environ.get("ADMIN_PASSWORD_HASH")
        if encoded:
            return _verify_against_encoded(password, encoded)

        # Convenience: plaintext password in env (bench/dev). Still salted+KDF'd.
        env_password = os.environ.get("ADMIN_PASSWORD")
        if env_password:
            # Hash both sides with a derived per-process salt and compare in
            # constant time. Using the configured password as the source of a
            # deterministic salt keeps this stateless while avoiding bare SHA-256.
            salt = hashlib.sha256(b"p1am-admin-salt:" + env_password.encode()).digest()
            expected = _pbkdf2(env_password, salt, _PBKDF2_ITERATIONS)
            provided = _pbkdf2(password, salt, _PBKDF2_ITERATIONS)
            return hmac.compare_digest(provided, expected)

        # Fail closed: no credential configured.
        return False


def _main() -> int:
    """CLI helper: hash a password for use as ADMIN_PASSWORD_HASH."""
    import argparse
    import getpass
    import sys

    parser = argparse.ArgumentParser(
        description="Generate a salted ADMIN_PASSWORD_HASH for the P1AM HMI.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=_PBKDF2_ITERATIONS,
        help=f"PBKDF2 iteration count (default: {_PBKDF2_ITERATIONS})",
    )
    args = parser.parse_args()

    pw1 = getpass.getpass("New admin password: ")
    pw2 = getpass.getpass("Confirm admin password: ")
    if pw1 != pw2:
        sys.stderr.write("Passwords do not match.\n")
        return 1
    if not pw1:
        sys.stderr.write("Password must be non-empty.\n")
        return 1
    encoded = hash_admin_password(pw1, iterations=args.iterations)
    sys.stdout.write("\nSet this in your environment (do NOT commit it):\n")
    sys.stdout.write(f'ADMIN_PASSWORD_HASH="{encoded}"\n')
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
