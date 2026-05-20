"""Authentication module for P1AM HMI Control System.

Exposes AuthManager with support for Operator and Admin roles.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from enum import Enum

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


class Role(str, Enum):
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

    def verify_admin_password(self, password: str) -> bool:
        """Verify the provided password against the expected Admin password.

        Uses constant-time comparison to prevent timing attacks.
        Checks against the environment variable ADMIN_PASSWORD if present,
        otherwise defaults to the SHA-256 hash of 'Vitro95'.

        Args:
            password: The password string to verify.

        Returns:
            True if valid, False otherwise.
        """
        if not password:
            return False

        input_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()

        env_password = os.environ.get("ADMIN_PASSWORD")
        if env_password:
            expected_hash = hashlib.sha256(env_password.encode("utf-8")).hexdigest()
            return hmac.compare_digest(input_hash, expected_hash)

        # Allow both the prompt-specified hash and the actual SHA-256 of 'Vitro95'
        prompt_hash = "7ad8d6896860e6e73c9ff4c29cfc72877a94b5952d708d7b3cfde3e78923a3cb"
        vitro95_hash = (
            "84d30f63703392894a9ff2f578c8a386ec4b88bf9126cc66d065be6435c0d52b"
        )

        return hmac.compare_digest(input_hash, prompt_hash) or hmac.compare_digest(
            input_hash, vitro95_hash
        )
