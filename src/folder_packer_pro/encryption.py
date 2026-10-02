"""Encryption utilities for Folder Packer Pro.

Provides AES-256 encryption and decryption using PBKDF2 key derivation.
"""

from __future__ import annotations

import base64
import os

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class EncryptionManager:
    """Handle encryption/decryption of packed files."""

    @staticmethod
    def derive_key(password: str, salt: bytes, iterations: int = 600000) -> bytes:
        """Derive encryption key from password using PBKDF2.

        Args:
            password: User password
            salt: Random salt bytes
            iterations: PBKDF2 iteration count

        Returns:
            32-byte encryption key
        """
        if password is None:
            raise ValueError("password must be provided")
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    @staticmethod
    def encrypt_data(data: bytes, password: str) -> bytes:
        """Encrypt data with password using AES-256.

        Args:
            data: Data to encrypt
            password: Encryption password

        Returns:
            Encrypted data with salt prepended
        """
        if data is None:
            raise ValueError("data must be provided")
        salt = os.urandom(16)
        # Security enhancement: Use 600,000 iterations for new encryptions
        # (OWASP recommended)
        key = EncryptionManager.derive_key(password, salt, iterations=600000)
        cipher = Fernet(key)
        encrypted: bytes = cipher.encrypt(data)
        result: bytes = salt + encrypted
        return result

    @staticmethod
    def decrypt_data(encrypted_data: bytes, password: str) -> bytes:
        """Decrypt data with password.

        Args:
            encrypted_data: Encrypted data with salt prepended
            password: Decryption password

        Returns:
            Decrypted data
        """
        import cryptography.fernet

        if encrypted_data is None:
            raise ValueError("encrypted_data must be provided")
        salt = encrypted_data[:16]
        encrypted = encrypted_data[16:]

        # Attempt decryption with modern secure iteration count (600,000)
        key = EncryptionManager.derive_key(password, salt, iterations=600000)
        cipher = Fernet(key)
        try:
            decrypted: bytes = cipher.decrypt(encrypted)
            return decrypted
        except cryptography.fernet.InvalidToken:
            # Fallback for legacy archives encrypted with weak 100,000 iteration count
            key_legacy = EncryptionManager.derive_key(password, salt, iterations=100000)
            cipher_legacy = Fernet(key_legacy)
            decrypted_legacy: bytes = cipher_legacy.decrypt(encrypted)
            return decrypted_legacy
