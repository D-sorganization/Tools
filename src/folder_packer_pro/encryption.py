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
    def derive_key(password: str, salt: bytes) -> bytes:
        """Derive encryption key from password using PBKDF2.

        Args:
            password: User password
            salt: Random salt bytes

        Returns:
            32-byte encryption key
        """
        assert password is not None, "password must be provided"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
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
        assert data is not None, "data must be provided"
        salt = os.urandom(16)
        key = EncryptionManager.derive_key(password, salt)
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
        assert encrypted_data is not None, "encrypted_data must be provided"
        salt = encrypted_data[:16]
        encrypted = encrypted_data[16:]
        key = EncryptionManager.derive_key(password, salt)
        cipher = Fernet(key)
        decrypted: bytes = cipher.decrypt(encrypted)
        return decrypted
