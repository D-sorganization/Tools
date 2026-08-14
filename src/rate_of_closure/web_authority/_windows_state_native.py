"""Handle-based pywin32 boundary for authority-state security."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, cast

import ntsecuritycon
import pywintypes
import win32api
import win32con
import win32file
import win32security

from .state_security import StateSecurityCode, StateSecurityError

FILE_ALL_ACCESS: Final = ntsecuritycon.FILE_ALL_ACCESS
DIRECTORY_ACE_FLAGS: Final = (
    win32security.OBJECT_INHERIT_ACE | win32security.CONTAINER_INHERIT_ACE
)
FILE_ACE_FLAGS: Final = 0
_OPEN_FLAGS: Final = (
    win32file.FILE_FLAG_OPEN_REPARSE_POINT | win32file.FILE_FLAG_BACKUP_SEMANTICS
)
_OPEN_ACCESS: Final = 0x80 | win32con.READ_CONTROL | win32con.WRITE_DAC
_OPEN_SHARING: Final = win32file.FILE_SHARE_READ | win32file.FILE_SHARE_WRITE
_FILE_ATTRIBUTE_REPARSE_POINT: Final = 0x400
_FILE_PERSISTENT_ACLS: Final = 0x8
_FILE_NAMED_STREAMS: Final = 0x40000
_SE_DACL_PROTECTED: Final = 0x1000
_SYSTEM_SID: Final = "S-1-5-18"
_ADMINISTRATORS_SID: Final = "S-1-5-32-544"


class _AclLike(Protocol):
    def GetAceCount(self) -> int: ...

    def GetAce(self, index: int) -> tuple[tuple[int, int], int, object]: ...


class _SecurityDescriptorLike(Protocol):
    def GetSecurityDescriptorControl(self) -> tuple[int, int]: ...

    def GetSecurityDescriptorDacl(self) -> _AclLike | None: ...

    def GetSecurityDescriptorOwner(self) -> object: ...


class _HandleLike(Protocol):
    def Close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class NativeIdentity:
    """Stable NTFS volume and file identifier from one retained handle."""

    volume_serial: int
    file_index: int


@dataclass(frozen=True, slots=True)
class SecuritySnapshot:
    """Security descriptor retained in memory solely for rollback."""

    descriptor: _SecurityDescriptorLike


@dataclass(slots=True)
class NativeGuard:
    """No-delete-share handle and the identity it pins."""

    api: NativeApi
    handle: object
    path: Path
    is_directory: bool
    identity: NativeIdentity
    link_count: int

    def close(self) -> None:
        handle, self.handle = self.handle, None
        if handle is not None:
            self.api.close_handle(handle)


def _error(code: StateSecurityCode) -> StateSecurityError:
    return StateSecurityError(code)


def _mapped_error(error: pywintypes.error) -> StateSecurityError:
    if error.winerror in {2, 3}:
        return _error(StateSecurityCode.MISSING)
    if error.winerror == 5:
        return _error(StateSecurityCode.ACCESS_DENIED)
    if error.winerror in {32, 33}:
        return _error(StateSecurityCode.PATH_BUSY)
    return _error(StateSecurityCode.OPERATING_SYSTEM_FAILURE)


def _security_information(*, protected: bool) -> int:
    flag = (
        win32security.PROTECTED_DACL_SECURITY_INFORMATION
        if protected
        else win32security.UNPROTECTED_DACL_SECURITY_INFORMATION
    )
    return int(win32security.DACL_SECURITY_INFORMATION | flag)


class NativeApi:
    """Win32 calls with complete ACE comparison and sanitized failures."""

    def __init__(self) -> None:
        process = win32api.GetCurrentProcess()
        token = win32security.OpenProcessToken(process, win32con.TOKEN_QUERY)
        try:
            self.owner_sid = win32security.GetTokenInformation(
                token, win32security.TokenUser
            )[0]
        finally:
            token.Close()
        self.system_sid = win32security.ConvertStringSidToSid(_SYSTEM_SID)
        self.admin_sid = win32security.ConvertStringSidToSid(_ADMINISTRATORS_SID)

    def open_guard(self, path: Path, *, security: bool) -> NativeGuard:
        access = _OPEN_ACCESS if security else 0x80
        try:
            handle = win32file.CreateFile(
                str(path),
                access,
                _OPEN_SHARING,
                None,
                win32file.OPEN_EXISTING,
                _OPEN_FLAGS,
                None,
            )
            return self._guard_from_handle(path, handle)
        except pywintypes.error as error:
            raise _mapped_error(error) from None

    def _guard_from_handle(self, path: Path, handle: object) -> NativeGuard:
        try:
            attributes = win32file.GetFileInformationByHandleEx(
                handle, win32file.FileAttributeTagInfo
            )["FileAttributes"]
            if attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
                raise _error(StateSecurityCode.REPARSE_POINT)
            information = win32file.GetFileInformationByHandle(handle)
            identity = NativeIdentity(
                information[4] & 0xFFFFFFFF,
                (information[8] << 32) | information[9],
            )
            return NativeGuard(
                self,
                handle,
                path,
                bool(attributes & win32file.FILE_ATTRIBUTE_DIRECTORY),
                identity,
                information[7],
            )
        except Exception:
            self.close_handle(handle)
            raise

    def create_private_directory(self, path: Path) -> None:
        attributes = win32security.SECURITY_ATTRIBUTES()
        descriptor = win32security.SECURITY_DESCRIPTOR()
        descriptor.SetSecurityDescriptorOwner(self.owner_sid, False)
        descriptor.SetSecurityDescriptorDacl(
            True,
            self._private_acl(directory=True),
            False,
        )
        descriptor.SetSecurityDescriptorControl(
            _SE_DACL_PROTECTED,
            _SE_DACL_PROTECTED,
        )
        attributes.SECURITY_DESCRIPTOR = descriptor
        try:
            win32file.CreateDirectory(str(path), attributes)
        except pywintypes.error as error:
            raise _mapped_error(error) from None

    def verify_volume(self, volume_root: NativeGuard) -> None:
        try:
            root = str(volume_root.path)
            if win32file.GetDriveType(root) != win32file.DRIVE_FIXED:
                raise _error(StateSecurityCode.UNSUPPORTED_VOLUME)
            information = win32api.GetVolumeInformation(root)
        except pywintypes.error as error:
            raise _mapped_error(error) from None
        serial = information[1] & 0xFFFFFFFF
        flags, filesystem = information[3], information[4]
        required = _FILE_PERSISTENT_ACLS | _FILE_NAMED_STREAMS
        if serial != volume_root.identity.volume_serial:
            raise _error(StateSecurityCode.IDENTITY_CHANGED)
        if flags & required != required or filesystem.upper() != "NTFS":
            raise _error(StateSecurityCode.UNSUPPORTED_VOLUME)

    def assert_default_stream_only(self, guard: NativeGuard) -> None:
        try:
            streams = win32file.GetFileInformationByHandleEx(
                guard.handle,
                win32file.FileStreamInfo,
            )
        except pywintypes.error as error:
            raise _mapped_error(error) from None
        names = tuple(str(stream["StreamName"]) for stream in streams)
        invalid_directory = guard.is_directory and any(
            name != "::$DATA" for name in names
        )
        invalid_file = not guard.is_directory and names != ("::$DATA",)
        if invalid_directory or invalid_file:
            raise _error(StateSecurityCode.UNEXPECTED_STREAM)

    def snapshot(self, handle: object) -> SecuritySnapshot:
        try:
            descriptor = win32security.GetSecurityInfo(
                handle,
                win32security.SE_FILE_OBJECT,
                win32security.OWNER_SECURITY_INFORMATION
                | win32security.DACL_SECURITY_INFORMATION,
            )
        except pywintypes.error as error:
            raise _mapped_error(error) from None
        return SecuritySnapshot(cast(_SecurityDescriptorLike, descriptor))

    def apply_private_acl(self, guard: NativeGuard) -> None:
        try:
            win32security.SetSecurityInfo(
                guard.handle,
                win32security.SE_FILE_OBJECT,
                _security_information(protected=True),
                None,
                None,
                self._private_acl(directory=guard.is_directory),
                None,
            )
        except pywintypes.error as error:
            raise _mapped_error(error) from None

    def restore(self, guard: NativeGuard, snapshot: SecuritySnapshot) -> None:
        descriptor = snapshot.descriptor
        control = descriptor.GetSecurityDescriptorControl()[0]
        protected = bool(control & _SE_DACL_PROTECTED)
        try:
            win32security.SetSecurityInfo(
                guard.handle,
                win32security.SE_FILE_OBJECT,
                _security_information(protected=protected),
                None,
                None,
                descriptor.GetSecurityDescriptorDacl(),
                None,
            )
        except pywintypes.error as error:
            raise _mapped_error(error) from None

    def has_private_acl(self, guard: NativeGuard) -> bool:
        descriptor = self.snapshot(guard.handle).descriptor
        control = descriptor.GetSecurityDescriptorControl()[0]
        if not control & _SE_DACL_PROTECTED:
            return False
        if descriptor.GetSecurityDescriptorOwner() != self.owner_sid:
            return False
        dacl = descriptor.GetSecurityDescriptorDacl()
        if dacl is None or dacl.GetAceCount() != 3:
            return False
        flags = DIRECTORY_ACE_FLAGS if guard.is_directory else FILE_ACE_FLAGS
        expected = sorted(
            (str(sid), FILE_ALL_ACCESS, flags)
            for sid in (self.owner_sid, self.system_sid, self.admin_sid)
        )
        actual = sorted(self._ace_tuple(dacl.GetAce(index)) for index in range(3))
        return actual == expected

    @staticmethod
    def _ace_tuple(
        ace: tuple[tuple[int, int], int, object],
    ) -> tuple[str, int, int]:
        header, mask, sid = ace
        ace_type, flags = header
        if ace_type != win32security.ACCESS_ALLOWED_ACE_TYPE:
            return ("invalid-ace", -1, -1)
        return (str(sid), mask, flags)

    def _private_acl(self, *, directory: bool) -> object:
        acl = win32security.ACL(256)
        flags = DIRECTORY_ACE_FLAGS if directory else FILE_ACE_FLAGS
        for sid in (self.owner_sid, self.system_sid, self.admin_sid):
            acl.AddAccessAllowedAceEx(
                win32security.ACL_REVISION_DS,
                flags,
                FILE_ALL_ACCESS,
                sid,
            )
        return acl

    @staticmethod
    def close_handle(handle: object) -> None:
        try:
            cast(_HandleLike, handle).Close()
        except pywintypes.error as error:
            raise _mapped_error(error) from None
