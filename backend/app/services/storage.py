from __future__ import annotations

import codecs
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile

from app.core.config import settings

CHUNK_SIZE = 64 * 1024
PDF_HEADER = b"%PDF-"
STORAGE_NAME_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.(pdf|txt)$"
)


class FileTooLargeError(Exception):
    """Uploaded file exceeded the configured size limit."""


class UnsupportedFileTypeError(Exception):
    """File is not an allowed PDF or UTF-8 text document."""


class InvalidFileContentError(Exception):
    """File signature or contents failed validation."""


class StoredFileMissingError(Exception):
    """Metadata points at a storage path that is not on disk."""


@dataclass(frozen=True)
class StoredFile:
    storage_path: str
    content_type: str
    filename: str
    size: int


class LocalStorage:
    def __init__(self, root: Path, max_bytes: int) -> None:
        self.root = root
        self.max_bytes = max_bytes

    def save_upload(self, upload: UploadFile) -> StoredFile:
        self.root.mkdir(parents=True, exist_ok=True)
        original_name = _safe_original_filename(upload.filename)
        temp_path = self.root / f".tmp-{uuid.uuid4().hex}"
        detected_type: str | None = None
        size = 0
        text_decoder = codecs.getincrementaldecoder("utf-8")("strict")

        try:
            with temp_path.open("wb") as buffer:
                while True:
                    chunk = upload.file.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > self.max_bytes:
                        raise FileTooLargeError("File exceeds the maximum upload size")
                    if detected_type is None:
                        detected_type = _detect_type(chunk)
                    if detected_type == "text/plain":
                        if b"\x00" in chunk:
                            raise InvalidFileContentError("Text files must not contain binary data")
                        try:
                            text_decoder.decode(chunk)
                        except UnicodeDecodeError as exc:
                            raise InvalidFileContentError("Text files must be valid UTF-8") from exc
                    buffer.write(chunk)

            if size == 0 or detected_type is None:
                raise InvalidFileContentError("Uploaded file is empty")

            if detected_type == "text/plain":
                try:
                    text_decoder.decode(b"", final=True)
                except UnicodeDecodeError as exc:
                    raise InvalidFileContentError("Text files must be valid UTF-8") from exc
            else:
                _validate_pdf(temp_path)

            extension = ".pdf" if detected_type == "application/pdf" else ".txt"
            storage_path = f"{uuid.uuid4()}{extension}"
            final_path = self.root / storage_path
            temp_path.replace(final_path)
            return StoredFile(
                storage_path=storage_path,
                content_type=detected_type,
                filename=_filename_with_extension(original_name, extension),
                size=size,
            )
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    def resolve(self, storage_path: str) -> Path:
        if not STORAGE_NAME_RE.match(storage_path):
            raise StoredFileMissingError("Stored file not found")
        path = (self.root / storage_path).resolve()
        root = self.root.resolve()
        if root not in path.parents and path != root:
            raise StoredFileMissingError("Stored file not found")
        if not path.is_file():
            raise StoredFileMissingError("Stored file not found")
        return path

    def delete(self, storage_path: str | None) -> None:
        if not storage_path:
            return
        try:
            self.resolve(storage_path).unlink()
        except StoredFileMissingError:
            return


def get_storage() -> LocalStorage:
    return LocalStorage(root=settings.upload_dir, max_bytes=settings.max_upload_bytes)


def _detect_type(first_chunk: bytes) -> str:
    if first_chunk.startswith(PDF_HEADER):
        return "application/pdf"
    if b"\x00" in first_chunk:
        raise UnsupportedFileTypeError("Only PDF and plain-text files are supported")
    sample = first_chunk.lstrip(b"\xef\xbb\xbf")
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UnsupportedFileTypeError("Only PDF and plain-text files are supported") from exc
    if _looks_binary(first_chunk):
        raise UnsupportedFileTypeError("Only PDF and plain-text files are supported")
    return "text/plain"


def _looks_binary(data: bytes) -> bool:
    if not data:
        return False
    control = sum(1 for byte in data if byte < 9 or (13 < byte < 32) or byte == 127)
    return (control / len(data)) > 0.30


def _validate_pdf(path: Path) -> None:
    with path.open("rb") as handle:
        header = handle.read(8)
        if not header.startswith(PDF_HEADER):
            raise InvalidFileContentError("File is not a valid PDF")
        handle.seek(0, 2)
        length = handle.tell()
        handle.seek(max(0, length - 1024))
        tail = handle.read()
    if b"%%EOF" not in tail:
        raise InvalidFileContentError("File is not a valid PDF")


def _safe_original_filename(name: str | None) -> str:
    candidate = Path(name or "document").name.strip()
    if not candidate or candidate in {".", ".."}:
        return "document"
    return candidate[:255]


def _filename_with_extension(name: str, extension: str) -> str:
    stem = Path(name).stem or "document"
    filename = f"{stem}{extension}"
    return filename[:255]
