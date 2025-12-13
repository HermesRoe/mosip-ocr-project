from pathlib import Path
import pytest


def get_file_extensions(filename: str) -> str | None:
    if not filename:
        return None
    file_extension = Path(filename).suffix.lower().replace(".", "").strip()
    return file_extension if file_extension != "" else None


@pytest.mark.parametrize("filename, expected", [
    ("document.pdf", "pdf"),
    ("image.jpg", "jpg"),
    ("script.js", "js"),
    ("archive.tar.gz", "gz"),
    ("photo.jpeg", "jpeg"),
    ("IMAGE.PNG", "png"),
    ("FiLeNaMe.tXT", "txt"),
    ("eXeCuTaBlE.ExE", "exe"),
    ("ZIPFILE.ZIP", "zip"),
    ("DOC.DocX", "docx"),
    ("configfile", None),
    ("README", None),
    ("no.ext.", None),
    ("file", None),
    (".bashrc", None),
    (".env.local", "local"),
    (".gitignore.txt", "txt"),
    ("backup.2023.10.27.tar.gz", "gz"),
    ("version.1.0.0.txt", "txt"),
    ("file.with.many.dots", "dots"),
    ("", None),
    (".", None),
    ("..", None),
    ("file_name.", None),
    ("file_name. ", None)
]
)
def test_get_file_extensions(filename, expected):  # type: ignore
    assert get_file_extensions(filename) == expected  # type: ignore
