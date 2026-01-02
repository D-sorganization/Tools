from pdf_renamer.utils import get_last_name, sanitize_filename, to_title_case


def test_to_title_case() -> None:
    assert to_title_case("the lord of the rings") == "The Lord of the Rings"
    assert to_title_case("a tale of two cities") == "A Tale of Two Cities"
    assert to_title_case("of mice and men") == "Of Mice and Men"
    assert to_title_case("THE GREAT GATSBY") == "The Great Gatsby"
    assert to_title_case("word") == "Word"
    assert to_title_case("") == ""


def test_sanitize_filename() -> None:
    assert sanitize_filename("valid-name") == "valid-name"
    assert sanitize_filename("invalid/name:test") == "invalidnametest"
    assert sanitize_filename("foo/bar\\baz") == "foobarbaz"


def test_get_last_name() -> None:
    assert get_last_name("John Doe") == "Doe"
    assert get_last_name("John Jacob Jingleheimer Schmidt") == "Schmidt"
    assert get_last_name("Cher") == "Cher"
    assert get_last_name("") == "Unknown"
