from pdf_renamer.renamer import Renamer


def test_renamer_styles() -> None:
    # Standard
    r_std = Renamer(style="standard")
    assert r_std.generate_new_filename("John Doe", "My Report") == "Doe - My Report.pdf"

    # Snake Case
    r_snake = Renamer(style="snake_case")
    assert r_snake.generate_new_filename("John Doe", "My Report") == "doe_my_report.pdf"
    assert (
        r_snake.generate_new_filename("John Doe", "Introduction to Python")
        == "doe_introduction_to_python.pdf"
    )

    # Kebab Case
    r_kebab = Renamer(style="kebab_case")
    assert r_kebab.generate_new_filename("John Doe", "My Report") == "doe-my-report.pdf"

    # Edge cases
    assert r_snake.generate_new_filename("", "") == "untitled_untitled.pdf"
    assert r_kebab.generate_new_filename("", "") == "untitled-untitled.pdf"
