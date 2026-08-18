"""UI labels and bounds shared by the variation controls."""

MODE_LABELS: dict[str, str] = {
    "delivery": "Delivery → Impact → Flight",
    "swing": "Pendulum Swing → Impact → Flight",
    "launch": "Launch Conditions → Flight",
}
BASE_SOURCES = ("Registry Defaults", "Explorer Scenario")
MAX_RUNS = 5000

__all__ = ["BASE_SOURCES", "MAX_RUNS", "MODE_LABELS"]
