import unittest

from streamlit.testing.v1 import AppTest


class StreamlitAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = AppTest.from_file("app.py", default_timeout=20).run()

    def assert_clean_run(self) -> None:
        self.assertEqual(list(self.app.exception), [])

    def test_default_overview_renders(self) -> None:
        self.assert_clean_run()
        self.assertEqual(
            self.app.title[0].value,
            "Do prediction-market features improve event-window equity forecasts?",
        )
        self.assertGreaterEqual(len(self.app.metric), 4)

    def test_all_navigation_pages_render(self) -> None:
        pages = {
            "Scenario Signals": "Simulated event-window scenario inputs",
            "Backtest": "Performance, robustness and model-selection evidence",
            "Agent Pipeline": "From prediction-market contract to equity decision",
            "Methodology": "Hypotheses, variables and identification",
            "Data & Limitations": "Data coverage, reproducibility and limitations",
        }
        for page, title in pages.items():
            self.app.sidebar.radio[0].set_value(page).run()
            self.assert_clean_run()
            self.assertEqual(self.app.title[0].value, title)

    def test_live_signal_empty_filters_do_not_crash(self) -> None:
        self.app.sidebar.radio[0].set_value("Scenario Signals").run()
        self.app.multiselect[0].set_value([])
        self.app.multiselect[1].set_value([])
        self.app.multiselect[2].set_value([]).run()
        self.assert_clean_run()


if __name__ == "__main__":
    unittest.main()
