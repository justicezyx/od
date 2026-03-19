import unittest
from unittest.mock import Mock, patch

from btc_agent import (
    DEFAULT_BTC_PRICE_USD,
    BitcoinEarningAgent,
    BuilderProfile,
)


class TestBitcoinEarningAgent(unittest.TestCase):
    def setUp(self):
        self.agent = BitcoinEarningAgent()

    def test_rank_opportunities_returns_sorted_scores(self):
        profile = BuilderProfile(
            skills=["python", "flask", "marketing", "sales"],
            weekly_hours=15,
            budget_usd=200.0,
            risk_tolerance="medium",
        )

        ranked = self.agent.rank_opportunities(profile=profile, top_n=3, btc_price_usd=70000.0)

        self.assertEqual(len(ranked), 3)
        self.assertGreaterEqual(ranked[0].total_score, ranked[1].total_score)
        self.assertGreaterEqual(ranked[1].total_score, ranked[2].total_score)
        self.assertGreater(ranked[0].expected_monthly_btc, 0)

    @patch("btc_agent.requests.get")
    def test_fetch_btc_price_uses_fallback_on_errors(self, mock_get):
        mock_get.side_effect = RuntimeError("network down")
        price = self.agent.fetch_btc_price_usd()
        self.assertEqual(price, DEFAULT_BTC_PRICE_USD)

    @patch("btc_agent.requests.get")
    def test_fetch_btc_price_success(self, mock_get):
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"bitcoin": {"usd": 90000}}
        mock_get.return_value = mock_response

        price = self.agent.fetch_btc_price_usd()
        self.assertEqual(price, 90000.0)

    def test_build_execution_plan_contains_required_sections(self):
        profile = BuilderProfile(
            skills=["python", "apis", "devops", "sales"],
            weekly_hours=10,
            budget_usd=0.0,
            risk_tolerance="low",
        )
        plan = self.agent.build_execution_plan(profile=profile, top_n=2, btc_price_usd=80000.0)

        self.assertIn("top_opportunities", plan)
        self.assertEqual(len(plan["top_opportunities"]), 2)
        self.assertIn("first_14_day_plan", plan)
        self.assertEqual(len(plan["first_14_day_plan"]), 5)
        self.assertIn("disclaimer", plan)


if __name__ == "__main__":
    unittest.main()
