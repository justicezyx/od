"""A practical BTC-earning planning agent for online builders.

This module does not promise profits. It creates a ranked build plan based on:
- your skills and available hours
- estimated time-to-launch
- monetization channels that can pay in BTC
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

import requests

DEFAULT_BTC_PRICE_USD = 65000.0


@dataclass(frozen=True)
class BuilderProfile:
    skills: Sequence[str]
    weekly_hours: int
    budget_usd: float
    risk_tolerance: str = "medium"


@dataclass(frozen=True)
class Opportunity:
    name: str
    description: str
    required_skills: Sequence[str]
    setup_hours: int
    market_demand: float
    competition: float
    base_monthly_usd: float
    channels: Sequence[str]
    first_revenue_days: int


@dataclass(frozen=True)
class OpportunityScore:
    opportunity: Opportunity
    total_score: float
    skill_fit: float
    execution_fit: float
    speed_score: float
    risk_score: float
    expected_monthly_btc: float

    def as_dict(self) -> Dict[str, object]:
        data = asdict(self)
        # Keep output shape clean for downstream serialization.
        data["opportunity"] = asdict(self.opportunity)
        return data


class BitcoinEarningAgent:
    """Ranks online build ideas that can be monetized for BTC."""

    def __init__(self) -> None:
        self.opportunities = self._seed_opportunities()

    def fetch_btc_price_usd(self) -> float:
        """Get live BTC price with a safe fallback."""
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": "bitcoin", "vs_currencies": "usd"}
        try:
            response = requests.get(url, params=params, timeout=8)
            response.raise_for_status()
            payload = response.json()
            price = float(payload["bitcoin"]["usd"])
            if price <= 0:
                return DEFAULT_BTC_PRICE_USD
            return price
        except (requests.RequestException, ValueError, KeyError, TypeError):
            return DEFAULT_BTC_PRICE_USD

    def rank_opportunities(
        self, profile: BuilderProfile, top_n: int = 3, btc_price_usd: float | None = None
    ) -> List[OpportunityScore]:
        if btc_price_usd is None:
            btc_price_usd = self.fetch_btc_price_usd()

        scores: List[OpportunityScore] = []
        normalized_skills = {s.strip().lower() for s in profile.skills if s.strip()}
        effective_hours = max(profile.weekly_hours, 1)
        risk_multiplier = self._risk_multiplier(profile.risk_tolerance)

        for opp in self.opportunities:
            required = {s.strip().lower() for s in opp.required_skills}
            overlap = len(required.intersection(normalized_skills))
            skill_fit = overlap / max(len(required), 1)
            execution_fit = min((effective_hours * 4) / max(opp.setup_hours, 1), 1.2)
            speed_score = max(0.1, 1.0 - (opp.first_revenue_days / 120.0))
            risk_score = max(0.1, 1.0 - abs(opp.competition - risk_multiplier))

            # Weighted score: prioritize demand and your ability to ship quickly.
            total_score = (
                (opp.market_demand * 0.35)
                + (skill_fit * 0.25)
                + (execution_fit * 0.20)
                + (speed_score * 0.10)
                + (risk_score * 0.10)
            )

            expected_usd = opp.base_monthly_usd * (0.5 + skill_fit * 0.7) * min(execution_fit, 1.0)
            expected_btc = expected_usd / max(btc_price_usd, 1.0)

            scores.append(
                OpportunityScore(
                    opportunity=opp,
                    total_score=round(total_score, 4),
                    skill_fit=round(skill_fit, 4),
                    execution_fit=round(execution_fit, 4),
                    speed_score=round(speed_score, 4),
                    risk_score=round(risk_score, 4),
                    expected_monthly_btc=round(expected_btc, 8),
                )
            )

        scores.sort(key=lambda item: item.total_score, reverse=True)
        return scores[: max(top_n, 1)]

    def build_execution_plan(
        self, profile: BuilderProfile, top_n: int = 3, btc_price_usd: float | None = None
    ) -> Dict[str, object]:
        ranked = self.rank_opportunities(profile=profile, top_n=top_n, btc_price_usd=btc_price_usd)
        selected = ranked[0]

        plan = {
            "objective": "Build online products/services that can be paid in BTC",
            "profile": asdict(profile),
            "btc_price_usd": btc_price_usd if btc_price_usd is not None else self.fetch_btc_price_usd(),
            "top_opportunities": [item.as_dict() for item in ranked],
            "first_14_day_plan": [
                {
                    "day_range": "Day 1-2",
                    "task": f"Interview 8-10 target users for '{selected.opportunity.name}' and capture pain points.",
                },
                {
                    "day_range": "Day 3-5",
                    "task": "Build a minimal MVP landing page, pricing, and BTC payment path (Lightning or on-chain).",
                },
                {
                    "day_range": "Day 6-9",
                    "task": "Ship core MVP feature, onboard 3 pilot users, and collect usability feedback.",
                },
                {
                    "day_range": "Day 10-12",
                    "task": "Launch distribution posts on communities where your users already hang out.",
                },
                {
                    "day_range": "Day 13-14",
                    "task": "Review conversion metrics, adjust offer, and create the next 2-week iteration backlog.",
                },
            ],
            "disclaimer": (
                "This agent gives structured guidance, not guaranteed returns. "
                "Only pursue legal/ethical opportunities and verify platform terms."
            ),
        }
        return plan

    @staticmethod
    def _risk_multiplier(risk_tolerance: str) -> float:
        tolerance = risk_tolerance.strip().lower()
        mapping = {"low": 0.35, "medium": 0.6, "high": 0.85}
        return mapping.get(tolerance, 0.6)

    @staticmethod
    def _seed_opportunities() -> List[Opportunity]:
        return [
            Opportunity(
                name="BTCPay setup + integration service",
                description="Offer BTC payment integrations for online stores and creators.",
                required_skills=["python", "apis", "devops", "sales"],
                setup_hours=35,
                market_demand=0.78,
                competition=0.52,
                base_monthly_usd=1800.0,
                channels=["freelance", "consulting", "retainers"],
                first_revenue_days=14,
            ),
            Opportunity(
                name="Lightning-powered micro-SaaS",
                description="Build a niche software tool with low-ticket recurring BTC payments.",
                required_skills=["python", "flask", "product", "marketing"],
                setup_hours=55,
                market_demand=0.81,
                competition=0.69,
                base_monthly_usd=2600.0,
                channels=["subscriptions", "usage-based"],
                first_revenue_days=21,
            ),
            Opportunity(
                name="Bitcoin education membership",
                description="Publish premium tutorials/templates and charge monthly in BTC.",
                required_skills=["content", "marketing", "community", "sales"],
                setup_hours=28,
                market_demand=0.66,
                competition=0.71,
                base_monthly_usd=1200.0,
                channels=["memberships", "digital-products", "sponsorships"],
                first_revenue_days=10,
            ),
            Opportunity(
                name="Open-source bounty automation toolkit",
                description="Build tools for maintainers and get paid via bounties/sponsorships.",
                required_skills=["python", "automation", "oss", "security"],
                setup_hours=45,
                market_demand=0.58,
                competition=0.4,
                base_monthly_usd=1400.0,
                channels=["bounties", "sponsorships", "support-contracts"],
                first_revenue_days=18,
            ),
            Opportunity(
                name="Nostr growth analytics dashboard",
                description="Track creator growth and tips, monetized with paid BTC analytics tiers.",
                required_skills=["python", "analytics", "frontend", "product"],
                setup_hours=60,
                market_demand=0.64,
                competition=0.5,
                base_monthly_usd=2100.0,
                channels=["subscriptions", "pro-features"],
                first_revenue_days=25,
            ),
        ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a BTC earning build plan.")
    parser.add_argument(
        "--skills",
        type=str,
        required=True,
        help="Comma-separated skills, e.g. python,flask,marketing,sales",
    )
    parser.add_argument("--hours", type=int, required=True, help="Weekly hours available")
    parser.add_argument("--budget", type=float, default=0.0, help="Budget in USD")
    parser.add_argument("--risk", type=str, default="medium", help="low|medium|high")
    parser.add_argument("--top-n", type=int, default=3, help="Number of opportunities to return")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    profile = BuilderProfile(
        skills=[part.strip() for part in args.skills.split(",") if part.strip()],
        weekly_hours=args.hours,
        budget_usd=args.budget,
        risk_tolerance=args.risk,
    )

    agent = BitcoinEarningAgent()
    plan = agent.build_execution_plan(profile=profile, top_n=args.top_n)

    if args.json:
        print(json.dumps(plan, indent=2))
        return

    top = plan["top_opportunities"][0]
    print(f"Top opportunity: {top['opportunity']['name']}")
    print(f"Expected monthly BTC: ~{top['expected_monthly_btc']}")
    print("14-day launch plan:")
    for step in plan["first_14_day_plan"]:
        print(f"- {step['day_range']}: {step['task']}")


if __name__ == "__main__":
    main()
