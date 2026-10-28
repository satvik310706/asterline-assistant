"""Deterministic, evidence-backed calculations for common pricing and SLA questions."""

from __future__ import annotations

import re

from .models import AnswerDraft, EvidencePack, RetrievedChunk


def _item(evidence: EvidencePack, chunk_id: str) -> RetrievedChunk | None:
    return next((item for item in evidence.chunks if item.chunk.chunk_id == chunk_id), None)


def _number(value: str) -> float:
    return float(value.replace(",", ""))


def _quantity(value: str, unit: str | None) -> float:
    scale = {"thousand": 1_000, "k": 1_000, "million": 1_000_000, "m": 1_000_000, "billion": 1_000_000_000}.get((unit or "").lower(), 1)
    return _number(value) * scale


def _plan(question: str) -> str | None:
    found = re.findall(r"\b(launch|team|enterprise)\b", question, flags=re.IGNORECASE)
    unique = {plan.lower() for plan in found}
    return next(iter(unique)) if len(unique) == 1 else None


def _citations(evidence: EvidencePack, chunk_ids: tuple[str, ...]) -> str | None:
    selected = [_item(evidence, chunk_id) for chunk_id in chunk_ids]
    if any(item is None for item in selected):
        return None
    return " ".join(f"[{item.chunk.chunk_id} — {item.chunk.title}]" for item in selected if item)


def _draft(answer: str, evidence: EvidencePack, chunk_ids: tuple[str, ...]) -> AnswerDraft | None:
    citations = _citations(evidence, chunk_ids)
    if citations is None:
        # Never calculate from a source that the retriever did not put in the pack.
        return None
    return AnswerDraft(
        answer=f"{answer} {citations}",
        cited_chunk_ids=list(chunk_ids),
        calculated_values=re.findall(r"(?<![A-Za-z])\$?\d[\d,]*(?:\.\d+)?%?", answer),
        trace_metadata={"provider": "calculator", "model": "deterministic", "attempts": 0, "degraded": False},
    )


def _plan_price_and_seats(text: str, plan: str) -> tuple[float, int] | None:
    match = re.search(
        rf"\b{plan}\s+costs\s+\$?([\d,]+)\s+per month and includes\s+([\d,]+)\s+seats\b",
        text,
        flags=re.IGNORECASE,
    )
    return (_number(match.group(1)), int(match.group(2).replace(",", ""))) if match else None


def calculate(question: str, evidence: EvidencePack) -> AnswerDraft | None:
    """Return a deterministic answer only when all required facts are in evidence."""
    normalized = question.casefold()
    pricing = _item(evidence, "pricing:0")
    pricing_details = _item(evidence, "pricing:1")
    retention = _item(evidence, "retention:0")
    sla = _item(evidence, "sla:0")

    # A later product specification is authoritative for current per-project limits;
    # cite the release history too when it records the prior value explicitly.
    if "ingestion rate" in normalized and "concurrency" not in normalized and not re.search(r"\b(?:version|in|at)\s+\d+(?:\.\d+)+", normalized):
        plan = _plan(question)
        ingestion = _item(evidence, "ingest:1")
        if plan and ingestion:
            rates = {name.casefold(): int(value) for value, name in re.findall(r"(\d+)\s+(?:requests per second\s+)?on\s+(Launch|Team|Enterprise)", ingestion.chunk.text, re.IGNORECASE)}
            if plan in rates:
                ids = ["ingest:1"]
                text = f"The current {plan.title()} ingestion limit is {rates[plan]} requests per second per project."
                if plan == "launch":
                    release = _item(evidence, "release-21:0")
                    changed = re.search(r"Launch ingestion increased from\s+(\d+)\s+to\s+(\d+)\s+requests per second per project", release.chunk.text, re.IGNORECASE) if release else None
                    if changed:
                        text += f" The earlier {changed.group(1)} requests per second limit was superseded when version 2.1 increased it to {changed.group(2)}."
                        ids.append("release-21:0")
                return _draft(text, evidence, tuple(ids))

    # State both searchable retention and cold recovery without conflating the two.
    if "cold" in normalized and "search" in normalized and "permanently deleted" not in normalized:
        plan = _plan(question)
        if plan and pricing_details and retention:
            searchable = re.search(rf"\b{plan}\b[^.;]*?\bfor\s+(\d+)\s+days\b", pricing_details.chunk.text, re.IGNORECASE)
            cold = re.search(r"cold recovery storage for\s+(\d+)\s+days", retention.chunk.text, re.IGNORECASE)
            if searchable and cold:
                ids = ["pricing:1", "retention:0"]
                explorer = _item(evidence, "retention:1")
                if explorer:
                    ids.append("retention:1")
                text = f"{plan.title()} run records remain searchable for {searchable.group(1)} days. Cold recovery adds {cold.group(1)} days after that period, but it is not searchable and therefore does not extend searchable retention."
                return _draft(text, evidence, tuple(ids))

    # Permanent deletion is searchable retention plus the separately stated cold period.
    if "permanently deleted" in normalized and ("after ingestion" in normalized or "from ingestion" in normalized):
        plan = _plan(question)
        if plan and pricing_details and retention:
            searchable = re.search(rf"\b{plan}\b[^.;]*?\bfor\s+(\d+)\s+days\b", pricing_details.chunk.text, re.IGNORECASE)
            cold = re.search(r"cold recovery storage for\s+(\d+)\s+days", retention.chunk.text, re.IGNORECASE)
            if searchable and cold:
                total = int(searchable.group(1)) + int(cold.group(1))
                return _draft(
                    f"{plan.title()} searchable retention is {searchable.group(1)} days, then cold recovery adds {cold.group(1)} days; permanent deletion is after {total} days from ingestion. Cold recovery is not searchable.",
                    evidence,
                    ("pricing:1", "retention:0"),
                )

    # Monthly usage overage: (used records - included records) / 10,000 * published rate.
    if "overage" in normalized and ("record" in normalized or "run" in normalized):
        plan = _plan(question)
        usage = re.search(r"([\d,.]+)\s*(million|thousand|billion|m|k)?\s+(?:run\s+)?records", question, re.IGNORECASE)
        if plan == "enterprise" and pricing_details and re.search(r"Enterprise overage is negotiated", pricing_details.chunk.text, re.IGNORECASE):
            return _draft("Enterprise run-record overage is negotiated; the corpus does not publish a numeric overage price.", evidence, ("pricing:1",))
        if plan and usage and pricing and pricing_details:
            included = re.search(
                rf"\b{plan}\s+costs.*?includes\s+[\d,]+\s+seats and\s+([\d,.]+)\s+(million|thousand|billion|m|k)?\s+run records per month",
                pricing.chunk.text,
                re.IGNORECASE,
            )
            rate = re.search(rf"\$?([\d.]+) per 10,000(?: records)? on {plan}\b", pricing_details.chunk.text, re.IGNORECASE)
            if included and rate:
                used_records = _quantity(usage.group(1), usage.group(2))
                included_records = _quantity(included.group(1), included.group(2))
                units = max(0, used_records - included_records) / 10_000
                cost = units * _number(rate.group(1))
                if used_records <= included_records:
                    text = f"{plan.title()} usage of {used_records:,.0f} records does not exceed the included {included_records:,.0f} monthly records, so no record overage is due."
                else:
                    text = f"{plan.title()} includes {included_records:,.0f} records monthly. {used_records:,.0f} records leaves {units:,.0f} overage units at ${rate.group(1)} per 10,000, for ${cost:,.2f}."
                return _draft(text, evidence, ("pricing:0", "pricing:1"))

    # Annual billing discounts apply to the published Launch/Team base subscription only.
    if re.search(r"\b(annual|yearly)\b", normalized) and "discount" in normalized and "enterprise" in normalized:
        discount_source = _item(evidence, "pricing:0")
        if discount_source:
            discount = re.search(r"annual billing gives a\s+(\d+)% discount on Launch and Team base subscriptions", discount_source.chunk.text, re.IGNORECASE)
            if discount:
                rate = int(discount.group(1))
                return _draft(
                    f"Annual billing gives a {rate}% discount on Launch and Team base subscriptions only. Enterprise pricing is negotiated, and the corpus does not state an Enterprise annual discount.",
                    evidence,
                    ("pricing:0",),
                )

    if re.search(r"\b(annual|yearly)\b", normalized) and re.search(r"\b(cost|price|bill|billing|total)\b", normalized):
        plan = _plan(question)
        if plan in {"launch", "team"} and pricing:
            base = _plan_price_and_seats(pricing.chunk.text, plan)
            discount = re.search(r"annual billing gives a\s+(\d+)% discount on Launch and Team base subscriptions", pricing.chunk.text, re.IGNORECASE)
            if base and discount:
                monthly, _ = base
                rate = int(discount.group(1))
                annual = monthly * 12 * (1 - rate / 100)
                return _draft(
                    f"The {plan.title()} base subscription is ${monthly:,.0f} per month. Annual billing applies a {rate}% discount to that base only: ${monthly:,.0f} × 12 × {1-rate/100:.2f} = ${annual:,.2f} per year, before extra seats.",
                    evidence,
                    ("pricing:0",),
                )

    # Total seat-cost questions use the included seats and published extra-seat rate.
    if "seat" in normalized and re.search(r"\b(total|including|including the base|monthly cost|cost)\b", normalized):
        plan = _plan(question)
        requested = re.search(r"\b(\d[\d,]*)\s+seats?\b", question, re.IGNORECASE)
        if plan and requested and pricing and pricing_details:
            base = _plan_price_and_seats(pricing.chunk.text, plan)
            if base:
                monthly, included = base
                total_seats = int(requested.group(1).replace(",", ""))
                if plan == "launch" and total_seats > included:
                    return _draft("Launch cannot purchase extra seats, so the requested seat total is not available on that plan.", evidence, ("pricing:0", "pricing:1"))
                if plan == "team":
                    extra_rate = re.search(r"Extra Team seats cost\s+\$?([\d,]+) per seat per month", pricing_details.chunk.text, re.IGNORECASE)
                    if not extra_rate:
                        return None
                    extras = max(0, total_seats - included)
                    total = monthly + extras * _number(extra_rate.group(1))
                    return _draft(
                        f"Team includes {included} seats at ${monthly:,.0f} per month. {total_seats} total seats require {extras} extra seats at ${_number(extra_rate.group(1)):,.0f} each, for ${total:,.0f} per month.",
                        evidence,
                        ("pricing:0", "pricing:1"),
                    )

    # Select the highest applicable (non-cumulative) Enterprise service-credit tier.
    if sla and ("credit" in normalized or "sla" in normalized) and "uptime" in normalized:
        observed = re.search(r"\b(\d{1,3}(?:\.\d+)?)\s*%", question)
        plan = _plan(question)
        if plan in {"launch", "team"}:
            return _draft("Only Enterprise has a contractual uptime SLA; the corpus does not establish a Team or Launch service credit.", evidence, ("sla:0",))
        if observed:
            percent = _number(observed.group(1))
            tiers = [(int(credit), _number(threshold)) for credit, threshold in re.findall(r"(\d+)% below (\d+(?:\.\d+)?)%", sla.chunk.text, re.IGNORECASE)]
            applicable = [credit for credit, threshold in tiers if percent < threshold]
            if tiers:
                credit = max(applicable, default=0)
                if credit:
                    return _draft(f"At {percent:g}% uptime, the highest applicable Enterprise service credit is {credit}%; credit tiers are not cumulative.", evidence, ("sla:0",))
                return _draft(f"At {percent:g}% uptime, none of the published Enterprise service-credit thresholds is met.", evidence, ("sla:0",))

    return None
