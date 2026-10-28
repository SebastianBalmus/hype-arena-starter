"""Deterministic, source-preserving false-claim paraphrases; no new evidence."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent

# Exclude text/form disagreements, uncertain scope, aspiration presented as a
# current measurement, and narrative statements that do not assert the form.
EXCLUDED = {
    "p-126", "p-212", "p-229", "p-542", "p-579", "p-583", "p-808",
    "p-938", "p-999", "p-1049", "p-1078", "p-1104", "p-1219",
    "p-1235", "p-1491", "p-1493", "p-1494", "p-1734", "p-331", "p-1046",
    "p-800", "p-1343", "p-1408", "p-1590", "p-1498", "p-803",
    "p-1031", "p-1033", "p-1045", "p-1053", "p-1055", "p-1056",
    "p-1058", "p-992", "p-1097", "p-871", "p-1131",
    "p-1183", "p-1325", "p-1583", "p-1865", "p-1871", "p-807",
    "p-869", "p-1117", "p-1126", "p-1399", "p-1447", "p-266",
    "p-1080", "p-1091", "p-1612",
}

# Deliberate natural-language names: these distinguish related attributes, such
# as measured uptime / its SLA, models / devices, and annual / monthly prices.
NOUNS = {
    "active_devices": "active device count",
    "active_users": "monthly active user count",
    "ad_free_price": "price for the ad-free option",
    "announced": "announcement date",
    "annual_price": "annual subscription price",
    "api_version": "API version",
    "attendance": "board attendance rate",
    "attendees": "attendance total",
    "base_storage": "base storage capacity",
    "battery_capacity": "battery capacity",
    "battery_life": "battery life",
    "beta_date": "beta opening date",
    "business_listings": "business listing count",
    "call_length": "results call length",
    "caption_languages": "caption language count",
    "ceo": "CEO",
    "chair": "board chair",
    "city": "event city",
    "cold_start_time": "cold start time",
    "colours": "number of available colours",
    "concurrent_chats": "concurrent chat limit",
    "context_files": "context file capacity",
    "context_window": "context window",
    "cost_per_hire": "average cost per hire",
    "countries_covered": "country coverage",
    "countries_served": "number of countries served",
    "cto": "CTO",
    "customer_countries": "customer country count",
    "customers": "customer count",
    "daily_requests": "daily request allowance",
    "data_centres": "data centre footprint",
    "data_residency_regions": "data residency coverage",
    "date": "event date",
    "desks": "desk count",
    "dialects_supported": "supported dialect count",
    "duration": "outage duration",
    "emissions_cut_target": "emissions reduction target",
    "employees": "employee headcount",
    "facilities_team": "facilities team size",
    "flawless_seats": "Hooly seat count",
    "founded": "founding date",
    "free_storage": "free storage allowance",
    "free_trial": "free trial length",
    "functions": "function count",
    "general_availability_date": "general availability date",
    "growth_opportunities": "share of respondents who see growth opportunities",
    "guidance": "revenue guidance for the next quarter",
    "hallucination_rate": "benchmark hallucination rate",
    "head": "division head",
    "head_of_product": "head of product",
    "head_of_security": "head of security",
    "head_of_workspace": "head of Workspace",
    "headcount": "quarter-end employee headcount",
    "headquarters": "headquarters location",
    "hinge_rating": "hinge durability rating",
    "history_retention": "history retention period",
    "hosting_regions": "hosting region count",
    "images_indexed": "indexed image count",
    "included_seats": "included seat allowance",
    "index_size": "search index size",
    "inner_screen_size": "inner screen size",
    "integrations": "integration count",
    "interview_rounds": "interview round count",
    "joined": "Hooly joining date",
    "languages": "supported language count",
    "languages_served": "language coverage",
    "languages_supported": "supported language count",
    "launched": "launch date",
    "lead": "team lead",
    "lease_end": "lease end date",
    "livestream_viewers": "livestream audience limit",
    "mailbox_storage": "mailbox storage allowance",
    "mapped_distance": "mapped distance",
    "max_guests": "event guest limit",
    "max_output_length": "maximum output length",
    "max_participants": "call participant limit",
    "max_trade_in_credit": "maximum trade-in credit",
    "max_upload_size": "file upload size limit",
    "maximum_seats": "maximum seat count",
    "median_response_time": "median response time",
    "meetings_per_year": "annual meeting count",
    "memory": "memory capacity",
    "monthly_active_users": "monthly active user count",
    "news_refresh_interval": "news refresh interval",
    "offices": "office count",
    "offices_run": "number of offices managed",
    "offline_languages": "offline language count",
    "onboarding_time": "onboarding duration",
    "open_engineering_roles": "number of open engineering positions",
    "open_roles": "open role count",
    "opened": "office opening date",
    "operating_margin": "operating margin",
    "owner": "report owner",
    "p99_response_time": "p99 response time",
    "paying_customers": "paying customer count",
    "photos_stored": "stored photo count",
    "price": "price",
    "pro_price": "Pro subscription price",
    "programming_languages": "programming language count",
    "published": "publication date",
    "publisher_revenue_share": "publisher share of advertising revenue",
    "rated_accuracy": "rated translation accuracy",
    "recruiters": "recruiter headcount",
    "refund_window": "refund window",
    "registrations": "registration count",
    "release_cadence": "interval between releases",
    "released": "release date",
    "rent": "office rent",
    "repair_centres": "repair centre count",
    "revenue": "quarterly revenue",
    "saved_projects": "saved project allowance",
    "seats": "board seat count",
    "series_a_amount": "Series A funding amount",
    "series_a_co_investor": "Series A co-investor",
    "series_a_lead": "Series A lead investor",
    "services": "service count",
    "sessions": "programme session count",
    "share_of_ad_revenue": "share of Hooly's advertising revenue",
    "share_of_devices": "share of devices",
    "shipping_android": "preinstalled Android version",
    "simultaneous_editors": "simultaneous editor limit",
    "soc2_audit_start": "SOC 2 audit start date",
    "soc2_auditor": "SOC 2 auditor",
    "spam_catch_rate": "spam catch rate",
    "speakers": "speaker count",
    "sponsors": "sponsor count",
    "stable_version": "stable browser version",
    "staff": "staff headcount",
    "storage": "storage allowance",
    "storage_per_user": "storage allowance per user",
    "store_rating": "average store rating",
    "suggestion_acceptance_rate": "suggestion acceptance rate",
    "support_first_response": "support first response time",
    "supported_file_types": "supported file format count",
    "supported_ides": "supported IDE count",
    "supported_models": "supported model count",
    "tab_sleep_delay": "tab sleep delay",
    "team_size": "team size",
    "templates": "template gallery size",
    "ticket_price": "ticket price",
    "time_to_hire": "average time to hire",
    "units_sold": "sales total",
    "update_size": "update download size",
    "uptime_last_quarter": "uptime last quarter",
    "uptime_sla": "uptime promise",
    "version": "version",
    "voice_latency": "voice response latency",
    "waitlist": "waitlist size",
    "webcast_viewers": "results webcast audience",
    "webhook_events": "webhook event count",
    "weight": "weight",
}

BOOL = {
    "ai_answers": ("offers AI answers", "does not offer AI answers"),
    "ai_sidebar": ("includes an AI sidebar", "does not include an AI sidebar"),
    "anonymous": ("has anonymous responses", "does not have anonymous responses"),
    "ar_walking": ("offers augmented reality walking", "does not offer augmented reality walking"),
    "desktop_mode": ("includes desktop mode", "does not include desktop mode"),
    "discontinued": ("is discontinued", "is not discontinued"),
    "end_to_end_encryption": ("uses end-to-end encryption", "does not use end-to-end encryption"),
    "fully_remote": ("is fully remote", "is not fully remote"),
    "generally_available": ("is generally available", "is not generally available"),
    "had_major_outage": ("has had a major outage", "has never had a major outage"),
    "offline_maps": ("offers offline maps", "does not offer offline maps"),
    "on_device_gemini": ("includes on-device Gemini", "does not include on-device Gemini"),
    "open_source": ("is open source", "is not open source"),
    "passkey_sync": ("supports passkey sync", "does not support passkey sync"),
    "pro_tier": ("offers a Pro tier", "does not offer a Pro tier"),
    "raised_series_b": ("has raised its Series B", "has not raised its Series B"),
    "satellite_messaging": ("includes satellite messaging", "does not include satellite messaging"),
    "soc2_certified": ("is SOC 2 Type II certified", "is not SOC 2 Type II certified"),
    "trade_in_programme": ("offers a trade-in programme", "does not offer a trade-in programme"),
}

BOOL_ALT = {
    "ai_sidebar": ("has an AI sidebar", "has no AI sidebar"),
    "anonymous": ("collects responses anonymously", "collects responses without anonymity"),
    "ar_walking": ("supports augmented walking directions", "has no augmented walking directions"),
    "desktop_mode": ("has desktop mode", "has no desktop mode"),
    "discontinued": ("has been discontinued", "has not been discontinued"),
    "end_to_end_encryption": ("encrypts messages end to end", "does not encrypt messages end to end"),
    "fully_remote": ("operates fully remotely", "does not operate fully remotely"),
    "generally_available": ("has reached general availability", "has not reached general availability"),
    "had_major_outage": ("has experienced a major outage", "has not experienced a major outage"),
    "offline_maps": ("supports maps offline", "has no offline map support"),
    "on_device_gemini": ("has Gemini on device", "has no on-device Gemini"),
    "open_source": ("has open source code", "does not have open source code"),
    "passkey_sync": ("syncs passkeys", "does not sync passkeys"),
    "raised_series_b": ("has closed a Series B round", "has not closed a Series B round"),
    "satellite_messaging": ("supports messaging by satellite", "has no satellite messaging support"),
    "soc2_certified": ("holds SOC 2 Type II certification", "does not hold SOC 2 Type II certification"),
    "trade_in_programme": ("has a trade-in scheme", "has no trade-in programme"),
}

# Active sentence templates offer more syntactic variety for common claims.
ACTIVE = {
    "price": "{s} costs {v}",
    "annual_price": "The annual option for {s} costs {v}",
    "pro_price": "The Pro option in {s} costs {v}",
    "paying_customers": "{s} serves {v} on paid plans",
    "customers": "{s} serves {v}",
    "employees": "{s} employs {v}",
    "languages": "{s} supports {v}",
    "programming_languages": "{s} works with {v} for programming",
    "languages_supported": "{s} translates between {v}",
    "monthly_active_users": "{s} has {v} active each month",
    "active_users": "{s} has {v} active each month",
    "active_devices": "{s} runs on {v}",
    "included_seats": "{s} includes {v}",
    "max_upload_size": "{s} permits file uploads of {v}",
    "supported_file_types": "{s} opens {v}",
    "context_window": "{s} has a context window of {v}",
    "median_response_time": "{s} replies in a median of {v}",
    "p99_response_time": "{s} has a p99 response time of {v}",
    "hallucination_rate": "{s} has a benchmark hallucination rate of {v}",
    "suggestion_acceptance_rate": "Users of {s} accept {v} of its suggestions",
    "store_rating": "{s} has an average store rating of {v}",
    "templates": "The template gallery in {s} contains {v}",
    "supported_models": "{s} supports {v}",
    "supported_ides": "{s} runs in {v}",
    "photos_stored": "{s} stores {v}",
    "index_size": "{s} indexes {v}",
    "update_size": "The update download for {s} is {v}",
    "weight": "{s} weighs {v}",
    "free_trial": "The free trial for {s} lasts {v}",
    "history_retention": "{s} keeps history for {v}",
    "free_storage": "{s} includes {v} of free storage",
    "battery_capacity": "{s} contains a battery rated at {v}",
    "battery_life": "The battery in {s} lasts {v}",
    "country_coverage": "{s} covers {v}",
    "daily_requests": "{s} allows {v} per day",
    "max_participants": "A call in {s} holds {v}",
    "caption_languages": "{s} offers captions in {v}",
    "webhook_events": "{s} emits {v} through webhooks",
    "units_sold": "{s} has sold {v}",
    "services": "{s} provides {v}",
    "recruiters": "{s} has {v} handling recruitment",
    "open_roles": "{s} has {v} open",
    "uptime_sla": "{s} promises {v} uptime",
    "uptime_last_quarter": "{s} achieved {v} uptime last quarter",
    "onboarding_time": "Onboarding with {s} takes {v}",
    "refund_window": "{s} allows refunds for {v}",
    "context_files": "{s} reads {v} in its context",
    "offices": "{s} operates {v}",
}

WORDS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
MONTHS = "January February March April May June July August September October November December".split()
PHYSICAL = {"B", "KB", "MB", "GB", "TB", "mm", "cm", "m", "km", "in", "inch", "g", "kg", "mAh", "Hz", "kHz", "MHz", "GHz", "W", "mW", "kW"}


def num(value, style):
    if isinstance(value, int):
        if style == 3 and 0 <= value <= 20:
            return WORDS[value]
        return f"{value:,}"
    return str(value)


def value_text(value, style):
    kind = value["kind"]
    v = value["value"]
    if kind == "name":
        text = v
    elif kind == "date":
        bits = v.split("-")
        if style % 2 and len(bits) == 3:
            text = f"{int(bits[2])} {MONTHS[int(bits[1])-1]} {bits[0]}"
        elif style % 2 and len(bits) == 2:
            text = f"{MONTHS[int(bits[1])-1]} {bits[0]}"
        else:
            text = v
    elif kind == "version":
        text = ("version " if style % 2 else "v") + v
    elif kind == "money":
        symbol = {"EUR": "€", "USD": "$", "GBP": "£"}.get(value["currency"])
        number = num(v, 0)
        text = f"{symbol}{number}" if symbol and style % 2 == 0 else f"{number} {value['currency']}"
        if value.get("per"):
            text += (" per " if style % 2 == 0 else " a ") + value["per"]
    elif kind == "percent":
        text = num(v, style) + (" percent" if style % 2 else "%")
    elif kind in ("quantity", "duration"):
        unit = value["unit"]
        if kind == "duration":
            unit = {"ms": "millisecond", "s": "second", "min": "minute", "h": "hour"}.get(unit, unit)
        if unit not in PHYSICAL:
            if v != 1:
                unit = {"company": "companies", "country": "countries", "watch": "watches", "person": "people"}.get(unit, unit + "s")
            if unit.lower() in ("ide", "ides"):
                unit = unit.upper().replace("IDES", "IDEs")
        text = f"{num(v, style)} {unit}"
    else:
        raise ValueError(kind)
    op = value.get("op", "eq")
    prefixes = {"eq": "", "gt": "more than " if style % 2 else "over ", "lt": "less than " if style % 2 else "under ", "ge": "at least ", "le": "at most "}
    return prefixes[op] + text


def clause(fact, style, source):
    subject, attr, val = fact["subject"], fact["attribute"], fact["value"]
    if val["kind"] == "boolean":
        if source["id"] == "p-1009":
            return [
                f"{subject} encrypts every message end to end",
                f"Every message in {subject} has end-to-end encryption",
                f"All messages in {subject} are encrypted end to end",
                f"End-to-end encryption covers every {subject} message",
                f"Each {subject} message is protected by end-to-end encryption",
            ][style]
        pair = BOOL_ALT.get(attr, BOOL[attr]) if style % 2 else BOOL[attr]
        phrase = pair[0 if val["value"] else 1]
        if style < 2:
            return f"{subject} {phrase}"
        if style == 2:
            return f"{subject} {phrase}".replace("does not", "doesn't").replace("is not", "isn't").replace("has not", "hasn't")
        return f"{subject} {phrase}"
    noun = NOUNS[attr]
    monthly_source = "month" in source["text"].casefold()
    if attr == "active_users" and not monthly_source:
        noun = "user count"
    if attr == "monthly_active_users" and not monthly_source:
        noun = "active user count"
    if attr == "share_of_ad_revenue" and source["id"] in {"p-331", "p-1046"}:
        noun = "advertising share of total revenue"
    if attr == "attendees" and source["id"] == "p-1066":
        noun = "expected attendance total"
    if attr == "sponsors" and source["id"] in {"p-232", "p-1106", "p-1066"}:
        noun = "confirmed sponsor count"
    if attr == "supported_ides" and source["id"] == "p-1092":
        noun = "supported IDE count from the beta opening"
    if attr == "date" and val["value"] > source["date"]:
        noun = "scheduled event date"
    if attr == "languages" and subject == "Hooly Jules":
        noun = "programming language count"
    v = value_text(val, style)
    # A results publication reports a customer count; it does not itself serve customers.
    if style == 0 and attr == "customers" and "Results" in subject:
        return f"{subject} reports {v}"
    if style == 0 and source["id"] == "p-1092" and attr == "supported_ides":
        return f"{subject} works in {v} from the first day of its beta"
    if style == 0 and attr in {"active_users", "monthly_active_users"} and not monthly_source:
        return f"{subject} has {v}"
    if style == 0 and attr in ACTIVE:
        return ACTIVE[attr].format(s=subject, v=v)
    if style == 0:
        return f"{subject}'s {noun} is {v}"
    if style == 1:
        return f"The {noun} of {subject} is {v}"
    if style == 2:
        return f"{subject}'s {noun} is {v}"
    if style == 3:
        return f"For {subject}, the {noun} is {v}"
    if val["kind"] in {"date", "version"}:
        return f"The {noun} for {subject} is {v}"
    if val["kind"] == "name":
        return f"{v} is {subject}'s {noun}"
    return f"The {noun} for {subject} stands at {v}"


def normalize(text):
    return re.sub(r"[\W_]+", " ", text.casefold()).strip()


def main():
    rows = list(csv.DictReader((ROOT / "data/practice.csv").open(newline="", encoding="utf-8")))
    lexicon = json.loads((ROOT / "data/lexicon.json").read_text())
    lex = {entry["name"]: set(entry["attributes"]) for entry in lexicon["subjects"]}
    seen = {normalize(row["text"]) for row in rows}
    # Parent merges true first. Skip its rows too, even when a value has changed
    # between dates and the otherwise identical sentence has a different label.
    true_candidates = HERE.parent / "true" / "candidates.jsonl"
    if true_candidates.exists():
        seen.update(normalize(json.loads(line)["text"]) for line in true_candidates.read_text().splitlines() if line)
    sources = [r for r in rows if r["label"] == "false" and r["id"] not in EXCLUDED]
    out = []
    for style in range(5):
        for row in sources:
            forms = json.loads(row["form"])
            assert all(f["subject"] in lex and f["attribute"] in lex[f["subject"]] for f in forms)
            clauses = [clause(fact, style, row) for fact in forms]
            # All annotated conjuncts stay together; no atom inherits false alone.
            text = "; ".join(clauses) + "."
            text = text[0].upper() + text[1:]
            norm = normalize(text)
            if len(text) > 300 or norm in seen:
                continue
            seen.add(norm)
            out.append({"source_id": row["id"], "text": text, "date": row["date"], "label": "false", "form": forms,
                        "transformation": f"semantic_clause_paraphrase_{style+1}"})
            if len(out) == 850:
                break
        if len(out) == 850:
            break
    assert len(out) == 850, len(out)
    source_by_id = {r["id"]: r for r in rows}
    for item in out:
        src = source_by_id[item["source_id"]]
        assert item["form"] == json.loads(src["form"])
        assert item["date"] == src["date"] and item["label"] == src["label"]
    (HERE / "candidates.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print(json.dumps({"count": len(out), "unique_sources": len({r['source_id'] for r in out}), "attributes": len({f['attribute'] for r in out for f in r['form']}), "excluded_sources": sorted(EXCLUDED)}))


if __name__ == "__main__":
    main()
