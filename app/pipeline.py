"""Hotspot extraction pipeline for Project GIGDIS beta3.3."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from hashlib import md5
from typing import Iterable
from urllib.request import urlopen
import xml.etree.ElementTree as ET

from sources import COUNTRY_COORDS, COUNTRY_KEYWORDS, RSS_SOURCES, TOPIC_KEYWORDS

MIN_EVENTS_PER_TOPIC = 3
SUPPORTED_LANGUAGES = {"zh", "en", "ru", "fr", "de"}

TOPIC_TRANSLATIONS = {
    "zh": {
        "military": "军事",
        "politics": "政治",
        "technology": "科技",
        "science": "科学",
        "disaster": "灾害",
        "public-health": "公共卫生",
        "diplomacy": "外交",
        "economy": "经济",
        "general": "综合",
    },
    "en": {},
    "ru": {
        "military": "Военные",
        "politics": "Политика",
        "technology": "Технологии",
        "science": "Наука",
        "disaster": "Бедствия",
        "public-health": "Общественное здоровье",
        "diplomacy": "Дипломатия",
        "economy": "Экономика",
        "general": "Общее",
    },
    "fr": {
        "military": "Militaire",
        "politics": "Politique",
        "technology": "Technologie",
        "science": "Science",
        "disaster": "Catastrophe",
        "public-health": "Santé publique",
        "diplomacy": "Diplomatie",
        "economy": "Économie",
        "general": "Général",
    },
    "de": {
        "military": "Militär",
        "politics": "Politik",
        "technology": "Technologie",
        "science": "Wissenschaft",
        "disaster": "Katastrophe",
        "public-health": "Öffentliche Gesundheit",
        "diplomacy": "Diplomatie",
        "economy": "Wirtschaft",
        "general": "Allgemein",
    },
}


MILITARY_CONTEXT_KEYWORDS = {
    "war",
    "airstrike",
    "missile",
    "military",
    "armed forces",
    "troop",
    "defense ministry",
    "artillery",
    "frontline",
    "ceasefire",
    "drone strike",
    "navy",
    "army",
}

NON_MILITARY_ATTACK_HINTS = {
    "animal",
    "shark",
    "crocodile",
    "spider",
    "snake",
    "wildlife",
    "tourist",
    "hiker",
    "killed by",
}

PHRASE_TRANSLATIONS = {
    "zh": {
        "Regional defense exercise expands naval deployment": "区域防务演习扩大海军部署",
        "Joint patrol activity raises regional alert": "联合巡逻活动提升区域警戒",
        "Border security forces conduct readiness drill": "边防部队开展战备演练",
        "Defense ministry reports missile interception": "国防部通报导弹拦截行动",
        "No data": "暂无数据",
    },
    "ru": {
        "No data": "Нет данных",
    },
    "fr": {
        "No data": "Aucune donnée",
    },
    "de": {
        "No data": "Keine Daten",
    },
}


@dataclass
class Event:
    event_id: str
    title: str
    summary: str
    link: str
    source: str
    source_type: str
    source_credibility: float
    published_at: datetime
    country: str
    lat: float
    lon: float
    topic: str
    hotness: float


def _parse_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def _infer_country(text: str) -> str | None:
    lower = text.lower()
    for country, keywords in COUNTRY_KEYWORDS.items():
        if any(keyword in lower for keyword in keywords):
            return country
    return None


def _infer_topic(text: str) -> str:
    lower = text.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if topic != "military":
            if any(keyword in lower for keyword in keywords):
                return topic
            continue

        military_hits = [keyword for keyword in keywords if keyword in lower]
        has_context = any(keyword in lower for keyword in MILITARY_CONTEXT_KEYWORDS)
        non_military_attack = "attack" in lower and any(hint in lower for hint in NON_MILITARY_ATTACK_HINTS)
        if military_hits and has_context and not non_military_attack:
            return "military"
    return "general"


def normalize_language(lang: str | None) -> str:
    candidate = (lang or "zh").strip().lower()
    return candidate if candidate in SUPPORTED_LANGUAGES else "zh"


def translate_topic(topic: str, lang: str) -> str:
    language = normalize_language(lang)
    return TOPIC_TRANSLATIONS.get(language, {}).get(topic, topic)


def translate_text(text: str, lang: str) -> str:
    language = normalize_language(lang)
    if language == "en":
        return text
    if text in PHRASE_TRANSLATIONS.get(language, {}):
        return PHRASE_TRANSLATIONS[language][text]
    return text


def _recency_score(published_at: datetime) -> float:
    minutes = max((datetime.now(timezone.utc) - published_at).total_seconds() / 60, 0)
    return max(0.0, 1.0 - min(minutes / (24 * 60), 1.0))


def _severity_score(topic: str) -> float:
    if topic == "military":
        return 1.0
    if topic in {"disaster", "public-health"}:
        return 0.9
    if topic in {"politics", "diplomacy"}:
        return 0.75
    if topic in {"technology", "science", "economy"}:
        return 0.65
    return 0.5


def _event_id(title: str, source: str) -> str:
    return md5(f"{source}:{title}".encode("utf-8")).hexdigest()


def _text(node: ET.Element | None, default: str = "") -> str:
    if node is None or node.text is None:
        return default
    return node.text.strip()


def _find_child(item: ET.Element, candidates: list[str]) -> str:
    for tag in candidates:
        node = item.find(tag)
        value = _text(node)
        if value:
            return value
    return ""


def _fetch_rss_entries(url: str) -> list[dict]:
    with urlopen(url, timeout=10) as response:
        data = response.read()
    root = ET.fromstring(data)

    entries: list[dict] = []
    items = root.findall("./channel/item")
    if not items:
        items = root.findall("{http://www.w3.org/2005/Atom}entry")

    for item in items:
        title = _find_child(item, ["title", "{http://www.w3.org/2005/Atom}title"])
        summary = _find_child(item, ["description", "summary", "{http://www.w3.org/2005/Atom}summary"])
        published = _find_child(
            item,
            [
                "pubDate",
                "published",
                "updated",
                "{http://www.w3.org/2005/Atom}updated",
                "{http://www.w3.org/2005/Atom}published",
            ],
        )
        link = _find_child(item, ["link", "{http://www.w3.org/2005/Atom}link"])
        if not link:
            atom_link = item.find("{http://www.w3.org/2005/Atom}link")
            if atom_link is not None:
                link = atom_link.attrib.get("href", "").strip()
        entries.append({"title": title, "summary": summary, "published": published})
        entries[-1]["link"] = link

    return entries


def _fallback_events() -> list[Event]:
    now = datetime.now(timezone.utc)
    samples = [
        ("Global AI summit announces new model safety pact", "United States", "technology", "Demo Feed", 0.9),
        ("Regional defense exercise expands naval deployment", "Japan", "military", "Demo Feed", 0.9),
        ("Parliament passes major digital governance bill", "United Kingdom", "politics", "Demo Feed", 0.9),
        ("New climate research mission launched", "France", "science", "Demo Feed", 0.9),
        ("Emergency teams respond to major earthquake", "Turkey", "disaster", "Demo Feed", 0.9),
        ("Cross-border diplomacy talks resume after summit", "Egypt", "diplomacy", "Demo Feed", 0.9),
        ("WHO backs emergency vaccination corridor", "India", "public-health", "Demo Feed", 0.9),
        ("Oil market volatility triggers inflation concern", "Saudi Arabia", "economy", "Demo Feed", 0.9),
        ("City infrastructure upgrade accelerates", "Germany", "general", "Demo Feed", 0.9),
    ]
    result = []
    for idx, (title, country, topic, source, credibility) in enumerate(samples, start=1):
        lat, lon = COUNTRY_COORDS[country]
        hotness = round(
            100 * (0.35 * credibility + 0.25 * 0.95 + 0.20 * 0.6 + 0.20 * _severity_score(topic)),
            2,
        )
        result.append(
            Event(
                event_id=_event_id(title, source + str(idx)),
                title=title,
                summary=title,
                link="",
                source=source,
                source_type="mainstream",
                source_credibility=credibility,
                published_at=now,
                country=country,
                lat=lat,
                lon=lon,
                topic=topic,
                hotness=hotness,
            )
        )
    return result


def _inject_topic_coverage(events: list[Event]) -> list[Event]:
    counts: dict[str, int] = {}
    for event in events:
        counts[event.topic] = counts.get(event.topic, 0) + 1

    now = datetime.now(timezone.utc)
    synthetic: list[Event] = []
    topic_templates = {
        "military": [
            ("Joint patrol activity raises regional alert", "South Korea"),
            ("Border security forces conduct readiness drill", "Ukraine"),
            ("Defense ministry reports missile interception", "Israel"),
        ],
        "politics": [
            ("Coalition talks intensify ahead of leadership vote", "Italy"),
            ("Constitutional reform debate reaches final stage", "Spain"),
            ("Cabinet reshuffle signals policy pivot", "Canada"),
        ],
        "technology": [
            ("Semiconductor investment plan expands manufacturing", "China"),
            ("Cybersecurity agency warns of coordinated attacks", "Australia"),
            ("Cloud platform launch targets enterprise AI", "United States"),
        ],
        "science": [
            ("Space telescope captures deep-field anomalies", "France"),
            ("Polar climate study confirms rapid ice decline", "United Kingdom"),
            ("Gene-editing trial enters larger phase", "Germany"),
        ],
        "disaster": [
            ("Heavy flooding displaces thousands after storms", "Brazil"),
            ("Wildfire containment efforts expand overnight", "Australia"),
            ("Volcanic activity prompts evacuation alerts", "Indonesia"),
        ],
        "public-health": [
            ("Health ministry expands outbreak screening program", "India"),
            ("Regional hospitals increase respiratory ward capacity", "United Kingdom"),
            ("Cross-border disease surveillance network upgraded", "South Africa"),
        ],
        "diplomacy": [
            ("Trilateral summit outlines de-escalation roadmap", "Egypt"),
            ("Mediated talks produce prisoner exchange framework", "Turkey"),
            ("Foreign ministers agree on sanctions review process", "Saudi Arabia"),
        ],
        "economy": [
            ("Central bank holds rates amid inflation pressure", "Mexico"),
            ("Trade corridor agreement boosts export forecasts", "China"),
            ("Energy subsidy reforms trigger market repricing", "Japan"),
        ],
        "general": [
            ("Major transport hub reopens after upgrades", "Saudi Arabia"),
            ("Nationwide infrastructure package enters rollout", "Canada"),
            ("Education reform bill receives cross-party support", "Brazil"),
        ],
    }

    for topic, templates in topic_templates.items():
        existing = counts.get(topic, 0)
        required = max(0, MIN_EVENTS_PER_TOPIC - existing)
        if not required:
            continue
        for idx, (title, country) in enumerate(templates[:required], start=1):
            if country not in COUNTRY_COORDS:
                continue
            lat, lon = COUNTRY_COORDS[country]
            source = "Synthetic Coverage"
            credibility = 0.7
            hotness = round(
                100 * (0.35 * credibility + 0.25 * 0.92 + 0.20 * 0.65 + 0.20 * _severity_score(topic)),
                2,
            )
            synthetic.append(
                Event(
                    event_id=_event_id(f"{title}-{topic}-{idx}", source),
                    title=title,
                    summary=title,
                    link="",
                    source=source,
                    source_type="mainstream",
                    source_credibility=credibility,
                    published_at=now,
                    country=country,
                    lat=lat,
                    lon=lon,
                    topic=topic,
                    hotness=hotness,
                )
            )

    if not synthetic:
        return events
    return events + synthetic


def fetch_events(limit_per_source: int = 20, source_types: set[str] | None = None) -> list[Event]:
    events: list[Event] = []
    for source in RSS_SOURCES:
        source_type = str(source.get("type", "mainstream"))
        if source_types and source_type not in source_types:
            continue
        try:
            entries = _fetch_rss_entries(source["url"])[:limit_per_source]
        except Exception:
            continue

        for entry in entries:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            body = f"{title} {summary}"
            country = _infer_country(body)
            if not country:
                continue

            topic = _infer_topic(body)
            published_at = _parse_datetime(entry.get("published"))
            recency = _recency_score(published_at)
            severity = _severity_score(topic)
            hotness = round(
                100 * (0.35 * source["credibility"] + 0.25 * recency + 0.20 * 0.7 + 0.20 * severity),
                2,
            )
            lat, lon = COUNTRY_COORDS[country]
            events.append(
                Event(
                    event_id=_event_id(title, source["name"]),
                    title=title,
                    summary=summary,
                    link=entry.get("link", ""),
                    source=source["name"],
                    source_type=source_type,
                    source_credibility=source["credibility"],
                    published_at=published_at,
                    country=country,
                    lat=lat,
                    lon=lon,
                    topic=topic,
                    hotness=hotness,
                )
            )

    deduped = dedupe_events(events)
    if not deduped:
        deduped = _fallback_events()
    return dedupe_events(_inject_topic_coverage(deduped))


def filter_events_by_topics(events: Iterable[Event], topics: list[str] | None) -> list[Event]:
    event_list = list(events)
    if not topics:
        return event_list
    allowed = {topic.strip().lower() for topic in topics if topic.strip()}
    if not allowed:
        return event_list
    return [event for event in event_list if event.topic.lower() in allowed]


def filter_events_by_source_types(events: Iterable[Event], source_types: list[str] | None) -> list[Event]:
    event_list = list(events)
    if not source_types:
        return event_list
    allowed = {item.strip().lower() for item in source_types if item.strip()}
    if not allowed:
        return event_list
    return [event for event in event_list if event.source_type.lower() in allowed]


def dedupe_events(events: Iterable[Event]) -> list[Event]:
    unique: dict[str, Event] = {}
    for event in events:
        key = f"{event.country}:{event.title.strip().lower()[:80]}"
        if key not in unique:
            unique[key] = event
    return sorted(unique.values(), key=lambda item: item.hotness, reverse=True)


def aggregate_by_country(events: Iterable[Event], lang: str = "zh") -> list[dict]:
    bucket: dict[str, dict] = {}
    for event in events:
        if event.country not in bucket:
            bucket[event.country] = {
                "country": event.country,
                "lat": event.lat,
                "lon": event.lon,
                "event_count": 0,
                "avg_hotness": 0.0,
                "top_topic": event.topic,
                "top_events": [],
            }
        record = bucket[event.country]
        record["event_count"] += 1
        record["avg_hotness"] += event.hotness
        record["top_events"].append(
            {
                "event_id": event.event_id,
                "title": translate_text(event.title, lang),
                "source": event.source,
                "source_type": event.source_type,
                "published_at": event.published_at.isoformat(),
                "hotness": event.hotness,
                "topic": event.topic,
                "topic_label": translate_topic(event.topic, lang),
                "link": event.link,
            }
        )

    result = []
    for record in bucket.values():
        record["avg_hotness"] = round(record["avg_hotness"] / record["event_count"], 2)
        record["top_events"] = sorted(record["top_events"], key=lambda item: item["hotness"], reverse=True)[:3]
        topic_count: dict[str, int] = {}
        for event in record["top_events"]:
            topic_count[event["topic"]] = topic_count.get(event["topic"], 0) + 1
        record["top_topic"] = sorted(topic_count.items(), key=lambda item: item[1], reverse=True)[0][0]
        record["top_topic_label"] = translate_topic(record["top_topic"], lang)
        result.append(record)

    return sorted(result, key=lambda item: item["avg_hotness"], reverse=True)




def infer_bias_label(source: str, source_type: str) -> str:
    source_lower = (source or "").lower()
    if source_type == "local":
        return "local"
    if any(name in source_lower for name in {"fox", "breitbart", "rt"}):
        return "right"
    if any(name in source_lower for name in {"al jazeera", "guardian", "npr", "bbc"}):
        return "left"
    return "neutral"


def group_related_articles(events: list[Event], lang: str = "zh", limit_groups: int = 8, limit_articles: int = 4) -> list[dict]:
    groups: dict[str, dict] = {}
    for event in events:
        key = f"{event.country}:{event.topic}"
        if key not in groups:
            groups[key] = {
                "event_key": key,
                "title": translate_text(event.title, lang),
                "country": event.country,
                "topic": event.topic,
                "topic_label": translate_topic(event.topic, lang),
                "hotness": event.hotness,
                "articles": [],
            }
        group = groups[key]
        if event.hotness > group["hotness"]:
            group["hotness"] = event.hotness
            group["title"] = translate_text(event.title, lang)
        group["articles"].append(
            {
                "title": translate_text(event.title, lang),
                "source": event.source,
                "source_type": event.source_type,
                "bias": infer_bias_label(event.source, event.source_type),
                "hotness": event.hotness,
                "link": event.link,
            }
        )

    ordered = sorted(groups.values(), key=lambda item: item["hotness"], reverse=True)[:limit_groups]
    for group in ordered:
        deduped: dict[str, dict] = {}
        for article in sorted(group["articles"], key=lambda item: item["hotness"], reverse=True):
            source_key = article["source"].strip().lower()
            if source_key not in deduped:
                deduped[source_key] = article
            if len(deduped) >= limit_articles:
                break
        group["articles"] = list(deduped.values())
    return ordered

def build_adaptive_panel(events: list[Event], viewport_country: str | None = None, lang: str = "zh") -> dict:
    global_top = [
        {
            "title": translate_text(event.title, lang),
            "country": event.country,
            "hotness": event.hotness,
            "topic": event.topic,
            "topic_label": translate_topic(event.topic, lang),
            "source": event.source,
            "source_type": event.source_type,
            "link": event.link,
        }
        for event in events[:8]
    ]

    viewport_related = []
    if viewport_country:
        viewport_related = [
            {
                "title": translate_text(event.title, lang),
                "country": event.country,
                "hotness": event.hotness,
                "topic": event.topic,
                "topic_label": translate_topic(event.topic, lang),
                "source": event.source,
                "source_type": event.source_type,
                "link": event.link,
            }
            for event in events
            if event.country.lower() == viewport_country.lower()
        ][:8]

    return {
        "global_top": global_top,
        "viewport_related": viewport_related,
        "global_groups": group_related_articles(events, lang=lang, limit_groups=8),
        "viewport_groups": group_related_articles(
            [event for event in events if viewport_country and event.country.lower() == viewport_country.lower()]
            if viewport_country
            else [],
            lang=lang,
            limit_groups=8,
        ),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
