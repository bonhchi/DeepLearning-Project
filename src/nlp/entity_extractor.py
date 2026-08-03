"""Rule-based entity and price-constraint extraction for product queries.

The baseline deliberately uses only the Python standard library.  It is meant to
be deterministic, cheap to run, and easy to extend while a learned NER model is
not yet available.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Final, TypedDict


LOGGER = logging.getLogger(__name__)


class EntityMatch(TypedDict, total=False):
    """One normalized entity together with extraction evidence."""

    value: str | int | float
    confidence: float
    matched_text: str
    source: str
    negated: bool
    currency: str


class ExtractedEntities(TypedDict):
    """Stable output schema returned by :meth:`EntityExtractor.extract`."""

    category: EntityMatch | None
    brand: EntityMatch | None
    color: EntityMatch | None
    size: EntityMatch | None
    material: EntityMatch | None
    feature: list[EntityMatch]
    purpose: list[EntityMatch]
    min_price: EntityMatch | None
    max_price: EntityMatch | None


ENTITY_FIELDS: Final[tuple[str, ...]] = (
    "category",
    "brand",
    "color",
    "size",
    "material",
    "feature",
    "purpose",
    "min_price",
    "max_price",
)


CATEGORY_ALIASES: Final[dict[str, set[str]]] = {
    "Appliances": {
        "appliance",
        "appliances",
        "do dien gia dung",
        "may giat",
        "may rua bat",
        "refrigerator",
        "fridge",
        "tu lanh",
    },
    "Automotive": {
        "automotive",
        "car accessories",
        "do choi xe hoi",
        "o to",
        "phu tung xe",
        "vehicle",
        "xe hoi",
    },
    "Electronics": {
        "camera",
        "computer",
        "dien thoai",
        "dien tu",
        "earbuds",
        "electronics",
        "headphone",
        "headphones",
        "laptop",
        "loa",
        "may tinh",
        "phone",
        "smartphone",
        "tai nghe",
    },
    "Health_and_Household": {
        "cham soc suc khoe",
        "do gia dung",
        "health",
        "healthcare",
        "household",
        "supplement",
        "thuc pham chuc nang",
        "vitamin",
    },
    "Beauty_and_Personal_Care": {
        "beauty",
        "cham soc ca nhan",
        "cham soc da",
        "cosmetics",
        "lam dep",
        "makeup",
        "my pham",
        "personal care",
        "skincare",
    },
    "Fashion": {"fashion", "thoi trang"},
    "bags": {"bag", "bags", "purse", "tui", "tui xach"},
    "bottoms": {"jeans", "pants", "quan", "trousers"},
    "dresses": {"dam", "dress", "dresses", "vay"},
    "eyewear": {"eyewear", "glasses", "kinh", "sunglasses"},
    "jewelry": {"jewelry", "nhan", "trang suc", "vong co"},
    "outerwear": {"ao khoac", "coat", "jacket", "outerwear"},
    "shoes": {"boot", "boots", "giay", "shoe", "shoes", "sneaker", "sneakers"},
    "socks": {"sock", "socks", "tat", "vo"},
    "tops": {"ao", "ao phong", "ao thun", "blouse", "shirt", "t shirt", "tops"},
    "watches": {"dong ho", "smartwatch", "watch", "watches"},
}

BRAND_ALIASES: Final[dict[str, set[str]]] = {
    "Adidas": {"adidas"},
    "Anker": {"anker", "soundcore"},
    "Apple": {"apple"},
    "Asus": {"asus"},
    "Bose": {"bose"},
    "Canon": {"canon"},
    "Dell": {"dell"},
    "Google": {"google", "pixel"},
    "HP": {"hewlett packard", "hp"},
    "JBL": {"jbl"},
    "Lenovo": {"lenovo"},
    "LG": {"lg"},
    "Logitech": {"logitech"},
    "Microsoft": {"microsoft", "surface"},
    "New Balance": {"new balance"},
    "Nike": {"nike"},
    "Nikon": {"nikon"},
    "OnePlus": {"oneplus", "one plus"},
    "Oppo": {"oppo"},
    "Panasonic": {"panasonic"},
    "Puma": {"puma"},
    "Samsung": {"samsung"},
    "Sony": {"sony"},
    "Under Armour": {"under armour"},
    "Vivo": {"vivo"},
    "Xiaomi": {"mi", "redmi", "xiaomi"},
}

COLOR_ALIASES: Final[dict[str, set[str]]] = {
    "beige": {"beige", "mau be"},
    "black": {"black", "mau den", "màu đen", "đen"},
    "blue": {"blue", "mau xanh duong", "navy", "xanh duong"},
    "brown": {"brown", "nau"},
    "gold": {"gold", "mau vang kim", "vang kim"},
    "gray": {"gray", "grey", "mau ghi", "xam"},
    "green": {"green", "mau xanh la", "xanh la"},
    "orange": {"cam", "orange"},
    "pink": {"hong", "pink"},
    "purple": {"purple", "tim"},
    "red": {"mau do", "màu đỏ", "red", "đỏ"},
    "silver": {"bac", "silver"},
    "white": {"mau trang", "trang", "white"},
    "yellow": {"mau vang", "vang", "yellow"},
}

SIZE_ALIASES: Final[dict[str, set[str]]] = {
    "XS": {"extra small", "x small"},
    "S": {"small"},
    "M": {"medium"},
    "L": {"large"},
    "XL": {"extra large", "x large"},
    "XXL": {"2xl", "double xl", "xxl"},
    "XXXL": {"3xl", "triple xl", "xxxl"},
}

MATERIAL_ALIASES: Final[dict[str, set[str]]] = {
    "aluminum": {"aluminium", "aluminum", "nhom"},
    "canvas": {"canvas", "vai bo"},
    "ceramic": {"ceramic", "gom", "su"},
    "cotton": {"cotton", "vai cotton"},
    "denim": {"denim", "vai jean"},
    "glass": {"glass", "kinh cuong luc"},
    "leather": {"da", "da that", "genuine leather", "leather"},
    "linen": {"linen", "vai lanh"},
    "metal": {"kim loai", "metal"},
    "plastic": {"nhua", "plastic"},
    "polyester": {"polyester"},
    "silicone": {"silicone"},
    "stainless_steel": {"inox", "stainless steel", "thep khong gi"},
    "wool": {"len", "wool"},
}

FEATURE_ALIASES: Final[dict[str, set[str]]] = {
    "anti_slip": {"anti slip", "chong truot", "non slip"},
    "bluetooth": {"bluetooth"},
    "breathable": {"breathable", "thoang khi"},
    "durable": {"ben bi", "bền", "durable"},
    "fast_charging": {"fast charging", "sac nhanh"},
    "foldable": {"co the gap", "foldable", "gap gon"},
    "lightweight": {"lightweight", "nhẹ", "sieu nhe"},
    "long_battery_life": {
        "long battery",
        "long battery life",
        "pin lau",
        "pin trau",
        "thoi luong pin dai",
    },
    "noise_cancelling": {
        "active noise cancellation",
        "anc",
        "chong on",
        "khu tieng on",
        "noise canceling",
        "noise cancelling",
    },
    "organic": {"huu co", "organic"},
    "touchscreen": {"cam ung", "touch screen", "touchscreen"},
    "waterproof": {"chong nuoc", "khong tham nuoc", "water resistant", "waterproof"},
    "wired": {"co day", "wired"},
    "wireless": {"khong day", "wireless"},
}

PURPOSE_ALIASES: Final[dict[str, set[str]]] = {
    "commuting": {"commute", "di lam hang ngay", "di tau xe"},
    "gaming": {"choi game", "gaming"},
    "gift": {"gift", "lam qua", "qua tang", "tang ban"},
    "office": {"cong so", "office", "van phong"},
    "outdoor": {"da ngoai", "ngoai troi", "outdoor"},
    "running": {"chay bo", "jogging", "running"},
    "skincare": {"cham soc da", "duong da", "skincare"},
    "sports": {"choi the thao", "sports", "the thao"},
    "study": {"hoc tap", "study"},
    "travel": {"di du lich", "du lich", "travel", "travelling", "traveling"},
    "work": {"lam viec", "work", "working"},
    "workout": {"gym", "tap gym", "workout"},
}


_VIETNAMESE_MARKERS: Final[set[str]] = {
    "ao",
    "can",
    "cho",
    "den",
    "duoi",
    "gia",
    "khong",
    "mau",
    "mot",
    "muon",
    "toi",
    "trieu",
}
# Accent removal is useful for users who type Vietnamese without a keyboard, but
# it can merge unrelated words ("tìm" -> "tim", "đã" -> "da").  When the input
# itself contains accents, these ambiguous aliases are checked against the valid
# surface forms below.
_ACCENT_SENSITIVE_SURFACES: Final[dict[str, set[str]]] = {
    "ao": {"áo"},
    "bac": {"bạc"},
    "be": {"be"},
    "ben": {"bền"},
    "da": {"da"},
    "den": {"đen"},
    "do": {"đỏ"},
    "giay": {"giày"},
    "hong": {"hồng"},
    "len": {"len"},
    "nau": {"nâu"},
    "nhe": {"nhẹ"},
    "nhom": {"nhôm"},
    "su": {"sứ"},
    "tim": {"tím"},
    "tui": {"túi"},
    "vang": {"vàng"},
    "vay": {"váy"},
    "vo": {"vớ"},
    "xam": {"xám"},
}
_NEGATION_PREFIX = re.compile(
    r"(?:\bkhong|\bnot|\bwithout|\bno|\bdon't|\bdont|\bdo not)\s+"
    r"(?:(?:thich|muon|can|want|like)\s+)?$"
)
_WORD_CHARACTER = r"a-z0-9"
_PRICE_NUMBER = (
    # Keep the grouped alternatives first so ``$1,299.99`` is consumed as one
    # amount instead of stopping at ``$1,299``.  Both common US and European
    # separator conventions are accepted.
    r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|"
    r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|"
    r"\d+(?:[.,]\d+)?)"
)
_PRICE_SCALE_UNIT = r"(?:million|trieu|nghin|ngan|tr|k|m)"
_PRICE_CURRENCY_UNIT = r"(?:dollars?|usd|vnd|dong)"
_PRICE_WORD_UNIT = rf"(?:{_PRICE_SCALE_UNIT}|{_PRICE_CURRENCY_UNIT})"
# The negative lookahead is important: without it the short unit ``m`` could
# be consumed from the start of a following word such as ``model``.
_PRICE_STRONG_SUFFIX = (
    rf"(?:\s*{_PRICE_SCALE_UNIT}(?![a-z0-9])"
    rf"(?:\s*{_PRICE_CURRENCY_UNIT}(?![a-z0-9])|\s+đ(?![a-z0-9])|\s*₫)?|"
    rf"\s*{_PRICE_CURRENCY_UNIT}(?![a-z0-9])|\s*[₫$])"
)
# ASCII ``d`` is accepted as shorthand for đồng only by regexes that have
# independent price context.  It is intentionally absent from the strong marker
# used for standalone amounts, preventing identifiers such as ``Canon 5D`` from
# becoming prices while still accepting ``price 500d``.
_PRICE_SUFFIX = rf"(?:{_PRICE_STRONG_SUFFIX}|\s*d(?![a-z0-9]))"
_PRICE_AMOUNT = rf"(?:\$\s*)?{_PRICE_NUMBER}(?:{_PRICE_SUFFIX})?"
_PRICE_STRONGLY_MARKED_AMOUNT = (
    rf"(?:\$\s*{_PRICE_NUMBER}(?:{_PRICE_STRONG_SUFFIX})?|"
    rf"{_PRICE_NUMBER}{_PRICE_STRONG_SUFFIX})"
)

_PRICE_RANGE_RE = re.compile(
    rf"(?:(?:tu|from|between|khoang)\s+)?"
    rf"(?P<low>{_PRICE_AMOUNT})\s*(?:den|toi|to|and|[-–—])\s*"
    rf"(?P<high>{_PRICE_AMOUNT})",
)
_PRICE_UPPER_RE = re.compile(
    rf"(?:duoi|thap hon|nho hon|toi da|khong qua|"
    rf"less than|under|below|up to|at most|maximum|max)"
    rf"\s*(?:muc\s*)?(?:gia|price|budget|ngan sach)?\s*(?P<amount>{_PRICE_AMOUNT})",
)
_PRICE_UPPER_POST_RE = re.compile(
    rf"(?P<amount>{_PRICE_AMOUNT})\s*(?:tro xuong|or less|or lower|maximum|max)",
)
_PRICE_LOWER_RE = re.compile(
    rf"(?:tren|cao hon|it nhat|toi thieu|tu|"
    rf"more than|greater than|over|above|at least|minimum|min)"
    rf"\s*(?:muc\s*)?(?:gia|price|budget|ngan sach)?\s*(?P<amount>{_PRICE_AMOUNT})",
)
_PRICE_LOWER_POST_RE = re.compile(
    rf"(?P<amount>{_PRICE_AMOUNT})\s*(?:tro len|or more|or higher|minimum|min)",
)
_PRICE_BUDGET_RE = re.compile(
    rf"(?:ngan sach|budget|tam gia)\s*(?:la|of|:)?\s*(?P<amount>{_PRICE_AMOUNT})",
)
_PRICE_EXACT_RE = re.compile(
    rf"(?:gia|price|cost|ngan sach|budget|tam gia)"
    rf"\s*(?:la|is|of|:)?\s*(?P<amount>{_PRICE_AMOUNT})",
)
_PRICE_MARKED_AMOUNT_RE = re.compile(
    rf"(?P<amount>{_PRICE_STRONGLY_MARKED_AMOUNT})",
)
_PRICE_CONTEXT_RE = re.compile(
    r"(?:\bgia\b|\bprice\b|\bcost\b|\bbudget\b|\bngan sach\b|\btam gia\b)"
)
_PRICE_MARKER_RE = re.compile(
    rf"(?:[$₫]|(?<![a-z0-9]){_PRICE_WORD_UNIT}(?![a-z0-9]))"
)
_EXPLICIT_SIZE_RE = re.compile(
    r"(?:size|co|kich thuoc)\s*(?:la|:)?\s*(?P<size>xxxl|xxl|xl|xs|s|m|l|\d{1,3}(?:[.,]\d)?)\b",
)
_EXPLICIT_BRAND_RE = re.compile(
    r"(?:brand|thuong hieu)\s*(?:la|:)?\s*"
    r"(?P<brand>[a-z0-9][a-z0-9&.+-]{1,30})\b",
)


def normalize_text(value: str) -> str:
    """Lowercase and remove Vietnamese accents while preserving character offsets."""

    normalized = unicodedata.normalize("NFKD", value.casefold())
    ascii_value = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return (
        ascii_value.replace("đ", "d")
        .replace("’", "'")
        .replace("–", "-")
        .replace("—", "-")
    )


def _empty_entities() -> ExtractedEntities:
    return {
        "category": None,
        "brand": None,
        "color": None,
        "size": None,
        "material": None,
        "feature": [],
        "purpose": [],
        "min_price": None,
        "max_price": None,
    }


def _alias_pattern(alias: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?<![{_WORD_CHARACTER}]){re.escape(alias)}(?![{_WORD_CHARACTER}])"
    )


def _is_negated(normalized_query: str, start: int, alias: str) -> bool:
    # "khong day" means wireless; the word "khong" is part of the entity rather
    # than a negation applied to it.
    if alias.startswith("khong "):
        return False
    prefix = normalized_query[max(0, start - 32) : start]
    return bool(_NEGATION_PREFIX.search(prefix))


def _dictionary_matches(
    query: str,
    normalized_query: str,
    aliases_by_value: dict[str, set[str]],
) -> list[EntityMatch]:
    candidates: list[tuple[int, int, int, EntityMatch]] = []
    for canonical_value, aliases in aliases_by_value.items():
        for raw_alias in aliases:
            alias = normalize_text(raw_alias)
            match = _alias_pattern(alias).search(normalized_query)
            if match is None:
                continue
            raw_match = query[match.start() : match.end()].casefold()
            valid_surfaces = _ACCENT_SENSITIVE_SURFACES.get(alias)
            if (
                valid_surfaces is not None
                and normalize_text(raw_match) != raw_match
                and raw_match not in valid_surfaces
            ):
                continue
            # Accent-sensitive aliases prevent e.g. Vietnamese "đến" (to) from
            # being mistaken for the color "đen" (black) after normalization.
            if normalize_text(raw_alias) != raw_alias.casefold():
                if raw_match != raw_alias.casefold():
                    continue
            confidence = 0.96 if alias == normalize_text(canonical_value) else 0.92
            # Very short dictionary aliases are useful but more ambiguous.
            if len(alias) <= 2:
                confidence -= 0.08
            entity: EntityMatch = {
                "value": canonical_value,
                "confidence": round(confidence, 2),
                "matched_text": query[match.start() : match.end()],
                "source": "dictionary",
            }
            if _is_negated(normalized_query, match.start(), alias):
                entity["negated"] = True
                entity["confidence"] = round(confidence - 0.04, 2)
            candidates.append((match.start(), -len(alias), match.end(), entity))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    # Keep the longest occurrence of each canonical value.
    output: list[EntityMatch] = []
    seen: set[str] = set()
    for _, _, _, entity in candidates:
        key = str(entity["value"])
        if key not in seen:
            seen.add(key)
            output.append(entity)
    return output


def _parse_numeric_value(raw_number: str, multiplier: float) -> int | float:
    compact = raw_number.strip()
    if multiplier != 1.0 and re.fullmatch(r"\d+[.,]\d{1,2}", compact):
        number = float(compact.replace(",", "."))
    elif re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", compact):
        number = float(compact.replace(",", ""))
    elif re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", compact):
        number = float(compact.replace(".", "").replace(",", "."))
    elif re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", compact):
        number = float(re.sub(r"[.,]", "", compact))
    else:
        number = float(compact.replace(",", "."))
    value = number * multiplier
    return int(value) if value.is_integer() else value


def _parse_price_amount(raw_amount: str, query_language: str) -> tuple[int | float, str]:
    normalized = normalize_text(raw_amount).strip()
    number_match = re.search(_PRICE_NUMBER, normalized)
    if number_match is None:  # guarded by the caller's regex
        raise ValueError(f"Invalid price amount: {raw_amount!r}")

    if re.search(r"(?:trieu|million|tr|m)\s*(?:vnd|dong|d|₫)?$", normalized):
        multiplier = 1_000_000.0
    elif re.search(r"(?:nghin|ngan|k)\s*(?:vnd|dong|d|₫)?$", normalized):
        multiplier = 1_000.0
    else:
        multiplier = 1.0

    if "$" in normalized or re.search(r"\b(?:usd|dollars?)\b", normalized):
        currency = "USD"
    elif (
        "₫" in normalized
        or re.search(r"(?:\bvnd\b|\bdong\b|trieu|nghin|ngan)", normalized)
        or re.search(r"d$", normalized)
        or (
            re.search(r"(?:k|tr|m)\s*$", normalized) is not None
            and query_language == "vi"
        )
    ):
        currency = "VND"
    else:
        currency = "VND" if query_language == "vi" else "UNKNOWN"

    return _parse_numeric_value(number_match.group(0), multiplier), currency


def _price_entity(
    query: str,
    normalized_query: str,
    start: int,
    end: int,
    query_language: str,
    confidence: float,
) -> EntityMatch:
    raw_amount = normalized_query[start:end]
    value, currency = _parse_price_amount(raw_amount, query_language)
    return {
        "value": value,
        "currency": currency,
        "confidence": confidence,
        "matched_text": query[start:end],
        "source": "regex",
    }


def _range_has_price_context(
    normalized_query: str,
    range_match: re.Match[str],
) -> bool:
    """Require evidence that a numeric range actually describes money.

    Plain numeric ranges are common for sizes, model numbers and ages.  A price
    range therefore needs either a currency/unit on one of its amounts or an
    adjacent price/budget phrase.  This still accepts natural queries such as
    ``from $20 to $40`` and ``price between 20 and 40``.
    """

    if _PRICE_MARKER_RE.search(range_match.group(0)):
        return True
    context_start = max(0, range_match.start() - 32)
    context = normalized_query[context_start : range_match.start()]
    return bool(_PRICE_CONTEXT_RE.search(context))


class EntityExtractor:
    """Extract bilingual product entities using dictionaries and regular expressions."""

    def detect_language(self, query: str) -> str:
        """Return ``vi`` for likely Vietnamese queries, otherwise ``en``."""

        if any(character in query for character in "ăâđêôơưĂÂĐÊÔƠƯ"):
            return "vi"
        normalized = normalize_text(query)
        words = set(re.findall(r"[a-z]+", normalized))
        return "vi" if len(words & _VIETNAMESE_MARKERS) >= 1 else "en"

    def extract(self, query: str) -> ExtractedEntities:
        """Extract entities from one query.

        Every scalar field is either ``None`` or an :class:`EntityMatch`.
        ``feature`` and ``purpose`` are lists because both commonly occur more
        than once.  Price values are normalized numbers and retain their currency.
        """

        entities = _empty_entities()
        if not isinstance(query, str) or not query.strip():
            return entities

        normalized_query = normalize_text(query)
        language = self.detect_language(query)

        scalar_dictionaries: tuple[tuple[str, dict[str, set[str]]], ...] = (
            ("category", CATEGORY_ALIASES),
            ("brand", BRAND_ALIASES),
            ("color", COLOR_ALIASES),
            ("size", SIZE_ALIASES),
            ("material", MATERIAL_ALIASES),
        )
        for field, dictionary in scalar_dictionaries:
            matches = _dictionary_matches(query, normalized_query, dictionary)
            if matches:
                entities[field] = matches[0]  # type: ignore[literal-required]

        if entities["brand"] is None:
            explicit_brand = _EXPLICIT_BRAND_RE.search(normalized_query)
            if explicit_brand is not None:
                brand_start, brand_end = explicit_brand.span("brand")
                entities["brand"] = {
                    "value": query[brand_start:brand_end].strip(),
                    "confidence": 0.88,
                    "matched_text": query[brand_start:brand_end],
                    "source": "regex",
                }

        entities["feature"] = _dictionary_matches(
            query, normalized_query, FEATURE_ALIASES
        )
        entities["purpose"] = _dictionary_matches(
            query, normalized_query, PURPOSE_ALIASES
        )

        explicit_size = _EXPLICIT_SIZE_RE.search(normalized_query)
        if explicit_size is not None:
            raw_size = explicit_size.group("size")
            size_start, size_end = explicit_size.span("size")
            entities["size"] = {
                "value": raw_size.upper() if raw_size.isalpha() else raw_size,
                "confidence": 0.98,
                "matched_text": query[size_start:size_end],
                "source": "regex",
            }

        self._extract_prices(query, normalized_query, language, entities)
        LOGGER.debug("Extracted entities from %r: %s", query, entities)
        return entities

    def extract_entities(self, query: str) -> ExtractedEntities:
        """Alias kept for pipeline components that use a descriptive method name."""

        return self.extract(query)

    def _extract_prices(
        self,
        query: str,
        normalized_query: str,
        language: str,
        entities: ExtractedEntities,
    ) -> None:
        range_matches = list(_PRICE_RANGE_RE.finditer(normalized_query))
        range_match = next(
            (
                match
                for match in range_matches
                if _range_has_price_context(normalized_query, match)
            ),
            None,
        )
        rejected_range_spans = [
            match.span()
            for match in range_matches
            if not _range_has_price_context(normalized_query, match)
        ]
        if range_match is not None:
            low_start, low_end = range_match.span("low")
            high_start, high_end = range_match.span("high")
            low = _price_entity(
                query, normalized_query, low_start, low_end, language, 0.99
            )
            high = _price_entity(
                query, normalized_query, high_start, high_end, language, 0.99
            )
            # A currency marker on one side of a range normally applies to both.
            if low["currency"] == "UNKNOWN" and high["currency"] != "UNKNOWN":
                low["currency"] = high["currency"]
            elif high["currency"] == "UNKNOWN" and low["currency"] != "UNKNOWN":
                high["currency"] = low["currency"]
            # Reversed bounds usually come from natural phrasing rather than an
            # intentional inverted interval.  Reorder only when both values use
            # the same currency; cross-currency amounts require conversion later.
            if (
                low["currency"] == high["currency"]
                and float(low["value"]) > float(high["value"])
            ):
                low, high = high, low
            entities["min_price"] = low
            entities["max_price"] = high
            return

        upper_match = _PRICE_UPPER_RE.search(normalized_query)
        if upper_match is None:
            upper_match = _PRICE_UPPER_POST_RE.search(normalized_query)
        if upper_match is None:
            upper_match = _PRICE_BUDGET_RE.search(normalized_query)
        if upper_match is not None:
            start, end = upper_match.span("amount")
            entities["max_price"] = _price_entity(
                query, normalized_query, start, end, language, 0.98
            )

        lower_match = _PRICE_LOWER_RE.search(normalized_query)
        if lower_match is None:
            lower_match = _PRICE_LOWER_POST_RE.search(normalized_query)
        if lower_match is not None and rejected_range_spans:
            amount_start, amount_end = lower_match.span("amount")
            if any(
                amount_start < rejected_end and amount_end > rejected_start
                for rejected_start, rejected_end in rejected_range_spans
            ):
                # Do not reinterpret the lower half of a rejected size/model
                # range (for example ``from 8 to 10``) as a price floor.
                lower_match = None
        if lower_match is not None:
            start, end = lower_match.span("amount")
            entities["min_price"] = _price_entity(
                query, normalized_query, start, end, language, 0.98
            )

        # An exact amount is useful for price-targeted lookup.  Represent it as
        # an inclusive one-point interval so downstream filtering can keep its
        # existing min/max API.  Only run this fallback when a directional or
        # budget constraint has not already been found.
        if entities["min_price"] is None and entities["max_price"] is None:
            exact_match = _PRICE_EXACT_RE.search(normalized_query)
            if exact_match is None:
                exact_match = _PRICE_MARKED_AMOUNT_RE.search(normalized_query)
            if exact_match is not None:
                start, end = exact_match.span("amount")
                exact = _price_entity(
                    query, normalized_query, start, end, language, 0.97
                )
                entities["min_price"] = dict(exact)
                entities["max_price"] = dict(exact)


_DEFAULT_EXTRACTOR = EntityExtractor()


def extract_entities(query: str) -> ExtractedEntities:
    """Convenience function for callers that do not need a configured instance."""

    return _DEFAULT_EXTRACTOR.extract(query)


def entity_value(entity: EntityMatch | None, default: Any = None) -> Any:
    """Read a normalized value from a nullable entity match."""

    return entity.get("value", default) if entity else default
