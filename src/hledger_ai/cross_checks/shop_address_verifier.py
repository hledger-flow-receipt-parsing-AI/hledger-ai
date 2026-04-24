"""Cross-check: verify shop address against known locations."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from typeguard import typechecked

log = logging.getLogger(__name__)


@typechecked
class ShopAddressVerifier:
    """Checks if the extracted shop name + address matches known
    locations from previous labels.

    A match boosts category confidence. A mismatch flags for review.
    """

    name = "shop_address_verifier"

    def __init__(
        self,
        *,
        known_locations: Optional[
            Dict[str, List[Dict[str, str]]]
        ] = None,
    ) -> None:
        # shop_name (lower) -> list of known address dicts
        # Each address dict has keys: street, city, zipcode, etc.
        self._known: Dict[str, List[Dict[str, str]]] = (
            known_locations or {}
        )

    def apply(self, merged: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(merged)

        shop_name = self._get_shop_name(merged)
        address = self._get_address(merged)

        if not shop_name or not address:
            return result

        normalized = shop_name.strip().lower()
        known_addrs = self._known.get(normalized, [])

        if not known_addrs:
            return result

        result["_checks"] = result.get("_checks", {})

        for known in known_addrs:
            if self._addresses_match(address, known):
                log.info(
                    "Shop address verified: '%s' at "
                    "known location",
                    shop_name,
                )
                result["_checks"]["shop_address"] = "verified"

                # Boost category confidence if present
                conf = result.get("category_confidence", 0.0)
                if isinstance(conf, (int, float)) and conf > 0:
                    result["category_confidence"] = min(
                        1.0, conf + 0.05
                    )
                return result

        log.warning(
            "Shop '%s' address does not match any known "
            "location: %s",
            shop_name,
            address,
        )
        result["_checks"]["shop_address"] = "unverified"
        return result

    def _get_shop_name(
        self, merged: Dict[str, Any]
    ) -> Optional[str]:
        shop_id = merged.get("shop_identifier")
        if isinstance(shop_id, dict):
            return shop_id.get("name")
        return None

    def _get_address(
        self, merged: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        shop_id = merged.get("shop_identifier")
        if isinstance(shop_id, dict):
            addr = shop_id.get("address")
            if isinstance(addr, dict):
                return addr
        return None

    def _addresses_match(
        self,
        extracted: Dict[str, Any],
        known: Dict[str, str],
    ) -> bool:
        """Fuzzy match: city and street must match (case-insensitive)."""
        for key in ("city", "street"):
            ext_val = extracted.get(key, "")
            known_val = known.get(key, "")
            if not ext_val or not known_val:
                continue
            if str(ext_val).strip().lower() != str(
                known_val
            ).strip().lower():
                return False
        # At least one field must have matched
        matched_any = any(
            extracted.get(k) and known.get(k)
            for k in ("city", "street", "zipcode")
        )
        return matched_any
