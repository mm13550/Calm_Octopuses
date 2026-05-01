"""
tests/test_menu_prices.py
=========================
Regression tests for display-safe menu price formatting.
"""

from core.data_loader import _format_menu_price, build_restaurant_catalog, load_menus_df


def test_menu_price_formatter_handles_common_price_shapes():
    assert _format_menu_price("") == ""
    assert _format_menu_price(None) == ""
    assert _format_menu_price("27") == "$27"
    assert _format_menu_price("10 / 19 / 38") == "$10 / $19 / $38"
    assert _format_menu_price("+20") == "+$20"
    assert _format_menu_price("$18") == "$18"
    assert _format_menu_price("13 (each) / 32 (all three)") == "$13 (each) / $32 (all three)"
    assert _format_menu_price("4 / 10 (3 scoops)") == "$4 / $10 (3 scoops)"
    assert _format_menu_price("Ask your server about today's creation") == ""
    assert _format_menu_price("鈥?25") == ""


def test_atomix_menu_prices_are_empty_not_garbled():
    catalog = build_restaurant_catalog()
    atomix_ids = set(
        catalog[catalog["restaurant_name"].str.contains("Atomix", case=False, na=False)]["rest_id"]
    )
    menus = load_menus_df()
    atomix_rows = menus[menus["rest_id"].isin(atomix_ids)]

    assert not atomix_rows.empty
    assert atomix_rows["price"].fillna("").map(_format_menu_price).eq("").all()


def test_all_menu_prices_format_without_mojibake_markers():
    menus = load_menus_df()
    formatted_prices = menus["price"].fillna("").map(_format_menu_price)
    forbidden_markers = ("�", "鈥", "Â", "â", "ã", "猸", "бд", "路")

    for marker in forbidden_markers:
        assert not formatted_prices.str.contains(marker, regex=False).any()
