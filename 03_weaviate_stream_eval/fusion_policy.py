"""Resolve the RRF policy from the amount of observed user history."""

FUSION_BANDS = ("1–9", "10–29", "30+")


def history_band(ratings: int) -> str:
    if ratings < 10:
        return FUSION_BANDS[0]
    if ratings < 30:
        return FUSION_BANDS[1]
    return FUSION_BANDS[2]


def strengthened_weight(ratings: int) -> float:
    if ratings < 30:
        return 0.50
    return 0.35 + 0.40 * ratings / (ratings + 20)


def recommendation_fusion_parameters(ratings: int) -> tuple[float, int]:
    """ALS weight and RRF depth constant for the validated recommendation policy."""
    if ratings <= 0:
        return 0.50, 60
    if ratings < 10:
        return 0.70, 100
    return 0.60, 60


def fusion_parameters(rating_count: int, settings: dict) -> tuple[float, int]:
    choice = settings[history_band(rating_count)]
    if choice["policy"] == "strengthened":
        return strengthened_weight(rating_count), 60
    return choice["als_weight"], choice["rrf_k"]
