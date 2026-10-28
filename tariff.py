"""
TNB domestic tariff (Peninsular Malaysia, RP4 structure effective 1 July 2025).

Rates below were checked on 28 Oct 2026 against published guides (Paultan, SoyaCincau, RinggitPlus,
Malaysia4U, Business Today). Tariffs, the AFA and temporary relief measures change - verify with
TNB's official bill calculator (www.mytnb.com.my/tariff) before relying on exact figures.
"""
from __future__ import annotations

import math

TARIFF_CHECKED = "28 Oct 2026"

# ---- energy components (sen/kWh) -------------------------------------------------------------
GEN_LOW, GEN_HIGH, GEN_LIMIT = 27.03, 37.03, 1500     # generation charge, step-up above 1,500 kWh
CAPACITY, NETWORK = 4.55, 12.85
RETAIL_RM = 10.0                                      # per month, waived at/below the exemption limit
KWTBB_RATE, KWTBB_EXEMPT_KWH = 0.016, 300             # renewable-energy levy, exempt up to 300 kWh
SST_RATE, SST_BASE_KWH = 0.08, 600                    # 8% service tax on kWh above 600
EEI_MAX_KWH = 1000                                    # Energy Efficiency Incentive stops above 1,000 kWh

# Energy Efficiency Incentive: (upper bound of band in kWh, sen/kWh rebate on ALL kWh)
EEI_BANDS = [(200, 25.0), (250, 24.5), (300, 22.5), (350, 21.0), (400, 17.0), (450, 14.5),
             (500, 12.0), (550, 10.5), (600, 9.0), (650, 7.5), (700, 5.5), (750, 4.5),
             (800, 4.0), (850, 2.5), (900, 1.0), (1000, 0.5)]

# ---- AFA (sen/kWh, + = surcharge, - = rebate) ------------------------------------------------
AFA_BY_MONTH = {
    "2025-07": (0.00, "published"), "2025-08": (-1.45, "published"), "2025-09": (-1.10, "published"),
    "2025-10": (-6.50, "published"), "2025-11": (-8.91, "published"), "2025-12": (-6.42, "published"),
    "2026-01": (-4.99, "published"), "2026-02": (-2.77, "published"), "2026-03": (-2.15, "published"),
    "2026-04": (-0.47, "published"), "2026-05": (1.38, "published"), "2026-06": (2.59, "published"),
    "2026-07": (3.59, "published"), "2026-08": (3.80, "published"), "2026-09": (3.67, "published"),
    "2026-10": (3.36, "TNB forecast"), "2026-11": (2.81, "TNB forecast"), "2026-12": (5.48, "TNB forecast"),
}
AFA_DEFAULT_LATER = 3.8   # assumed for months with no published/forecast rate (about the Jul-Dec 2026 level)

# Temporary relief: AFA, retail charge and SST exemption raised from 600 to 800 kWh (Sep-Dec 2026)
RELIEF_START, RELIEF_END, RELIEF_KWH = "2026-09", "2026-12", 800


def _key(period) -> str:
    return f"{period.year}-{period.month:02d}"


def exemption_threshold(period, mode: str = "auto") -> int:
    """Usage (kWh) up to which AFA, retail charge and service tax are waived."""
    if mode == "600":
        return 600
    if mode == "800":
        return 800
    return RELIEF_KWH if RELIEF_START <= _key(period) <= RELIEF_END else 600


def afa_for(period, override: float | None = None, later_default: float = AFA_DEFAULT_LATER):
    """Returns (AFA sen/kWh, status text) for a month."""
    if override is not None:
        return float(override), "your override"
    if _key(period) in AFA_BY_MONTH:
        return AFA_BY_MONTH[_key(period)]
    return float(later_default), "assumed"


def eei_sen(kwh: float) -> float:
    k = int(math.floor(kwh + 0.5))
    if k > EEI_MAX_KWH:
        return 0.0
    for upper, rate in EEI_BANDS:
        if k <= upper:
            return rate
    return 0.0


def compute_bill(kwh: float, afa_sen: float, threshold: int = 600, tier_all: bool = True) -> dict:
    """Monthly bill (RM) for one domestic account.

    tier_all=True : once usage passes 1,500 kWh the 37.03 sen generation rate applies to ALL kWh
                    (Paultan / Malaysia4U reading).
    tier_all=False: only the kWh above 1,500 are charged 37.03 sen (RinggitPlus 'two-block' reading).
    """
    kwh = max(float(kwh), 0.0)
    if tier_all:
        gen = kwh * (GEN_LOW if kwh <= GEN_LIMIT else GEN_HIGH) / 100
    else:
        gen = (min(kwh, GEN_LIMIT) * GEN_LOW + max(kwh - GEN_LIMIT, 0) * GEN_HIGH) / 100
    cap, net = kwh * CAPACITY / 100, kwh * NETWORK / 100
    energy = gen + cap + net

    over = kwh > threshold
    retail = RETAIL_RM if over else 0.0
    afa = kwh * afa_sen / 100 if over else 0.0
    eei = -kwh * eei_sen(kwh) / 100
    kwtbb = KWTBB_RATE * (energy + afa + eei) if kwh > KWTBB_EXEMPT_KWH else 0.0
    avg_energy_rate = energy / kwh if kwh else 0.0
    sst = SST_RATE * avg_energy_rate * (kwh - SST_BASE_KWH) if over else 0.0

    parts = {"Generation": gen, "Capacity": cap, "Network": net, "Retail charge": retail,
             "AFA": afa, "Energy Efficiency Incentive": eei, "RE levy (KWTBB)": kwtbb,
             "Service tax (8%)": sst}
    total = max(sum(parts.values()), 3.0 if kwh > 0 else 0.0)   # RM3 minimum charge
    return {"kwh": kwh, "parts": {k: round(v, 2) for k, v in parts.items()},
            "total": round(total, 2), "sen_per_kwh": round(total / kwh * 100, 1) if kwh else 0.0,
            "threshold": threshold, "afa_sen": afa_sen}


# ---- assessment & recommendations ------------------------------------------------------------
def level(kwh: float, threshold: int) -> tuple[str, str]:
    """(label, colour key) describing how heavy the usage is under the tariff rules."""
    if kwh > GEN_LIMIT:
        return "Very high", "red"
    if kwh > EEI_MAX_KWH:
        return "High", "red"
    if kwh > threshold:
        return "Above exemption limit", "orange"
    if kwh > KWTBB_EXEMPT_KWH:
        return "Moderate", "blue"
    return "Low", "green"


def savings_table(kwh, afa, threshold, tier_all, pcts=(5, 10, 15, 20)):
    base = compute_bill(kwh, afa, threshold, tier_all)["total"]
    rows = []
    for p in pcts:
        k = kwh * (1 - p / 100)
        b = compute_bill(k, afa, threshold, tier_all)["total"]
        rows.append({"Reduce usage by": f"{p}%", "New usage (kWh)": round(k), "New bill (RM)": b,
                     "Saving (RM)": round(base - b, 2), "Saving (%)": round((base - b) / base * 100, 1) if base else 0})
    return rows


def cliffs(kwh, afa, threshold, tier_all, max_cut_pct=35):
    """Thresholds just below current usage where the bill structure improves."""
    base = compute_bill(kwh, afa, threshold, tier_all)["total"]
    cands = [(f"{threshold} kWh: AFA, retail charge and service tax are waived", threshold),
             (f"{EEI_MAX_KWH:,} kWh: Energy Efficiency Incentive applies again", EEI_MAX_KWH),
             (f"{GEN_LIMIT:,} kWh: lower generation rate", GEN_LIMIT),
             (f"{KWTBB_EXEMPT_KWH} kWh: RE levy waived", KWTBB_EXEMPT_KWH)]
    out = []
    for name, limit in cands:
        if kwh > limit and (kwh - limit) / kwh * 100 <= max_cut_pct:
            nb = compute_bill(limit, afa, threshold, tier_all)["total"]
            out.append({"target": name, "cut_kwh": round(kwh - limit), "cut_pct": round((kwh - limit) / kwh * 100, 1),
                        "bill_after": nb, "saving": round(base - nb, 2)})
    return out


def recommendations(kwh, bill, afa, threshold, tier_all, usual_bill=None):
    """List of (severity, title, text). severity in {'success','info','warning','error'}."""
    out = []
    lab, _ = level(kwh, threshold)
    total = bill["total"]

    if kwh > threshold:
        cut = kwh - threshold
        nb = compute_bill(threshold, afa, threshold, tier_all)["total"]
        out.append(("error" if kwh > EEI_MAX_KWH else "warning",
                    f"Usage is above the {threshold} kWh exemption limit",
                    f"At about {kwh:,.0f} kWh, retail charge, AFA ({afa:+.2f} sen/kWh) and 8% service tax apply, "
                    f"adding RM{bill['parts']['Retail charge'] + bill['parts']['AFA'] + bill['parts']['Service tax (8%)']:,.2f} to the bill. "
                    f"Cutting about {cut:,.0f} kWh ({cut / kwh * 100:.0f}%) to reach {threshold} kWh would lower the bill "
                    f"from RM{total:,.2f} to about RM{nb:,.2f}."))
    elif kwh > threshold * 0.9:
        out.append(("warning", f"Close to the {threshold} kWh limit",
                    f"Only {threshold - kwh:,.0f} kWh below the point where retail charge, AFA and service tax start. "
                    "A hot month can push usage over it."))
    elif kwh > KWTBB_EXEMPT_KWH:
        out.append(("info", "Usage is in the moderate range",
                    f"About {threshold - kwh:,.0f} kWh below the {threshold} kWh limit. "
                    "Energy Efficiency Incentive is still reducing the bill."))
    else:
        out.append(("success", "Low usage",
                    "Usage is at or below 300 kWh: the RE levy is waived and the Energy Efficiency Incentive is at its highest."))

    if kwh > EEI_MAX_KWH:
        out.append(("error", "No Energy Efficiency Incentive above 1,000 kWh",
                    "Every kWh is charged at the full rate. Bringing usage back under 1,000 kWh restores part of the incentive."))
    if kwh > GEN_LIMIT:
        out.append(("error", "Above 1,500 kWh the generation rate steps up",
                    "Generation rises from 27.03 to 37.03 sen/kWh. Check the tier setting in the sidebar to see how this is applied."))
    if usual_bill and total > usual_bill * 1.10:
        out.append(("warning", "Higher than your usual month",
                    f"This month's bill (RM{total:,.2f}) is {(total / usual_bill - 1) * 100:.0f}% above the 12-month average "
                    f"projection (RM{usual_bill:,.2f})."))
    if kwh > 800:
        out.append(("info", "Consider a Time-of-Use (ToU) tariff",
                    "Households with a smart meter can opt into ToU, which has a lower off-peak energy rate. It tends to suit "
                    "homes with higher usage that can shift loads outside weekday 2pm-10pm. Compare using TNB's calculator before switching."))
    return out


GENERAL_TIPS = {
    "high": [
        "Air-conditioning is usually the biggest load: set it to 24-25 °C, use a fan to circulate air, and clean the filter regularly.",
        "Run the AC only in rooms in use and close doors/curtains during the hottest hours.",
        "Switch off devices at the socket instead of leaving them on standby (TV, game consoles, chargers).",
        "Run washing machines and dishwashers with full loads and air-dry clothes when the weather allows.",
        "Replace old fridges and ACs with inverter, star-rated models when it is time to upgrade.",
    ],
    "normal": [
        "Keep the AC at a moderate temperature and service it regularly.",
        "Unplug or switch off idle devices to avoid standby use.",
        "Use LED lighting and switch off lights in empty rooms.",
    ],
}