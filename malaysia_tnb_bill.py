"""TNB domestic (Tariff A, Peninsular Malaysia) bill estimator, post 1 July 2025 structure.

Rates are collected from published TNB / Energy Commission material (Oct 2026).
CHECK them against your own bill before presenting. Items marked VERIFY come from secondary sources.
"""
import math

GEN_LOW, GEN_HIGH = 0.2703, 0.3703   # RM/kWh, generation charge (<=1500 / >1500 kWh)
CAPACITY = 0.0455                    # RM/kWh
NETWORK = 0.1285                     # RM/kWh
RETAIL = 10.00                       # RM/month, waived at or below protection threshold
KWTBB_RATE = 0.016                   # VERIFY: 1.6% RE levy, exempt <= 300 kWh
SST_RATE = 0.08                      # 8% service tax
SST_FREE_KWH = 600                   # VERIFY: SST applies only to kWh above 600

# Energy Efficiency Incentive: (upper bound of band in kWh, sen/kWh), applied to ALL kWh
EEI_BANDS = [
    (200, 25.00), (250, 24.50), (300, 22.50), (350, 21.00), (400, 17.00),
    (450, 14.50), (500, 12.00), (550, 10.50), (600, 9.00), (650, 7.50),
    (700, 5.50), (750, 4.50), (800, 4.00), (850, 2.50), (900, 1.00), (1000, 0.50),
]


def eei_rate_sen(kwh: int) -> float:
    for upper, rate in EEI_BANDS:
        if kwh <= upper:
            return rate
    return 0.0


def calc_bill(kwh: float, afa_sen: float = 0.0, protect_kwh: int = 600) -> dict:
    """Estimate one month's bill (RM).

    afa_sen: Automatic Fuel Adjustment in sen/kWh (can be negative = rebate).
    protect_kwh: usage at or below this is exempt from retail charge, AFA and SST
                 (600 normally; 800 for Sep-Dec 2026 under the temporary extension).
    """
    kwh = max(int(round(kwh)), 0)
    gen_rate = GEN_LOW if kwh <= 1500 else GEN_HIGH
    all_in = gen_rate + CAPACITY + NETWORK

    generation = kwh * gen_rate
    capacity = kwh * CAPACITY
    network = kwh * NETWORK
    retail = RETAIL if kwh > protect_kwh else 0.0
    eei = -kwh * eei_rate_sen(kwh) / 100
    afa = kwh * afa_sen / 100 if kwh > protect_kwh else 0.0

    kwtbb_base = generation + capacity + network + afa + eei
    kwtbb = KWTBB_RATE * kwtbb_base if kwh > 300 else 0.0
    sst = SST_RATE * max(kwh - SST_FREE_KWH, 0) * all_in if kwh > protect_kwh else 0.0

    total = generation + capacity + network + retail + eei + afa + kwtbb + sst
    return {
        "kwh": kwh,
        "Generation charge": generation,
        "Capacity charge": capacity,
        "Network charge": network,
        "Retail charge": retail,
        "Energy Efficiency Incentive": eei,
        "AFA": afa,
        "KWTBB (1.6%)": kwtbb,
        "Service tax (8%)": sst,
        "Total": total,
    }