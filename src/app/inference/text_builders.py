import re

def property_to_text(row):
    return (
        f"{row['deal_type']} {row['property_type']} in {row['city']} {row['state']}. "
        f"{int(row['beds'])} bed {row['baths']} bath, {int(row['sqft'])} sqft. "
        f"Purchase {int(row['purchase_price'])}, ARV {int(row['arv'])}, "
        f"Entry {int(row['entry_fee'])}, Payment {row['estimated_monthly_payment']}. "
        f"Condition {row['condition']}, Occupancy {row['occupancy']}."
    )

def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", str(text).lower())

def detect_missing_criteria(text: str):
    t = text.lower()
    has_beds = re.search(r"\d+\s*\+?\s*bed", t) is not None
    has_price = re.search(r"(under|max|<=)\s*\$?\s*[\d\.,]+[km]?", t) is not None or "$" in t
    has_location = any(k in t for k in [
        "az","arizona","tx","texas","fl","florida","ga","georgia","nc","north carolina",
        "sc","south carolina","tn","tennessee","ca","california"
    ])
    missing = []
    if not has_location: missing.append("location (city/state)")
    if not has_beds: missing.append("bedrooms (e.g., 3 bed)")
    if not has_price: missing.append("max purchase price (e.g., under 350k)")
    return missing
