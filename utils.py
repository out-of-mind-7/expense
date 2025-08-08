import dateparser

def normalize_date(date_str):
    parsed = dateparser.parse(date_str)
    return parsed.strftime("%Y-%m-%d") if parsed else None