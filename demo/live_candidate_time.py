"""Observed publication calendar metadata for live navigation, never fake clocks."""
from datetime import date, datetime
import re
import unicodedata

MONTHS = {
    "january":1,"januari":1,"januar":1,"janvier":1,"enero":1,"gennaio":1,"janeiro":1,
    "february":2,"februari":2,"februar":2,"fevrier":2,"febrero":2,"febbraio":2,"fevereiro":2,
    "march":3,"maart":3,"marz":3,"mars":3,"marzo":3,"marco":3,
    "april":4,"avril":4,"abril":4,"aprile":4,
    "may":5,"mei":5,"mai":5,"mayo":5,"maggio":5,"maio":5,
    "june":6,"juni":6,"juin":6,"junio":6,"giugno":6,"junho":6,
    "july":7,"juli":7,"juillet":7,"julio":7,"luglio":7,"julho":7,
    "august":8,"augustus":8,"aout":8,"agosto":8,
    "september":9,"septembre":9,"septiembre":9,"settembre":9,"setembro":9,
    "october":10,"oktober":10,"octobre":10,"octubre":10,"ottobre":10,"outubro":10,
    "november":11,"novembre":11,"noviembre":11,"novembro":11,
    "december":12,"decembre":12,"diciembre":12,"dicembre":12,"dezembro":12,"dezember":12,
}


def observed_calendar_day(value):
    if not isinstance(value,str) or not value.strip():return None
    value="".join(c for c in unicodedata.normalize("NFKD",value.casefold()) if not unicodedata.combining(c))
    iso=re.search(r"(?<!\d)(20\d\d)[-年/](\d{1,2})[-月/](\d{1,2})(?:日|\b|T)",value,re.I)
    if iso:
        try:return date(*map(int,iso.groups()))
        except ValueError:return None
    words=re.sub(r"[,.-]"," ",value).split()
    for i,word in enumerate(words):
        if word not in MONTHS:continue
        try:
            if i>0 and words[i-1].isdigit() and i+1<len(words) and re.fullmatch(r"20\d\d",words[i+1]):
                return date(int(words[i+1]),MONTHS[word],int(words[i-1]))
            if i+2<len(words) and words[i+1].isdigit() and re.fullmatch(r"20\d\d",words[i+2]):
                return date(int(words[i+2]),MONTHS[word],int(words[i+1]))
        except ValueError:return None
    return None


def candidate_time_scope(candidate, at):
    """A date is not a timestamp. Two calendar days cover timezone uncertainty."""
    stamp=datetime.fromisoformat(at.replace("Z","+00:00"))
    for field in ("source_published_at","publication_raw"):
        day=observed_calendar_day(candidate.get(field))
        if day is not None:break
    else:return {"status":"unknown","source_published_at_imputed":False,"calendar_day":None}
    age=(stamp.date()-day).days
    status="stale" if age>32 else "future" if age < -2 else "recent"
    return {"status":status,"calendar_day":day.isoformat(),"source_field":field,
            "age_calendar_days":age,"max_age_days":30,"timezone_guard_days":2,
            "source_published_at_imputed":False,
            "basis":"observed_publication_metadata_not_url_or_first_availability"}
