"""shape. The words and markers text arrives with, per language."""

__all__ = [
    "NAME_TITLES",
    "OTP_CUES",
    "SPACE_UNICODES",
    "SPOKEN_AT",
    "SPOKEN_DIGITS",
    "SPOKEN_DOT",
    "THINKING_MARKERS",
]


# --- invisible characters ----------------------------------------------------

SPACE_UNICODES = ["\u202f", "\u00a0", "\u2009"]


# --- what a reasoning model wraps its thinking in ----------------------------

THINKING_MARKERS: list[tuple[str, str]] = [
    ("<think>", "</think>"),
    ("<reasoning>", "</reasoning>"),
    ("<thought>", "</thought>"),
    ("<internal>", "</internal>"),
    ("<scratchpad>", "</scratchpad>"),
    ("[think]", "[/think]"),
    ("[reasoning]", "[/reasoning]"),
    ("[thought]", "[/thought]"),
]


# --- how a digit is said aloud -----------------------------------------------

SPOKEN_DIGITS: dict[str, str] = {
    "vi": r"khong|không|chin|chín|bay|bon|bảy|bốn|hai|lam|lăm|mot|mốt|một|nam|năm|sau|sáu|tam|tám|ba|tu|tư",
    "en": r"eight|seven|three|five|four|nine|zero|one|six|two|oh",
}


# --- how an address is said aloud --------------------------------------------

SPOKEN_AT: dict[str, str] = {"vi": r"a\s+cong|a\s+còng|a\s+moc|a\s+móc|at", "en": r"at"}

SPOKEN_DOT: dict[str, str] = {"vi": r"cham|chấm|dot", "en": r"point|dot"}


# --- what announces a name ---------------------------------------------------

NAME_TITLES: dict[str, str] = {
    "vi": r"ten\s+toi\s+la|tên\s+tôi\s+là|toi\s+ten|tôi\s+tên|anh|bac|bác|chi|chu|chú|chị|ong|ten|tên|ông|ba|bà|co|cô|em",
    "en": r"my\s+name\s+is|customer|i\s+am|miss|mrs|dr|mr|ms",
}


# --- what announces a one-time code ------------------------------------------

OTP_CUES: dict[str, str] = {
    "vi": r"ma\s+xac\s+nhan|ma\s+xac\s+thuc|mã\s+xác\s+nhận|mã\s+xác\s+thực|ma\s+bao\s+mat|mã\s+bảo\s+mật|ma\s+otp|ma\s+pin|mã\s+otp|mã\s+pin|otp",
    "en": r"one\s+time\s+password|verification\s+code|security\s+code|otp|pin",
}
