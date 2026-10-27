"""What the owner said, read as an answer to a question the agent asked through the speakers (surface.py):
a one-time code ("четыре пять один два", "сорок пять двенадцать", "4512") or a yes / no."""
import re

_UNITS = {"ноль": 0, "нуль": 0, "один": 1, "одна": 1, "одну": 1, "раз": 1, "два": 2, "две": 2, "три": 3,
          "четыре": 4, "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9}
_TEENS = {"десять": 10, "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13, "четырнадцать": 14,
          "пятнадцать": 15, "шестнадцать": 16, "семнадцать": 17, "восемнадцать": 18, "девятнадцать": 19}
_TENS = {"двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50, "шестьдесят": 60, "семьдесят": 70,
         "восемьдесят": 80, "девяносто": 90}
_HUNDREDS = {"сто": 100, "двести": 200, "триста": 300, "четыреста": 400, "пятьсот": 500, "шестьсот": 600,
             "семьсот": 700, "восемьсот": 800, "девятьсот": 900}

_WORD_RE = re.compile(r"\d+|[а-яё]+", re.I)


def digits(text: str) -> str:
    """The digits of a dictated code. Number words make groups the way people read codes out: "сорок пять
    двенадцать" is 45 12, "сто двадцать три" is 123, "четыре пять" is 4 5; a word that cannot continue the
    current group (a unit after a unit, a ten after a ten) starts a new one. Other words are ignored."""
    out, group, slots = [], 0, set()  # slots of the group already filled: "h", "t", "u"

    def close():
        nonlocal group, slots
        if slots:
            out.append(str(group))
        group, slots = 0, set()

    for w in _WORD_RE.findall((text or "").lower().replace("ё", "е")):
        if w.isdigit():
            close()
            out.append(w)
        elif w in _HUNDREDS:
            close()
            group, slots = _HUNDREDS[w], {"h"}
        elif w in _TENS:
            if slots & {"t", "u"}:
                close()
            group += _TENS[w]
            slots.add("t")
        elif w in _TEENS:
            if slots & {"t", "u"}:
                close()
            group += _TEENS[w]
            slots |= {"t", "u"}
        elif w in _UNITS:
            if "u" in slots or (_UNITS[w] == 0 and slots):
                close()
            group += _UNITS[w]
            slots.add("u")
            if _UNITS[w] == 0 and len(slots) == 1:
                close()  # a lone zero is its own digit ("ноль пять" = 05)
        # anything else ("код", "это", "ну") is filler
    close()
    return "".join(out)


_YES = {"да", "ага", "угу", "давай", "давайте", "подтверждаю", "подтверждаем", "оплачивай", "оплати", "оплачивайте",
        "плати", "заказывай", "закажи", "конечно", "верно", "можно", "хорошо", "ок", "окей", "okay", "ok", "yes",
        "согласен", "согласна", "поехали", "вперед", "го"}
_NO = {"нет", "неа", "не", "отмена", "отмени", "отменить", "стоп", "стой", "подожди", "погоди", "no", "нельзя",
       "хватит", "передумал", "передумала"}


def yes_no(text: str) -> str:
    """"yes", "no" or "unclear". Any no-word wins ("не надо", "да нет, стоп" are no): a mistaken no costs a
    repeat, a mistaken yes can cost money."""
    words = set(_WORD_RE.findall((text or "").lower().replace("ё", "е")))
    if words & _NO:
        return "no"
    if words & _YES:
        return "yes"
    return "unclear"
