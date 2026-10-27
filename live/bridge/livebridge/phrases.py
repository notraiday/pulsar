"""The few short lines the bridge says itself, not the dialog model (they have to be instant): a handful of ways to say
each, never the same one twice in a row. A fixed phrase heard every time is what gives a machine away."""
import random

POOLS = {
    # the owner has answered the agent's question (surface.py)
    "ack_code": ["Ввожу.", "Так, ввожу.", "Ага, ввожу код.", "Секунду, ввожу."],
    "ack_yes": ["Хорошо.", "Ага, делаю.", "Поняла, делаю.", "Окей.", "Сейчас."],
    "ack_no": ["Поняла, не делаю.", "Ладно, не буду.", "Хорошо, не буду.", "Ок, не делаю."],
    "ack_text": ["Поняла.", "Ага.", "Ясно.", "Угу, поняла.", "Хорошо."],
    # before asking again / giving up on an answer that could not be made out
    "again": ["Не расслышала. ", "Прости, не поняла. ", "Ещё раз? ", "Не разобрала. "],
    "skip": ["Не разобрала, ладно, пропущу.", "Не поняла — пока пропущу.", "Не расслышала, пропускаю."],
    # before a background job's question asked in a dialog that did not start it
    "lead": ["Слушай, ", "Кстати, ", "Слушай, тут по тому делу: ", "Вопрос по тому делу: "],
}

_last: dict[str, str] = {}


def say(kind: str) -> str:
    pool = POOLS[kind]
    choice = random.choice([p for p in pool if p != _last.get(kind)] or pool)
    _last[kind] = choice
    return choice


def ack(kind: str, value: str = "") -> str:
    """What is said once the owner's answer to the agent's question is taken."""
    return say({"code": "ack_code", "text": "ack_text"}.get(kind) or ("ack_yes" if value == "yes" else "ack_no"))
