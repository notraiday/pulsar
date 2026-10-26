"""Hand-off: the bridge does not take this dialog. The station runs its own pipeline (Home Assistant's
conversation agent) and replays the last moments of the microphone into it, so the first words are not lost."""
import logging

log = logging.getLogger("live")


async def run(station, http, options):
    log.info("dialog from %s handed off", station.peer)
    await station.finish(b"R")
