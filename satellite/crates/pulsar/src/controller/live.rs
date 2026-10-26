//! Live dialogs through the bridge (`pulsar-live`): the speaker only streams the microphone and
//! plays what comes back; the bridge decides what a dialog is, and may hand it to the Home
//! Assistant pipeline instead. Without a bridge address, or when the bridge cannot be reached,
//! the HA pipeline answers as before.

use log::{info, warn};
use pulsar_core::UiState;
use pulsar_live::{End, EndKind, LiveEvent, LiveState, StartParams};
use pulsar_media::Channel;

use super::logic::{self, Phase};
use super::{Controller, SOUNDS, Tk};

/// A dialog is ended after this long.
const LIVE_MAX_MS: u64 = 15 * 60 * 1000;
/// Longest wait for the last words of an ended dialog to play out.
const LIVE_TAIL_MAX_MS: u64 = 20_000;

/// The dialog in progress, or one the bridge ended whose last words are still playing.
pub(super) struct Dialog {
    id: u64,
    closing: bool,
    ui: UiState,
}

impl Dialog {
    pub(super) fn is_closing(&self) -> bool {
        self.closing
    }

    /// What the ring and the clock show: the bridge's state; once it ended, the last words.
    pub(super) fn ui_state(&self) -> UiState {
        if self.closing {
            UiState::Speaking
        } else {
            self.ui
        }
    }
}

/// A JSON string literal; the values are ids and names, so only the escapes JSON requires.
fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if c < ' ' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

impl Controller {
    /// A dialog is running or its last words are playing.
    pub(super) fn live_active(&self) -> bool {
        self.live_dialog.is_some()
    }

    /// The microphone goes to the bridge.
    pub(super) fn live_streaming(&self) -> bool {
        self.live_dialog.as_ref().is_some_and(|d| !d.closing)
    }

    /// The bridge to talk to: the `live_bridge` file of the state directory (one line,
    /// `host:port`, or `off`), read at each dialog so it can be changed without a restart, else
    /// the `live_bridge=` key of state.conf. None: Home Assistant answers.
    pub(super) fn live_bridge(&self) -> Option<String> {
        let file = std::fs::read_to_string(logic::in_dir(&self.state_dir, "live_bridge"))
            .unwrap_or_default();
        let file = file.lines().next().unwrap_or_default().trim();
        if file == "off" {
            return None;
        }
        [file, self.st.live_bridge.trim()]
            .into_iter()
            .find(|s| !s.is_empty())
            .map(str::to_owned)
    }

    /// `preroll`: what was said since the wake word (16 kHz), sent first.
    pub(super) fn start_live(&mut self, bridge: String, preroll: Vec<i16>) {
        self.cancel_arbitration();
        if self.live_active() {
            self.teardown_live(); // the last words of an ended dialog give way to this one
        }
        if self.phase != Phase::Idle {
            self.finish_interaction();
        }
        info!("live: starting a dialog via {bridge}");
        let hello = format!(
            r#"{{"wake_word":{},"station":{},"protocol":1}}"#,
            json_str(&self.st.wake_word),
            json_str(&self.station)
        );
        let id = self.live.start(StartParams {
            bridge,
            hello,
            preroll,
        });
        self.live_dialog = Some(Dialog {
            id,
            closing: false,
            ui: UiState::Listening,
        });
        self.commands_blocked_until_ms = 0;
        let sound = format!("{SOUNDS}{}", self.model.wake_sound);
        self.player.play(Channel::Announce, vec![sound]);
        self.mic.set_streaming(true);
        self.timers.set(Tk::LiveMax, LIVE_MAX_MS);
        self.update_duck();
        self.apply_gains(); // the new session's mute
        self.update_ui();
    }

    /// The owner ends the dialog (button, "стоп", the microphone muted) or it hit its limit: the
    /// answer is cut and the bridge told. A dialog the bridge already ended gives up its last
    /// words.
    pub(super) fn end_live(&mut self, why: &str) {
        let Some(d) = &self.live_dialog else {
            return;
        };
        if !d.closing {
            info!("live: ending the dialog ({why})");
            self.live.end(why);
        }
        self.teardown_live();
    }

    pub(super) fn teardown_live(&mut self) {
        self.timers.cancel(Tk::LiveMax);
        self.timers.cancel(Tk::LiveTail);
        self.live.teardown();
        self.live_dialog = None;
        self.mic
            .set_streaming(self.phase == Phase::Listening || self.arbitrating);
        self.update_duck();
        self.update_ui();
    }

    pub(super) fn on_live(&mut self, ev: LiveEvent) {
        let Some(id) = self.live_dialog.as_ref().map(|d| d.id) else {
            return;
        };
        match ev {
            LiveEvent::Connected { .. } => {}
            LiveEvent::State { id: i, state } if i == id => {
                if let Some(d) = &mut self.live_dialog {
                    d.ui = match state {
                        LiveState::Listening => UiState::Listening,
                        LiveState::Thinking => UiState::Thinking,
                        LiveState::Speaking => UiState::Speaking,
                    };
                }
                self.update_ui();
            }
            LiveEvent::PlaybackIdle { id: i } if i == id => {
                if self.live_dialog.as_ref().is_some_and(|d| d.closing) {
                    self.teardown_live();
                }
            }
            LiveEvent::Ended { id: i, end } if i == id => self.live_ended(end),
            // An earlier dialog's.
            _ => {}
        }
    }

    fn live_ended(&mut self, end: End) {
        match end.kind {
            // No dialog happened, or the bridge declined it: Home Assistant answers with what
            // was said meanwhile.
            EndKind::ConnectFailed | EndKind::Handoff => {
                if end.kind == EndKind::ConnectFailed {
                    warn!("live: {}, using the Home Assistant pipeline", end.reason);
                } else {
                    info!("live: {}", end.reason);
                }
                self.teardown_live();
                if !self.mic_muted() && self.phase == Phase::Idle {
                    self.start_pipeline("", false, end.unsent);
                }
            }
            // The bridge ended it (or the link broke): what it said last may still be playing.
            EndKind::Remote | EndKind::Failed | EndKind::Local => {
                if end.kind == EndKind::Failed {
                    warn!("live: {}", end.reason);
                }
                if let Some(d) = &mut self.live_dialog {
                    d.closing = true;
                }
                self.mic
                    .set_streaming(self.phase == Phase::Listening || self.arbitrating);
                self.timers.cancel(Tk::LiveMax);
                if end.playing {
                    self.timers.set(Tk::LiveTail, LIVE_TAIL_MAX_MS);
                    self.update_ui();
                } else {
                    self.teardown_live();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::json_str;

    #[test]
    fn json_strings_are_escaped() {
        assert_eq!(json_str("alisa"), r#""alisa""#);
        assert_eq!(json_str("a\"b\\c\n"), r#""a\"b\\c\u000a""#);
        assert_eq!(json_str("Алиса"), "\"Алиса\"");
    }
}
