//! Live (speech-to-speech) dialogs over a plain TCP link to a bridge service, which talks to a
//! realtime model and, through it, to the agent's tools. The speaker is a thin client: it
//! streams the microphone (VQE-cleaned, so the speaker's own voice is already removed and the
//! bridge's voice activity detection handles barge-in), plays what comes back, and shows the
//! state the bridge reports. What a dialog is, which model answers it, or whether Home Assistant
//! should answer instead (the `R` hand-off frame) is the bridge's decision. Wire format:
//! [`wire`].
//!
//! One session at a time. `LiveHandle::start` returns its id; every event carries it, and the
//! controller ignores the ones of a session it has left. `Ended` is sent exactly once per
//! session, except after `teardown()` / a replacing `start()` (the owner dropped it).
#![deny(unsafe_code)]

mod playback;
mod session;
pub mod wire;

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;

use pulsar_core::actors;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};
use tokio::sync::oneshot;

use crate::playback::{AlsaOut, Output, Playback};
use crate::session::{Args, Cmd};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LiveState {
    Listening,
    Thinking,
    Speaking,
}

pub struct StartParams {
    /// "host:port".
    pub bridge: String,
    /// JSON hello: who is calling.
    pub hello: String,
    /// The microphone audio up to now (16 kHz mono): sent first.
    pub preroll: Vec<i16>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum EndKind {
    /// The bridge could not be reached, or the answer audio cannot be played: no live dialog
    /// happened. `unsent` is the microphone audio for the Home Assistant pipeline.
    ConnectFailed,
    /// The bridge declined the dialog (`R`): the Home Assistant pipeline answers; `unsent` is the
    /// last second and a half of microphone audio.
    Handoff,
    /// The bridge ended it (`X`, or closed the link).
    Remote,
    /// The station ended it (`LiveHandle::end`).
    Local,
    /// The link broke.
    Failed,
}

#[derive(Debug, PartialEq, Eq)]
pub struct End {
    pub kind: EndKind,
    pub reason: String,
    pub unsent: Vec<i16>,
    /// Answer audio is still queued or playing.
    pub playing: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub enum LiveEvent {
    Connected {
        id: u64,
    },
    State {
        id: u64,
        state: LiveState,
    },
    /// Everything received so far has been played (or was cut by a flush while playing).
    PlaybackIdle {
        id: u64,
    },
    Ended {
        id: u64,
        end: End,
    },
}

struct Link {
    id: u64,
    cmds: UnboundedSender<Cmd>,
    playback: Arc<Playback>,
}

struct Inner {
    /// `Model::playback_pcm`.
    pcm: &'static str,
    next_id: Cell<u64>,
    events: UnboundedSender<LiveEvent>,
    link: RefCell<Option<Link>>,
}

#[derive(Clone)]
pub struct LiveHandle {
    inner: Rc<Inner>,
}

/// The handle and the events of its sessions. `pcm`: ALSA PCM of the answers
/// (`Model::playback_pcm`), opened per session.
pub fn spawn(pcm: &'static str) -> (LiveHandle, UnboundedReceiver<LiveEvent>) {
    let (events, rx) = unbounded_channel();
    let inner = Inner {
        pcm,
        next_id: Cell::new(0),
        events,
        link: RefCell::new(None),
    };
    (
        LiveHandle {
            inner: Rc::new(inner),
        },
        rx,
    )
}

impl LiveHandle {
    /// Starts a session (ending a running one without a report) and returns its id, from 1. The
    /// connect and the audio output open in the background; audio sent meanwhile is kept.
    pub fn start(&self, p: StartParams) -> u64 {
        let pcm = self.inner.pcm;
        self.start_with(p, move || AlsaOut::open(pcm))
    }

    fn start_with<O: Output>(
        &self,
        p: StartParams,
        open: impl FnOnce() -> Result<O, String> + Send + 'static,
    ) -> u64 {
        self.teardown();
        let id = self.inner.next_id.get() + 1;
        self.inner.next_id.set(id);
        let (cmds, cmds_rx) = unbounded_channel();
        let (ready_tx, ready_rx) = oneshot::channel();
        let idle = {
            // Weak: the playback thread must not keep the session's command channel open, or a
            // teardown would never end the session task.
            let cmds = cmds.downgrade();
            let events = self.inner.events.clone();
            move || {
                if let Some(cmds) = cmds.upgrade() {
                    let _ = cmds.send(Cmd::Drained);
                }
                let _ = events.send(LiveEvent::PlaybackIdle { id });
            }
        };
        let playback = Arc::new(Playback::start(open, idle, ready_tx));
        *self.inner.link.borrow_mut() = Some(Link {
            id,
            cmds,
            playback: playback.clone(),
        });
        actors::spawn_actor(
            "live",
            session::run(Args {
                id,
                bridge: p.bridge,
                hello: p.hello,
                preroll: p.preroll,
                cmds: cmds_rx,
                events: self.inner.events.clone(),
                playback,
                ready: ready_rx,
            }),
        );
        id
    }

    /// The running session's id (also after its `Ended`, until `teardown`: the answer may still
    /// be playing).
    pub fn active(&self) -> Option<u64> {
        self.inner.link.borrow().as_ref().map(|l| l.id)
    }

    /// Microphone audio, 16 kHz mono.
    pub fn send_audio(&self, pcm: Vec<i16>) {
        if let Some(l) = &*self.inner.link.borrow() {
            let _ = l.cmds.send(Cmd::Audio(pcm));
        }
    }

    /// Cuts the answer audio at once (the wake word during a dialog).
    pub fn flush_playback(&self) {
        if let Some(l) = &*self.inner.link.borrow() {
            l.playback.flush();
        }
    }

    /// 0..1 on the answer audio (the media mute).
    pub fn set_gain(&self, g: f32) {
        if let Some(l) = &*self.inner.link.borrow() {
            l.playback.set_gain(g);
        }
    }

    /// Answer audio is queued or playing.
    pub fn playing(&self) -> bool {
        self.inner
            .link
            .borrow()
            .as_ref()
            .is_some_and(|l| l.playback.playing())
    }

    /// Ends the dialog: the answer audio is cut, the bridge is told, `Ended` (Local) follows.
    pub fn end(&self, reason: &str) {
        if let Some(l) = &*self.inner.link.borrow() {
            l.playback.flush();
            let _ = l.cmds.send(Cmd::End(reason.to_owned()));
        }
    }

    /// Drops the session: the link closes, the audio output closes. No `Ended` follows.
    pub fn teardown(&self) {
        self.inner.link.borrow_mut().take();
    }
}

#[cfg(test)]
mod tests;
