//! One dialog's link to the bridge: connect (the microphone audio of the first moments is kept),
//! hello, then microphone audio out and answer audio, flushes, UI states and the end in. Runs as
//! a task on the runtime thread; every write is batched from whatever the controller queued since
//! the last one (16 ms capture chunks would otherwise be 60 small packets a second).

use std::io;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use log::{debug, info};
use socket2::{SockRef, TcpKeepalive};
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::net::{TcpStream, lookup_host};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::sync::oneshot;
use tokio::time::timeout;

use crate::playback::Playback;
use crate::wire::{self, Decoder, Frame};
use crate::{End, EndKind, LiveEvent, LiveState};

/// A bridge on the LAN answers at once; a longer wait is a bridge that is down.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(3);
/// ~2 min of microphone audio.
const MAX_BACKLOG: usize = 4 << 20;
/// Microphone audio kept while connecting (the HA pipeline takes it over if the bridge is not
/// there), and the audio kept after it for a hand-off: 16 kHz, 3 s.
const KEEP_SAMPLES: usize = 16_000 * 3;
/// The last second and a half is what a hand-off replays.
const TAIL_SAMPLES: usize = 16_000 * 3 / 2;
/// A silent peer (power cut, Wi-Fi gone) is noticed after about 25 s.
const KEEPALIVE_IDLE: Duration = Duration::from_secs(10);
const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(5);
const KEEPALIVE_RETRIES: u32 = 3;
/// The goodbye is best effort.
const GOODBYE_WAIT: Duration = Duration::from_millis(300);
const BATCH: usize = 64;

pub(crate) enum Cmd {
    Audio(Vec<i16>),
    End(String),
    /// Everything received so far has been played.
    Drained,
}

pub(crate) struct Args {
    pub id: u64,
    pub bridge: String,
    pub hello: String,
    pub preroll: Vec<i16>,
    pub cmds: UnboundedReceiver<Cmd>,
    pub events: UnboundedSender<LiveEvent>,
    pub playback: Arc<Playback>,
    pub ready: oneshot::Receiver<Result<(), String>>,
}

async fn connect(bridge: &str) -> io::Result<TcpStream> {
    let addr = match bridge.parse::<SocketAddr>() {
        Ok(a) => a,
        Err(_) => lookup_host(bridge)
            .await?
            .find(SocketAddr::is_ipv4)
            .ok_or_else(|| io::Error::other(format!("cannot resolve '{bridge}'")))?,
    };
    let s = TcpStream::connect(addr).await?;
    s.set_nodelay(true)?;
    let ka = TcpKeepalive::new()
        .with_time(KEEPALIVE_IDLE)
        .with_interval(KEEPALIVE_INTERVAL)
        .with_retries(KEEPALIVE_RETRIES);
    SockRef::from(&s).set_tcp_keepalive(&ka)?;
    Ok(s)
}

/// The most recent microphone audio, bounded; what a hand-off replays.
#[derive(Default)]
struct Tail(Vec<i16>);

impl Tail {
    fn push(&mut self, pcm: &[i16]) {
        self.0.extend_from_slice(pcm);
        // Trimmed in steps: one drain per ~1.5 s, not per chunk.
        if self.0.len() > TAIL_SAMPLES * 2 {
            let cut = self.0.len() - TAIL_SAMPLES;
            self.0.drain(..cut);
        }
    }

    fn take(&mut self) -> Vec<i16> {
        let start = self.0.len().saturating_sub(TAIL_SAMPLES);
        self.0.split_off(start)
    }
}

fn keep_last(v: &mut Vec<i16>, add: Vec<i16>) {
    if v.is_empty() {
        *v = add;
    } else {
        v.extend_from_slice(&add);
    }
    if v.len() > KEEP_SAMPLES {
        let cut = v.len() - KEEP_SAMPLES;
        v.drain(..cut);
    }
}

fn state_of(s: &str) -> LiveState {
    match s {
        "speaking" => LiveState::Speaking,
        "thinking" => LiveState::Thinking,
        _ => LiveState::Listening,
    }
}

pub(crate) async fn run(a: Args) {
    let Args {
        id,
        bridge,
        hello,
        preroll,
        mut cmds,
        events,
        playback,
        ready,
    } = a;
    let ended = |kind: EndKind, reason: String, unsent: Vec<i16>| {
        info!("live: session ends ({reason})");
        let _ = events.send(LiveEvent::Ended {
            id,
            end: End {
                kind,
                reason,
                unsent,
                playing: playback.playing(),
            },
        });
    };

    info!("live: connecting to {bridge}");
    let mut early = preroll;
    let connecting = async {
        timeout(CONNECT_TIMEOUT, connect(&bridge))
            .await
            .unwrap_or_else(|_| Err(io::ErrorKind::TimedOut.into()))
    };
    tokio::pin!(connecting);
    let stream = loop {
        tokio::select! {
            r = &mut connecting => match r {
                Ok(s) => break s,
                Err(e) => {
                    return ended(
                        EndKind::ConnectFailed,
                        format!("bridge connect failed: {e}"),
                        early,
                    );
                }
            },
            cmd = cmds.recv() => match cmd {
                Some(Cmd::Audio(pcm)) => keep_last(&mut early, pcm),
                Some(Cmd::End(why)) => return ended(EndKind::Local, why, early),
                Some(Cmd::Drained) => {}
                None => return,
            },
        }
    };
    match ready.await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            return ended(EndKind::ConnectFailed, format!("audio output: {e}"), early);
        }
        Err(_) => {
            return ended(
                EndKind::ConnectFailed,
                "audio output: playback thread did not start".into(),
                early,
            );
        }
    }
    info!("live: connected to {bridge}");
    let _ = events.send(LiveEvent::Connected { id });

    let (mut rd, mut wr) = stream.into_split();
    let mut out: Vec<u8> = Vec::new();
    let mut off = 0;
    let mut tail = Tail::default();
    wire::put_frame(&mut out, wire::HELLO, hello.as_bytes());
    wire::put_audio(&mut out, &early);
    tail.push(&early);
    drop(early);

    let mut dec = Decoder::default();
    let mut rbuf = vec![0u8; 16 << 10];
    let mut batch: Vec<Cmd> = Vec::with_capacity(BATCH);
    let (kind, reason, unsent) = 'link: loop {
        tokio::select! {
            r = rd.read(&mut rbuf) => match r {
                Ok(0) => {
                    break 'link (
                        EndKind::Remote,
                        "bridge closed the connection".to_owned(),
                        Vec::new(),
                    );
                }
                Ok(n) => {
                    dec.push(&rbuf[..n]);
                    loop {
                        match dec.pull() {
                            Ok(Some(f)) => {
                                if let Some(end) = on_frame(f, &playback, &events, id, &mut tail) {
                                    break 'link end;
                                }
                            }
                            Ok(None) => break,
                            Err(wire::BadFrame(len)) => {
                                break 'link (
                                    EndKind::Failed,
                                    format!("bad frame from bridge ({len} bytes announced)"),
                                    Vec::new(),
                                );
                            }
                        }
                    }
                }
                Err(e) => {
                    break 'link (
                        EndKind::Failed,
                        format!("bridge receive failed: {e}"),
                        Vec::new(),
                    );
                }
            },
            n = cmds.recv_many(&mut batch, BATCH) => {
                if n == 0 {
                    return; // the handle was torn down: no report
                }
                for cmd in batch.drain(..) {
                    match cmd {
                        Cmd::Audio(pcm) => {
                            wire::put_audio(&mut out, &pcm);
                            tail.push(&pcm);
                        }
                        Cmd::Drained => wire::put_frame(&mut out, wire::DRAINED, &[]),
                        Cmd::End(why) => {
                            wire::put_frame(&mut out, wire::END, why.as_bytes());
                            let _ = timeout(GOODBYE_WAIT, wr.write_all(&out[off..])).await;
                            break 'link (EndKind::Local, why, Vec::new());
                        }
                    }
                }
            },
            w = wr.write(&out[off..]), if off < out.len() => match w {
                Ok(n) => {
                    off += n;
                    if off == out.len() {
                        out.clear();
                        off = 0;
                    } else if off > (64 << 10) {
                        out.drain(..off);
                        off = 0;
                    }
                }
                Err(e) => {
                    break 'link (
                        EndKind::Failed,
                        format!("bridge send failed: {e}"),
                        Vec::new(),
                    );
                }
            },
        }
        if out.len() - off > MAX_BACKLOG {
            break 'link (
                EndKind::Failed,
                "bridge too slow (send backlog)".to_owned(),
                Vec::new(),
            );
        }
    };
    ended(kind, reason, unsent)
}

/// A frame from the bridge; Some = the session is over.
fn on_frame(
    f: Frame,
    playback: &Playback,
    events: &UnboundedSender<LiveEvent>,
    id: u64,
    tail: &mut Tail,
) -> Option<(EndKind, String, Vec<i16>)> {
    match f {
        Frame::Audio(pcm) => playback.push(pcm),
        Frame::Flush => playback.flush(),
        Frame::State(s) => {
            let _ = events.send(LiveEvent::State {
                id,
                state: state_of(&s),
            });
        }
        Frame::End(why) => return Some((EndKind::Remote, format!("bridge: {why}"), Vec::new())),
        Frame::Handoff => {
            return Some((
                EndKind::Handoff,
                "bridge handed the dialog to Home Assistant".to_owned(),
                tail.take(),
            ));
        }
        Frame::Unknown(t) => {
            debug!("live: unknown frame '{}' from the bridge", char::from(t));
        }
    }
    None
}
