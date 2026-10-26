//! Sessions against a bridge on loopback and a fake audio device.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::task::LocalSet;
use tokio::time::timeout;

use crate::playback::fake::{Dev, Fake, lock_dev};
use crate::wire::{self, Decoder, Frame};
use crate::{EndKind, LiveEvent, LiveHandle, LiveState, StartParams, spawn};

const T: Duration = Duration::from_secs(3);

fn params(bridge: String, preroll: Vec<i16>) -> StartParams {
    StartParams {
        bridge,
        hello: r#"{"wake_word":"alisa","protocol":1}"#.into(),
        preroll,
    }
}

fn fake_dev() -> (
    Arc<Mutex<Dev>>,
    impl FnOnce() -> Result<Fake, String> + Send + 'static,
) {
    let dev = Arc::new(Mutex::new(Dev::default()));
    let d = dev.clone();
    (dev, move || Ok(Fake(d)))
}

async fn next(rx: &mut UnboundedReceiver<LiveEvent>) -> LiveEvent {
    timeout(T, rx.recv()).await.expect("no event").unwrap()
}

/// What the bridge side reads, frame by frame.
struct Peer {
    s: TcpStream,
    dec: Decoder,
    /// Raw type bytes and payloads of station frames.
    got: Vec<(u8, Vec<u8>)>,
}

impl Peer {
    /// Reads until `f` is satisfied by the frames received so far.
    async fn until(&mut self, f: impl Fn(&[(u8, Vec<u8>)]) -> bool) {
        let mut buf = [0u8; 4096];
        let mut raw: Vec<u8> = Vec::new();
        while !f(&self.got) {
            let n = timeout(T, self.s.read(&mut buf))
                .await
                .expect("bridge read timed out")
                .unwrap();
            assert!(n > 0, "station closed the link");
            raw.extend_from_slice(&buf[..n]);
            while raw.len() >= 5 {
                let len = u32::from_le_bytes([raw[1], raw[2], raw[3], raw[4]]) as usize;
                if raw.len() < 5 + len {
                    break;
                }
                self.got.push((raw[0], raw[5..5 + len].to_vec()));
                raw.drain(..5 + len);
            }
        }
        let _ = &self.dec;
    }

    async fn send(&mut self, ty: u8, payload: &[u8]) {
        let mut out = Vec::new();
        wire::put_frame(&mut out, ty, payload);
        self.s.write_all(&out).await.unwrap();
    }

    fn mic(&self) -> Vec<i16> {
        self.got
            .iter()
            .filter(|(t, _)| *t == b'A')
            .flat_map(|(_, p)| p.as_chunks::<2>().0.iter().map(|c| i16::from_le_bytes(*c)))
            .collect()
    }
}

async fn bridge() -> (TcpListener, String) {
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap().to_string();
    (l, addr)
}

async fn accept(l: &TcpListener) -> Peer {
    let (s, _) = timeout(T, l.accept())
        .await
        .expect("no connection")
        .unwrap();
    Peer {
        s,
        dec: Decoder::default(),
        got: Vec::new(),
    }
}

fn local<F: std::future::Future<Output = ()>>(f: F) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let set = LocalSet::new();
    rt.block_on(set.run_until(f));
}

fn start(h: &LiveHandle, bridge: String, preroll: Vec<i16>) -> (u64, Arc<Mutex<Dev>>) {
    let (dev, open) = fake_dev();
    (h.start_with(params(bridge, preroll), open), dev)
}

#[test]
fn a_dialog_streams_the_microphone_and_plays_the_answer() {
    local(async {
        let (h, mut rx) = spawn("test");
        let (l, addr) = bridge().await;
        let (id, dev) = start(&h, addr, vec![1, 2, 3]);
        let mut peer = accept(&l).await;
        assert_eq!(next(&mut rx).await, LiveEvent::Connected { id });
        h.send_audio(vec![4, 5]);
        peer.until(|g| g.iter().any(|(t, _)| *t == b'A')).await;
        peer.until(|g| g.iter().filter(|(t, _)| *t == b'A').count() >= 1 && g.len() >= 2)
            .await;
        // Hello first, then the preroll, then what the controller sent.
        assert_eq!(peer.got[0].0, b'H');
        assert_eq!(
            peer.got[0].1,
            br#"{"wake_word":"alisa","protocol":1}"#.to_vec()
        );
        h.send_audio(vec![6]);
        peer.until(|g| g.iter().flat_map(|(_, p)| p.iter()).count() > 40)
            .await;
        let mic = peer.mic();
        assert_eq!(&mic[..3], [1, 2, 3]);

        peer.send(b'S', b"thinking").await;
        assert_eq!(
            next(&mut rx).await,
            LiveEvent::State {
                id,
                state: LiveState::Thinking
            }
        );
        let mut audio = Vec::new();
        wire::put_audio(&mut audio, &[100; 480]);
        peer.s.write_all(&audio).await.unwrap();
        // The answer reached the device and its end was reported to both sides.
        assert_eq!(next(&mut rx).await, LiveEvent::PlaybackIdle { id });
        assert_eq!(lock_dev(&dev).written, vec![100; 480]);
        peer.until(|g| g.iter().any(|(t, _)| *t == b'D')).await;

        peer.send(b'X', b"goodbye").await;
        match next(&mut rx).await {
            LiveEvent::Ended { id: i, end } => {
                assert_eq!(i, id);
                assert_eq!(end.kind, EndKind::Remote);
                assert_eq!(end.reason, "bridge: goodbye");
                assert!(!end.playing);
            }
            other => panic!("{other:?}"),
        }
    });
}

#[test]
fn an_unreachable_bridge_returns_the_audio_for_home_assistant() {
    local(async {
        let (h, mut rx) = spawn("test");
        // A port nothing listens on.
        let (l, addr) = bridge().await;
        drop(l);
        let (id, _dev) = start(&h, addr, vec![9; 10]);
        h.send_audio(vec![8; 5]);
        match next(&mut rx).await {
            LiveEvent::Ended { id: i, end } => {
                assert_eq!(i, id);
                assert_eq!(end.kind, EndKind::ConnectFailed);
                assert!(
                    end.reason.starts_with("bridge connect failed"),
                    "{}",
                    end.reason
                );
                // The preroll for sure; the audio sent while connecting if it was queued
                // before the refusal arrived.
                assert_eq!(&end.unsent[..10], [9; 10]);
            }
            other => panic!("{other:?}"),
        }
    });
}

#[test]
fn a_failed_audio_output_is_a_failed_start() {
    local(async {
        let (h, mut rx) = spawn("test");
        let (l, addr) = bridge().await;
        let id = h.start_with(params(addr, vec![7; 4]), || {
            Err::<Fake, _>("no such pcm".to_owned())
        });
        let _peer = accept(&l).await;
        match next(&mut rx).await {
            LiveEvent::Ended { id: i, end } => {
                assert_eq!(i, id);
                assert_eq!(end.kind, EndKind::ConnectFailed);
                assert_eq!(end.reason, "audio output: no such pcm");
                assert_eq!(end.unsent, [7; 4]);
            }
            other => panic!("{other:?}"),
        }
    });
}

#[test]
fn a_hand_off_replays_the_recent_microphone_audio() {
    local(async {
        let (h, mut rx) = spawn("test");
        let (l, addr) = bridge().await;
        let (id, _dev) = start(&h, addr, vec![1; 8000]);
        let mut peer = accept(&l).await;
        assert_eq!(next(&mut rx).await, LiveEvent::Connected { id });
        h.send_audio(vec![2; 1000]);
        peer.until(|g| g.iter().flat_map(|(_, p)| p.iter()).count() > 18_000)
            .await;
        peer.send(b'R', b"").await;
        match next(&mut rx).await {
            LiveEvent::Ended { end, .. } => {
                assert_eq!(end.kind, EndKind::Handoff);
                let mut want = vec![1; 8000];
                want.extend(vec![2; 1000]);
                assert_eq!(end.unsent, want);
            }
            other => panic!("{other:?}"),
        }
    });
}

#[test]
fn ending_tells_the_bridge_and_cuts_the_answer() {
    local(async {
        let (h, mut rx) = spawn("test");
        let (l, addr) = bridge().await;
        let (id, dev) = start(&h, addr, Vec::new());
        let mut peer = accept(&l).await;
        assert_eq!(next(&mut rx).await, LiveEvent::Connected { id });
        lock_dev(&dev).room = Some(10);
        let mut audio = Vec::new();
        wire::put_audio(&mut audio, &[5; 24_000]);
        peer.s.write_all(&audio).await.unwrap();
        peer.until(|g| g.iter().any(|(t, _)| *t == b'H')).await;
        while !h.playing() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        h.end("button");
        peer.until(|g| g.iter().any(|(t, _)| *t == b'E')).await;
        let (_, why) = peer.got.iter().find(|(t, _)| *t == b'E').unwrap();
        assert_eq!(why, b"button");
        loop {
            match next(&mut rx).await {
                LiveEvent::Ended { end, .. } => {
                    assert_eq!(end.kind, EndKind::Local);
                    assert_eq!(end.reason, "button");
                    break;
                }
                LiveEvent::PlaybackIdle { .. } => {}
                other => panic!("{other:?}"),
            }
        }
        assert!(!h.playing());
    });
}

#[test]
fn the_bridge_closing_the_link_ends_the_dialog() {
    local(async {
        let (h, mut rx) = spawn("test");
        let (l, addr) = bridge().await;
        let (id, _dev) = start(&h, addr, Vec::new());
        let mut peer = accept(&l).await;
        assert_eq!(next(&mut rx).await, LiveEvent::Connected { id });
        // Read what was sent first: closing with unread data would reset the connection.
        peer.until(|g| g.iter().any(|(t, _)| *t == b'H')).await;
        drop(peer);
        match next(&mut rx).await {
            LiveEvent::Ended { end, .. } => assert_eq!(end.kind, EndKind::Remote),
            other => panic!("{other:?}"),
        }
    });
}

#[test]
fn a_flush_frame_cuts_what_is_playing() {
    local(async {
        let (h, mut rx) = spawn("test");
        let (l, addr) = bridge().await;
        let (id, dev) = start(&h, addr, Vec::new());
        let mut peer = accept(&l).await;
        assert_eq!(next(&mut rx).await, LiveEvent::Connected { id });
        lock_dev(&dev).room = Some(10);
        let mut audio = Vec::new();
        wire::put_audio(&mut audio, &[5; 24_000]);
        peer.s.write_all(&audio).await.unwrap();
        while !h.playing() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        tokio::time::sleep(Duration::from_millis(30)).await;
        peer.send(b'F', b"").await;
        assert_eq!(next(&mut rx).await, LiveEvent::PlaybackIdle { id });
        assert!(lock_dev(&dev).written.len() < 24_000);
        peer.until(|g| g.iter().any(|(t, _)| *t == b'D')).await;
    });
}

#[test]
fn teardown_closes_the_link_without_a_report() {
    local(async {
        let (h, mut rx) = spawn("test");
        let (l, addr) = bridge().await;
        let (id, _dev) = start(&h, addr, Vec::new());
        let mut peer = accept(&l).await;
        assert_eq!(next(&mut rx).await, LiveEvent::Connected { id });
        assert_eq!(h.active(), Some(id));
        h.teardown();
        assert_eq!(h.active(), None);
        // The station side of the socket is gone.
        let mut buf = [0u8; 256];
        loop {
            match timeout(T, peer.s.read(&mut buf))
                .await
                .expect("link still open")
            {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
        }
        assert!(rx.try_recv().is_err());
    });
}

#[test]
fn frames_are_decoded_by_the_same_rules_as_the_station_writes_them() {
    // The decoder is for the bridge's frames; the station's own (H, A, D, E) are unknown types
    // to it, which a loopback test of both directions would otherwise trip over.
    let mut d = Decoder::default();
    let mut out = Vec::new();
    wire::put_frame(&mut out, wire::END, b"x");
    d.push(&out);
    assert_eq!(d.pull().unwrap(), Some(Frame::Unknown(b'E')));
}
