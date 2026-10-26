//! Answer audio of a live dialog: its own ALSA stream on the model's playback PCM (softvol + plug +
//! dmix, so the master volume applies and it mixes with music and earcons), 24 kHz mono. A thread
//! plays from a queue that `flush` empties at once (barge-in), applies the mute gain and reports
//! when everything received so far has been played (`on_idle`), which the bridge needs to know
//! when the owner may be answered and the controller to close a finished session.
//!
//! The stream starts with the first written frame (start threshold 1): the device is only open
//! while a dialog lasts and a short tail would otherwise never reach the start threshold. Gaps
//! between sentences longer than the 125 ms buffer underrun and restart on the next write.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::SeqCst};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread;
use std::time::Duration;

use log::{info, warn};
use pulsar_alsa::{Format, HwRequest, Pcm, Stream};
use tokio::sync::oneshot;

pub const RATE: u32 = 24_000;
const PERIOD_FRAMES: u64 = RATE as u64 / 50; // 20 ms
const BUFFER_FRAMES: u64 = RATE as u64 / 8; // 125 ms
/// Most that is queued ahead of the device (the bridge streams sentence by sentence; more is a
/// runaway peer).
const MAX_QUEUED: usize = RATE as usize * 90;
/// How often the end of playback is looked for once the queue is empty.
const DRAIN_POLL: Duration = Duration::from_millis(10);
/// Longest wait for room in the device inside one write.
const WAIT_SLICE_MS: i32 = 20;
const UNITY: u32 = 256;

/// The device the thread plays on.
pub(crate) trait Output: Send + 'static {
    /// Frames per period; the thread writes at most two at a time so a flush is heard soon.
    fn period(&self) -> usize;
    /// Writes what fits, waiting at most one slice for room; returns how many samples it took
    /// (0: still full). Err: the device is gone.
    fn write(&mut self, samples: &[i16]) -> Result<usize, String>;
    /// Samples written and not yet played; None: nothing queued (ran dry, or never started).
    fn queued(&mut self) -> Option<usize>;
    /// Drops what is queued and re-arms the device for the next stream.
    fn reset(&mut self);
}

#[derive(Default)]
struct Queue {
    chunks: VecDeque<Vec<i16>>,
    /// Offset into the first chunk.
    head: usize,
    samples: usize,
}

impl Queue {
    fn take(&mut self, max: usize, out: &mut Vec<i16>) {
        out.clear();
        while out.len() < max {
            let Some(front) = self.chunks.front() else {
                break;
            };
            let n = (front.len() - self.head).min(max - out.len());
            out.extend_from_slice(&front[self.head..self.head + n]);
            self.head += n;
            self.samples -= n;
            if self.head == front.len() {
                self.chunks.pop_front();
                self.head = 0;
            }
        }
    }

    fn clear(&mut self) {
        self.chunks.clear();
        self.head = 0;
        self.samples = 0;
    }
}

struct Shared {
    q: Mutex<Queue>,
    cv: Condvar,
    /// Q8, `UNITY` = as received.
    gain: AtomicU32,
    /// Something is queued or still playing.
    busy: AtomicBool,
    /// Counts flushes; the write loop looks at it without the lock.
    flushes: AtomicU64,
    stop: AtomicBool,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

pub(crate) struct Playback {
    shared: Arc<Shared>,
}

impl Playback {
    /// Starts the thread. It opens the device itself (a dmix attach takes a few tens of ms) and
    /// reports the result through `ready`; `on_idle` runs on the thread whenever playback ran
    /// out or was flushed while it was playing.
    pub(crate) fn start<O: Output>(
        open: impl FnOnce() -> Result<O, String> + Send + 'static,
        on_idle: impl Fn() + Send + 'static,
        ready: oneshot::Sender<Result<(), String>>,
    ) -> Playback {
        let shared = Arc::new(Shared {
            q: Mutex::new(Queue::default()),
            cv: Condvar::new(),
            gain: AtomicU32::new(UNITY),
            busy: AtomicBool::new(false),
            flushes: AtomicU64::new(0),
            stop: AtomicBool::new(false),
        });
        let thread_shared = shared.clone();
        let spawned = thread::Builder::new()
            .name("live-play".into())
            .spawn(move || match open() {
                Ok(out) => {
                    let _ = ready.send(Ok(()));
                    run(&thread_shared, out, &on_idle);
                }
                Err(e) => {
                    let _ = ready.send(Err(e));
                }
            });
        if let Err(e) = spawned {
            warn!("live: cannot start the playback thread: {e}");
        }
        Playback { shared }
    }

    /// Queues answer audio.
    pub(crate) fn push(&self, pcm: Vec<i16>) {
        if pcm.is_empty() {
            return;
        }
        let mut q = lock(&self.shared.q);
        if q.samples + pcm.len() > MAX_QUEUED {
            warn!("live: playback queue full, answer audio dropped");
            return;
        }
        q.samples += pcm.len();
        q.chunks.push_back(pcm);
        self.shared.busy.store(true, SeqCst);
        drop(q);
        self.shared.cv.notify_all();
    }

    /// Drops what is queued and cuts what plays.
    pub(crate) fn flush(&self) {
        let mut q = lock(&self.shared.q);
        q.clear();
        self.shared.flushes.fetch_add(1, SeqCst);
        // What the device still holds is cut within a few ms: from here on nothing plays.
        self.shared.busy.store(false, SeqCst);
        drop(q);
        self.shared.cv.notify_all();
    }

    /// 0..1; 0 is silence (the media mute).
    pub(crate) fn set_gain(&self, g: f32) {
        let q8 = (g.clamp(0.0, 1.0) * UNITY as f32).round() as u32;
        self.shared.gain.store(q8, SeqCst);
    }

    /// Audio is queued or still playing (right after a `push`, even before the thread has seen
    /// it).
    pub(crate) fn playing(&self) -> bool {
        self.shared.busy.load(SeqCst)
    }
}

impl Drop for Playback {
    /// The thread is not joined: a write may be waiting for room, and the runtime thread must
    /// not. It closes the device and ends by itself.
    fn drop(&mut self) {
        self.shared.stop.store(true, SeqCst);
        // Under the lock so the wake-up cannot slip between the thread's check and its wait.
        drop(lock(&self.shared.q));
        self.shared.cv.notify_all();
    }
}

enum Step {
    Stop,
    Flush,
    Poll,
    Data,
}

fn run<O: Output>(shared: &Shared, mut out: O, on_idle: &dyn Fn()) {
    let period = out.period().max(1);
    let mut seen = shared.flushes.load(SeqCst);
    let mut active = false; // frames written that have not finished playing
    let mut failed = false;
    let mut buf: Vec<i16> = Vec::with_capacity(period * 2);
    let idle = |active: &mut bool| {
        *active = false;
        let q = lock(&shared.q);
        shared.busy.store(q.samples > 0, SeqCst);
        drop(q);
        on_idle();
    };
    loop {
        let step = {
            let mut q = lock(&shared.q);
            let mut poll = false;
            loop {
                if shared.stop.load(SeqCst) {
                    break Step::Stop;
                }
                let f = shared.flushes.load(SeqCst);
                if f != seen {
                    seen = f;
                    break Step::Flush;
                }
                if q.samples > 0 {
                    q.take(period * 2, &mut buf);
                    break Step::Data;
                }
                if active && poll {
                    break Step::Poll;
                }
                if active {
                    let (g, t) = shared
                        .cv
                        .wait_timeout(q, DRAIN_POLL)
                        .unwrap_or_else(PoisonError::into_inner);
                    q = g;
                    poll = t.timed_out();
                } else {
                    q = shared.cv.wait(q).unwrap_or_else(PoisonError::into_inner);
                }
            }
        };
        match step {
            Step::Stop => return,
            Step::Flush => {
                out.reset();
                failed = false;
                if active {
                    idle(&mut active);
                }
            }
            Step::Poll => {
                if out.queued().is_none() {
                    out.reset();
                    idle(&mut active);
                }
            }
            Step::Data => {
                active = true;
                let g = shared.gain.load(SeqCst);
                if g != UNITY {
                    for s in &mut buf {
                        *s = ((i32::from(*s) * g as i32) >> 8) as i16;
                    }
                }
                let mut off = 0;
                while off < buf.len()
                    && !shared.stop.load(SeqCst)
                    && shared.flushes.load(SeqCst) == seen
                {
                    match out.write(&buf[off..]) {
                        Ok(n) => off += n,
                        Err(e) => {
                            if !failed {
                                warn!("live: playback failed: {e}");
                            }
                            failed = true;
                            out.reset();
                            // The rest of what was queued is lost with the device.
                            lock(&shared.q).clear();
                            idle(&mut active);
                            break;
                        }
                    }
                }
            }
        }
    }
}

/// The model's playback PCM through libasound.
pub(crate) struct AlsaOut {
    pcm: Pcm,
    period: usize,
    bytes: Vec<u8>,
}

impl AlsaOut {
    pub(crate) fn open(name: &str) -> Result<AlsaOut, String> {
        let step = |what: &str, e: i32| format!("{what}: {}", pulsar_alsa::strerror(e));
        let mut pcm = Pcm::open(name, Stream::Playback, false).map_err(|e| step("open", e))?;
        let t = pcm.set_hw_params_traced(&HwRequest {
            format: Format::S16Le,
            channels: 1,
            rate: RATE,
            rate_exact: false,
            period_frames: Some(PERIOD_FRAMES),
            buffer_frames: Some(BUFFER_FRAMES),
        });
        if let Some((what, e)) = t.failed {
            return Err(step(what, e));
        }
        if t.actual.rate != RATE {
            warn!("live: ALSA rate {} instead of {RATE}", t.actual.rate);
        }
        let period = t.actual.period_frames.max(1);
        pcm.set_sw_params(1, period)
            .map_err(|e| step("sw_params", e))?;
        pcm.prepare().map_err(|e| step("prepare", e))?;
        info!(
            "live: ALSA '{name}' open {} Hz mono (period {period}, buffer {} frames)",
            t.actual.rate, t.actual.buffer_frames
        );
        Ok(AlsaOut {
            pcm,
            period: period as usize,
            bytes: Vec::new(),
        })
    }

    fn recover(&mut self, err: i32) -> Result<(), String> {
        if err == -libc::EINTR || err == -libc::EAGAIN {
            return Ok(());
        }
        self.pcm.recover(err, true).map_err(pulsar_alsa::strerror)
    }
}

impl Output for AlsaOut {
    fn period(&self) -> usize {
        self.period
    }

    fn write(&mut self, samples: &[i16]) -> Result<usize, String> {
        for _ in 0..2 {
            let avail = match self.pcm.avail_update() {
                Ok(n) => n,
                Err(e) => {
                    self.recover(e)?;
                    continue;
                }
            };
            if avail == 0 {
                if let Err(e) = self.pcm.wait(WAIT_SLICE_MS) {
                    self.recover(e)?;
                }
                continue;
            }
            let n = avail.min(samples.len());
            self.bytes.clear();
            for s in &samples[..n] {
                self.bytes.extend_from_slice(&s.to_le_bytes());
            }
            match self.pcm.writei(&self.bytes, n) {
                Ok(w) => return Ok(w),
                Err(e) => self.recover(e)?,
            }
        }
        Ok(0)
    }

    fn queued(&mut self) -> Option<usize> {
        match self.pcm.delay() {
            Ok(d) if d > 0 => Some(d as usize),
            _ => None,
        }
    }

    fn reset(&mut self) {
        let _ = self.pcm.drop_frames();
        let _ = self.pcm.prepare();
    }
}

#[cfg(test)]
pub(crate) mod fake {
    use super::*;

    #[derive(Default)]
    pub(crate) struct Dev {
        pub written: Vec<i16>,
        pub resets: u32,
        /// `queued` reports audio this many more times before the device runs dry.
        pub playing_polls: u32,
        /// Room per write (None: all).
        pub room: Option<usize>,
        pub fail_next: bool,
    }

    pub(crate) struct Fake(pub Arc<Mutex<Dev>>);

    impl Output for Fake {
        fn period(&self) -> usize {
            480
        }
        fn write(&mut self, samples: &[i16]) -> Result<usize, String> {
            let mut d = lock(&self.0);
            if std::mem::take(&mut d.fail_next) {
                return Err("gone".into());
            }
            let n = d.room.unwrap_or(samples.len()).min(samples.len());
            d.written.extend_from_slice(&samples[..n]);
            d.playing_polls = 3;
            let slow = d.room.is_some();
            drop(d);
            if slow {
                // A device with little room takes time to accept more.
                thread::sleep(Duration::from_millis(2));
            }
            Ok(n)
        }
        fn queued(&mut self) -> Option<usize> {
            let mut d = lock(&self.0);
            if d.playing_polls == 0 {
                return None;
            }
            d.playing_polls -= 1;
            Some(100)
        }
        fn reset(&mut self) {
            let mut d = lock(&self.0);
            d.resets += 1;
            d.playing_polls = 0;
        }
    }

    pub(crate) fn lock_dev(d: &Mutex<Dev>) -> MutexGuard<'_, Dev> {
        lock(d)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc::{Receiver, channel};

    use super::fake::{Dev, Fake};
    use super::*;

    fn start() -> (Playback, Arc<Mutex<Dev>>, Receiver<()>) {
        let dev = Arc::new(Mutex::new(Dev::default()));
        let (idle_tx, idle_rx) = channel();
        let (ready_tx, mut ready_rx) = oneshot::channel();
        let d = dev.clone();
        let p = Playback::start(
            move || Ok(Fake(d)),
            move || {
                let _ = idle_tx.send(());
            },
            ready_tx,
        );
        // The thread answers at once; poll the oneshot without a runtime.
        let t0 = std::time::Instant::now();
        loop {
            match ready_rx.try_recv() {
                Ok(r) => {
                    assert_eq!(r, Ok(()));
                    break;
                }
                Err(_) if t0.elapsed() < Duration::from_secs(2) => {
                    thread::sleep(Duration::from_millis(1));
                }
                Err(e) => panic!("device never opened: {e}"),
            }
        }
        (p, dev, idle_rx)
    }

    const T: Duration = Duration::from_secs(2);

    #[test]
    fn plays_in_order_and_reports_the_end_once() {
        let (p, dev, idle) = start();
        p.push((0..1000).collect());
        p.push((1000..1500).collect());
        assert!(p.playing());
        idle.recv_timeout(T).unwrap();
        assert!(!p.playing());
        assert_eq!(lock(&dev).written, (0..1500).collect::<Vec<i16>>());
        assert!(idle.recv_timeout(Duration::from_millis(100)).is_err());
        // The device was re-armed for the next stream.
        assert_eq!(lock(&dev).resets, 1);
    }

    #[test]
    fn a_new_stream_after_the_end_plays_and_reports_again() {
        let (p, dev, idle) = start();
        p.push(vec![1; 100]);
        idle.recv_timeout(T).unwrap();
        p.push(vec![2; 100]);
        idle.recv_timeout(T).unwrap();
        let w = lock(&dev).written.clone();
        assert_eq!((w.len(), w[0], w[199]), (200, 1, 2));
    }

    #[test]
    fn gain_scales_what_is_written() {
        let (p, dev, idle) = start();
        p.set_gain(0.5);
        p.push(vec![1000, -1000, 7]);
        idle.recv_timeout(T).unwrap();
        assert_eq!(lock(&dev).written, [500, -500, 3]);
        p.set_gain(0.0);
        p.push(vec![1000; 10]);
        idle.recv_timeout(T).unwrap();
        assert!(lock(&dev).written[3..].iter().all(|&s| s == 0));
    }

    #[test]
    fn flush_drops_the_queue_and_reports_the_cut_playback() {
        let (p, dev, idle) = start();
        // A device that takes 100 samples per write: most of this is still queued.
        lock(&dev).room = Some(100);
        p.push(vec![5; 20_000]);
        thread::sleep(Duration::from_millis(30));
        p.flush();
        idle.recv_timeout(T).unwrap();
        assert!(!p.playing());
        let n = lock(&dev).written.len();
        assert!(n < 20_000, "{n} samples reached the device");
        thread::sleep(Duration::from_millis(50));
        assert_eq!(
            lock(&dev).written.len(),
            n,
            "nothing is written after the flush"
        );
    }

    #[test]
    fn flush_while_silent_reports_nothing() {
        let (p, _dev, idle) = start();
        p.flush();
        assert!(idle.recv_timeout(Duration::from_millis(100)).is_err());
        p.push(vec![1; 10]);
        idle.recv_timeout(T).unwrap();
    }

    #[test]
    fn a_lost_device_drops_the_answer_and_goes_idle() {
        let (p, dev, idle) = start();
        lock(&dev).fail_next = true;
        p.push(vec![1; 5000]);
        idle.recv_timeout(T).unwrap();
        assert!(!p.playing());
        assert!(lock(&dev).written.is_empty());
    }

    #[test]
    fn dropping_the_handle_ends_the_thread_and_its_device() {
        let (p, dev, _idle) = start();
        assert_eq!(Arc::strong_count(&dev), 2);
        drop(p);
        let t0 = std::time::Instant::now();
        while Arc::strong_count(&dev) > 1 {
            assert!(t0.elapsed() < T, "the playback thread did not end");
            thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn a_failed_open_is_reported() {
        let (ready_tx, mut ready_rx) = oneshot::channel();
        let _p = Playback::start(|| Err::<Fake, _>("no such pcm".to_owned()), || {}, ready_tx);
        let t0 = std::time::Instant::now();
        loop {
            if let Ok(r) = ready_rx.try_recv() {
                assert_eq!(r, Err("no such pcm".to_owned()));
                break;
            }
            assert!(t0.elapsed() < T);
            thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn the_queue_is_bounded() {
        let (p, dev, idle) = start();
        lock(&dev).room = Some(1);
        p.push(vec![1; MAX_QUEUED - 10]);
        p.push(vec![2; 100]); // over the limit: dropped
        thread::sleep(Duration::from_millis(30));
        p.flush();
        idle.recv_timeout(T).unwrap();
        assert!(!lock(&dev).written.contains(&2));
    }
}
