//! Framing of the bridge link, both ways: `[u8 type][u32 little-endian length][payload]`.
//!
//! station → bridge: `H` hello (JSON), `A` microphone audio (s16le 16 kHz mono, VQE-cleaned),
//! `D` playback drained (everything received so far has been played), `E` end (payload: reason).
//! bridge → station: `A` answer audio (s16le 24 kHz mono), `F` flush (the model was interrupted:
//! drop what is queued), `S` UI state (`listening` | `thinking` | `speaking`), `X` end of the
//! session (payload: reason), `R` hand-off (the bridge declines the dialog: the station runs the
//! Home Assistant pipeline instead; sent right after the hello, so only the first moments of the
//! request were streamed and the station replays them).

/// Larger frames are a broken peer, not audio: the C++ build refused them too.
pub const MAX_FRAME: usize = 8 << 20;

pub const HELLO: u8 = b'H';
pub const AUDIO: u8 = b'A';
pub const DRAINED: u8 = b'D';
pub const END: u8 = b'E';
const FLUSH: u8 = b'F';
const STATE: u8 = b'S';
const REMOTE_END: u8 = b'X';
const HANDOFF: u8 = b'R';

const HEADER: usize = 5;

/// What the bridge sends.
#[derive(Debug, PartialEq, Eq)]
pub enum Frame {
    /// 24 kHz mono samples.
    Audio(Vec<i16>),
    Flush,
    State(String),
    End(String),
    Handoff,
    /// A type this build does not know (a newer bridge): skipped.
    Unknown(u8),
}

#[derive(Debug, PartialEq, Eq)]
pub struct BadFrame(pub usize);

fn header(out: &mut Vec<u8>, ty: u8, len: usize) {
    out.push(ty);
    // Frames are bounded by MAX_FRAME on the way in; ours are a hello, a reason or one
    // microphone chunk.
    out.extend_from_slice(&(len as u32).to_le_bytes());
}

pub fn put_frame(out: &mut Vec<u8>, ty: u8, payload: &[u8]) {
    header(out, ty, payload.len());
    out.extend_from_slice(payload);
}

pub fn put_audio(out: &mut Vec<u8>, pcm: &[i16]) {
    header(out, AUDIO, pcm.len() * 2);
    out.reserve(pcm.len() * 2);
    for s in pcm {
        out.extend_from_slice(&s.to_le_bytes());
    }
}

/// Reassembles frames from the byte stream. Consumed bytes are dropped lazily so a burst of small
/// frames does not shift the buffer once per frame.
#[derive(Default)]
pub struct Decoder {
    buf: Vec<u8>,
    off: usize,
}

impl Decoder {
    pub fn push(&mut self, data: &[u8]) {
        if self.off > 0 && self.off == self.buf.len() {
            self.buf.clear();
            self.off = 0;
        } else if self.off > (64 << 10) {
            self.buf.drain(..self.off);
            self.off = 0;
        }
        self.buf.extend_from_slice(data);
    }

    /// The next complete frame; `Ok(None)`: more bytes needed. `Err`: the peer announced a frame
    /// above `MAX_FRAME` (its length); the stream is not usable any more.
    pub fn pull(&mut self) -> Result<Option<Frame>, BadFrame> {
        let b = &self.buf[self.off..];
        if b.len() < HEADER {
            return Ok(None);
        }
        let len = u32::from_le_bytes([b[1], b[2], b[3], b[4]]) as usize;
        if len > MAX_FRAME {
            return Err(BadFrame(len));
        }
        if b.len() < HEADER + len {
            return Ok(None);
        }
        let payload = &b[HEADER..HEADER + len];
        let frame = match b[0] {
            AUDIO => Frame::Audio(
                payload
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| i16::from_le_bytes(*c))
                    .collect(),
            ),
            FLUSH => Frame::Flush,
            STATE => Frame::State(String::from_utf8_lossy(payload).into_owned()),
            REMOTE_END => Frame::End(String::from_utf8_lossy(payload).into_owned()),
            HANDOFF => Frame::Handoff,
            other => Frame::Unknown(other),
        };
        self.off += HEADER + len;
        Ok(Some(frame))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frames(d: &mut Decoder) -> Vec<Frame> {
        std::iter::from_fn(|| d.pull().unwrap()).collect()
    }

    #[test]
    fn frames_have_a_little_endian_length() {
        let mut out = Vec::new();
        put_frame(&mut out, HELLO, b"{}");
        assert_eq!(out, [b'H', 2, 0, 0, 0, b'{', b'}']);
        out.clear();
        put_audio(&mut out, &[1, -2, 0x0304]);
        assert_eq!(out, [b'A', 6, 0, 0, 0, 1, 0, 0xfe, 0xff, 0x04, 0x03]);
        out.clear();
        put_frame(&mut out, DRAINED, &[]);
        assert_eq!(out, [b'D', 0, 0, 0, 0]);
    }

    #[test]
    fn decodes_every_bridge_frame_type() {
        let mut d = Decoder::default();
        let mut wire = Vec::new();
        put_audio(&mut wire, &[7, -7]);
        put_frame(&mut wire, b'F', &[]);
        put_frame(&mut wire, b'S', b"speaking");
        put_frame(&mut wire, b'Z', b"future");
        put_frame(&mut wire, b'R', b"ha");
        put_frame(&mut wire, b'X', b"goodbye");
        d.push(&wire);
        assert_eq!(
            frames(&mut d),
            [
                Frame::Audio(vec![7, -7]),
                Frame::Flush,
                Frame::State("speaking".into()),
                Frame::Unknown(b'Z'),
                Frame::Handoff,
                Frame::End("goodbye".into()),
            ]
        );
        assert_eq!(d.pull(), Ok(None));
    }

    #[test]
    fn a_frame_split_at_any_byte_is_reassembled() {
        let mut wire = Vec::new();
        put_audio(&mut wire, &[1, 2, 3, 4]);
        put_frame(&mut wire, b'S', b"thinking");
        for cut in 0..wire.len() {
            let mut d = Decoder::default();
            d.push(&wire[..cut]);
            let mut got = frames(&mut d);
            d.push(&wire[cut..]);
            got.extend(frames(&mut d));
            assert_eq!(
                got,
                [
                    Frame::Audio(vec![1, 2, 3, 4]),
                    Frame::State("thinking".into())
                ],
                "cut at {cut}"
            );
        }
    }

    #[test]
    fn an_odd_audio_byte_is_dropped_like_the_c_build_did() {
        let mut d = Decoder::default();
        d.push(&[b'A', 3, 0, 0, 0, 1, 0, 9]);
        assert_eq!(frames(&mut d), [Frame::Audio(vec![1])]);
    }

    #[test]
    fn an_oversized_frame_is_refused_before_its_payload_arrives() {
        let mut d = Decoder::default();
        let len = (MAX_FRAME as u32 + 1).to_le_bytes();
        d.push(&[b'A', len[0], len[1], len[2], len[3]]);
        assert_eq!(d.pull(), Err(BadFrame(MAX_FRAME + 1)));
        // The limit itself is fine.
        let mut d = Decoder::default();
        let len = (MAX_FRAME as u32).to_le_bytes();
        d.push(&[b'A', len[0], len[1], len[2], len[3]]);
        assert_eq!(d.pull(), Ok(None));
    }

    #[test]
    fn consumed_bytes_are_released() {
        let mut d = Decoder::default();
        let mut wire = Vec::new();
        for _ in 0..20_000 {
            put_frame(&mut wire, b'S', b"listening");
        }
        d.push(&wire);
        assert_eq!(frames(&mut d).len(), 20_000);
        d.push(&[]);
        assert!(d.buf.is_empty());
    }
}
