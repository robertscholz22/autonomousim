//! Watching a running simulation (`attach <host:port>`): a thread reads the stream a
//! [`StreamSink`](autonomousim_sim::stream::StreamSink) serves (connecting again whenever the
//! connection ends) and the replay follows the latest recorded state, a little behind it.

use autonomousim_sim::record::Recording;
use autonomousim_sim::stream::{StreamMessage, StreamReader};
use serde_json::Value;
use std::net::TcpStream;
use std::sync::Mutex;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Duration;

/// Playback time behind the latest recorded state when following (s): a frame or two of the
/// state rate, so arriving states do not make the agents stutter.
pub const LAG: f64 = 0.1;
/// Following falls at most this far behind (s): a simulation that runs faster than real time
/// is shown in jumps.
pub const MAX_BEHIND: f64 = 1.0;
/// Episodes kept for scrubbing back.
pub const KEEP: usize = 50;
/// Messages taken from the reader per frame at most (the rest the next frame).
const PER_FRAME: usize = 20_000;

/// What the reader thread delivers.
pub enum Feed {
    /// A connection, with its `/meta` message.
    Connected(Vec<u8>),
    Message(StreamMessage),
    /// The connection ended (or could not be made); it is tried again.
    Disconnected(String),
}

/// A live feed of a recording.
pub struct Live {
    pub addr: String,
    /// In a mutex since a receiver is not `Sync` (and the viewer's resources must be).
    rx: Mutex<Receiver<Feed>>,
    /// Why the stream is not followed (connection lost, a different run).
    pub status: Option<String>,
    /// The connection streams another scenario than the one shown; its messages are skipped.
    foreign: bool,
    /// The connection started with the cached episode, which may be the last one shown.
    resumed: bool,
    /// Episodes dropped from the front (only [`KEEP`] are kept).
    pub dropped: usize,
}

/// Read the stream at `addr` on a new thread, reconnecting every second.
pub fn spawn(addr: &str) -> Receiver<Feed> {
    let (tx, rx) = channel();
    let addr = addr.to_owned();
    std::thread::Builder::new()
        .name("autonomousim-attach".into())
        .spawn(move || {
            let mut last_error = String::new();
            loop {
                let e = match read(&addr, &tx) {
                    Ok(()) => "the stream ended".to_owned(),
                    Err(Gone) => return,
                    Err(Failed(e)) => e,
                };
                if e != last_error && tx.send(Feed::Disconnected(e.clone())).is_err() {
                    return;
                }
                last_error = e;
                std::thread::sleep(Duration::from_secs(1));
            }
        })
        .expect("a thread");
    rx
}

use ReadError::{Failed, Gone};
enum ReadError {
    /// The viewer is gone.
    Gone,
    Failed(String),
}

fn read(addr: &str, tx: &Sender<Feed>) -> Result<(), ReadError> {
    let stream = TcpStream::connect(addr).map_err(|e| Failed(format!("connecting to {addr}: {e}")))?;
    let mut first = true;
    for m in StreamReader::new(stream) {
        let m = m.map_err(|e| Failed(e.to_string()))?;
        let feed = if first {
            if m.topic != "/meta" {
                return Err(Failed(format!("{addr} streams no recording ({} first)", m.topic)));
            }
            first = false;
            Feed::Connected(m.data)
        } else {
            Feed::Message(m)
        };
        tx.send(feed).map_err(|_| Gone)?;
    }
    Ok(())
}

/// Wait for the stream at `addr` to deliver a recording with an episode (printing why while
/// it does not); the rest follows through [`Live::poll`].
pub fn connect(addr: &str) -> anyhow::Result<(Recording, Live)> {
    let rx = spawn(addr);
    let mut rec: Option<Recording> = None;
    loop {
        match rx.recv()? {
            Feed::Disconnected(e) => {
                println!("{e}; trying again");
                rec = None;
            }
            Feed::Connected(meta) => rec = Some(Recording::from_meta(&meta)?),
            Feed::Message(m) => {
                if let Some(r) = &mut rec {
                    r.push(&m.topic, m.log_time_ns, &m.data)?;
                }
            }
        }
        if let Some(r) = &rec
            && r.episodes.last().is_some_and(|e| e.states.iter().any(|s| !s.is_empty()))
        {
            break;
        }
    }
    let live =
        Live { addr: addr.to_owned(), rx: Mutex::new(rx), status: None, foreign: false, resumed: false, dropped: 0 };
    Ok((rec.expect("a recording"), live))
}

/// `/episode`'s number and seed.
fn episode_id(data: &[u8]) -> Option<(u64, String)> {
    let v: Value = serde_json::from_slice(data).ok()?;
    Some((v.get("episode")?.as_u64()?, v.get("seed")?.as_str()?.to_owned()))
}

impl Live {
    /// Add what arrived to `rec`; returns the number of episodes dropped from its front.
    pub fn poll(&mut self, rec: &mut Recording) -> usize {
        let rx = self.rx.get_mut().unwrap_or_else(std::sync::PoisonError::into_inner);
        for _ in 0..PER_FRAME {
            let Ok(feed) = rx.try_recv() else { break };
            match feed {
                Feed::Disconnected(e) => {
                    self.status = Some(format!("{e}; trying again"));
                }
                Feed::Connected(meta) => {
                    self.foreign = serde_json::from_slice::<Value>(&meta).ok().as_ref() != Some(&rec.meta);
                    self.resumed = true;
                    self.status =
                        self.foreign.then(|| format!("{} streams another run (restart the viewer)", self.addr));
                }
                Feed::Message(m) if !self.foreign => {
                    if m.topic == "/episode" && std::mem::take(&mut self.resumed) {
                        // The episode shown when the connection was lost goes on.
                        let last = rec.episodes.last().map(|e| (e.number, e.seed.clone()));
                        if last.is_some() && episode_id(&m.data) == last {
                            continue;
                        }
                    }
                    if let Err(e) = rec.push(&m.topic, m.log_time_ns, &m.data) {
                        self.status = Some(e.to_string());
                    }
                }
                Feed::Message(_) => {}
            }
        }
        let extra = rec.episodes.len().saturating_sub(KEEP);
        rec.episodes.drain(..extra);
        self.dropped += extra;
        extra
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replay::Replay;
    use autonomousim_sim::record::{Recorder, RecorderConfig};
    use autonomousim_sim::{BatchSim, Scenario};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    /// Wait (updating the replay) until `done` or fail after 20 s.
    fn until(r: &mut Replay, what: &str, mut done: impl FnMut(&Replay) -> bool) {
        let start = Instant::now();
        let mut last = Instant::now();
        while !done(r) {
            assert!(start.elapsed() < Duration::from_secs(20), "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(5));
            r.update(last.elapsed().as_secs_f64(), false);
            last = Instant::now();
            if r.follow {
                let end = r.current().duration();
                // (Closer than LAG only at the start of an episode.)
                assert!(r.time <= end && r.time >= end - MAX_BEHIND - 1e-9, "{} of {end}", r.time);
            }
        }
    }

    /// A simulation in real time on its own thread: streams from the start, resets at step
    /// 100, restarts its stream at step 150 (as if the process came back) and stops at 400.
    #[test]
    fn the_viewer_attaches_mid_run_follows_resets_and_reattaches() {
        let steps = Arc::new(AtomicU64::new(0));
        let (tx, rx) = channel();
        let sim = {
            let steps = steps.clone();
            std::thread::spawn(move || {
                let sc = Scenario::load(concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/scenarios/hover.toml"));
                let mut b = BatchSim::new(sc.unwrap(), 1, 5, 1).unwrap();
                let (recorder, addr) = Recorder::stream("127.0.0.1:0", RecorderConfig::default()).unwrap();
                b.attach_recorder(0, recorder);
                tx.send(addr).unwrap();
                for k in 1..=400 {
                    std::thread::sleep(Duration::from_millis(20));
                    b.step(&[&[0.0; 4]]);
                    steps.store(k, Ordering::SeqCst);
                    if k == 100 {
                        b.reset(None, Some(&[8]));
                    }
                    if k == 150 {
                        b.detach_recorder(0).unwrap().finish().unwrap();
                        std::thread::sleep(Duration::from_millis(1500));
                        b.attach_recorder(0, Recorder::stream(addr, RecorderConfig::default()).unwrap().0);
                    }
                }
            })
        };
        let addr = rx.recv().unwrap().to_string();
        while steps.load(Ordering::SeqCst) < 20 {
            std::thread::sleep(Duration::from_millis(5));
        }

        // Mid-run: the running episode from where the viewer came in, followed.
        let (recording, live) = connect(&addr).unwrap();
        let first = recording.episodes[0].states[0][0].time;
        assert!(first > 0.3, "joined at {first}");
        let mut r = Replay::attached(recording, live);
        assert!(r.follow && r.live.as_ref().unwrap().status.is_none());
        until(&mut r, "the playback to move", |r| r.time > first + 0.5);
        // Scrubbing back stops following; End goes back to it.
        r.seek(0.0);
        assert!(!r.follow);
        r.follow_latest();
        // The reset: the next episode is followed.
        until(&mut r, "the next episode", |r| r.current().number == 1);
        assert_eq!(r.episode_number(), (2, 2));
        // The stream goes away and comes back: the viewer says so, reconnects and goes on.
        until(&mut r, "the stream to end", |r| r.live.as_ref().unwrap().status.is_some());
        until(&mut r, "the stream to come back", |r| r.live.as_ref().unwrap().status.is_none());
        until(&mut r, "the restarted stream's episode", |r| r.recording.episodes.len() == 3);
        let resumed = r.time;
        until(&mut r, "the playback to go on", |r| r.time > resumed + 0.5);
        assert_eq!(r.episode, 2);

        // Another viewer attaches (the first one detaches): it starts with the running episode.
        let seed = r.current().seed.clone();
        drop(r);
        let (again, _) = connect(&addr).unwrap();
        assert_eq!(again.episodes.len(), 1);
        assert_eq!(again.episodes[0].seed, seed);
        sim.join().unwrap();
    }
}
