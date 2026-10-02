//! Live telemetry: viewers attach to a running simulation over TCP, mid-run, and follow it.

use autonomousim_sim::record::{Recorder, RecorderConfig, Recording};
use autonomousim_sim::stream::{StreamMessage, StreamReader};
use autonomousim_sim::{BatchSim, Scenario};
use std::io::Read;
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

fn hover() -> BatchSim {
    let sc = Scenario::load(concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/scenarios/hover.toml")).unwrap();
    BatchSim::new(sc, 1, 3, 1).unwrap()
}

fn connect(addr: SocketAddr) -> StreamReader<TcpStream> {
    let s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
    StreamReader::new(s)
}

/// Read up to and including `/meta` (from then on the sink sends the viewer every message).
fn meta<R: Read>(r: &mut StreamReader<R>) -> StreamMessage {
    let m = r.next_message().unwrap().unwrap();
    assert_eq!(m.topic, "/meta");
    m
}

/// The rest of a stream as a recording.
fn read_all<R: Read>(first: StreamMessage, r: StreamReader<R>) -> Recording {
    let mut rec = Recording::from_meta(&first.data).unwrap();
    for m in r {
        let m = m.unwrap();
        rec.push(&m.topic, m.log_time_ns, &m.data).unwrap();
    }
    rec
}

#[test]
fn viewers_attach_mid_run_follow_resets_and_reattach() {
    let mut b = hover();
    let (recorder, addr) = Recorder::stream("127.0.0.1:0", RecorderConfig::default()).unwrap();
    b.attach_recorder(0, recorder);
    let step = |b: &mut BatchSim, n: usize| (0..n).for_each(|_| b.step(&[&[0.0, 0.0, 0.0, 0.0]]));
    step(&mut b, 20);

    // A viewer attaches mid-episode: it gets the recording's meta and the running episode.
    let mut first = connect(addr);
    let m = meta(&mut first);
    let first = std::thread::spawn(move || read_all(m, first));
    // Another one comes and goes.
    let mut leaving = connect(addr);
    meta(&mut leaving);
    step(&mut b, 20);
    assert_eq!(leaving.next_message().unwrap().unwrap().topic, "/episode");
    drop(leaving);
    step(&mut b, 10);
    b.reset(None, Some(&[11]));
    step(&mut b, 30);
    // A viewer attaching after the reset starts with the new episode.
    let mut late = connect(addr);
    let m = meta(&mut late);
    let late = std::thread::spawn(move || read_all(m, late));
    step(&mut b, 30);
    let end = b.world(0).time();
    b.detach_recorder(0).unwrap().finish().unwrap();

    let first = first.join().unwrap();
    assert_eq!(first.episodes.len(), 2);
    let (a, b2) = (&first.episodes[0], &first.episodes[1]);
    // From some time into the first episode (the episode's own message is sent first) to its
    // end, then all of the second; at the state rate without gaps.
    let dt = 1.0 / f64::from(first.state_hz);
    let times = |e: &autonomousim_sim::record::RecordedEpisode| e.states[0].iter().map(|s| s.time).collect::<Vec<_>>();
    for t in [times(a), times(b2)] {
        assert!(t.windows(2).all(|w| (w[1] - w[0] - dt).abs() < 1e-9), "{t:?}");
    }
    assert!(times(a)[0] > 0.3 && times(a)[0] < 0.5, "joined at {}", times(a)[0]);
    assert!((times(a).last().unwrap() - 1.0).abs() < 1e-9);
    assert_eq!((times(b2)[0], *times(b2).last().unwrap()), (0.0, end));
    assert_eq!((a.number, b2.number), (0, 1));
    assert!(!a.goals[0].is_empty() && b2.goals[0] != a.goals[0]);

    let late = late.join().unwrap();
    assert_eq!(late.episodes.len(), 1);
    let e = &late.episodes[0];
    assert_eq!((e.number, e.seed.as_str(), e.goals[0].clone()), (1, b2.seed.as_str(), b2.goals[0].clone()));
    assert!(times(e)[0] > 0.6 && times(e)[0] < 0.7 && *times(e).last().unwrap() == end);
    assert_eq!(late.meta, first.meta);
}

#[test]
fn a_viewer_that_falls_behind_is_disconnected() {
    let mut b = hover();
    let (recorder, addr) = Recorder::stream("127.0.0.1:0", RecorderConfig::default()).unwrap();
    b.attach_recorder(0, recorder);
    let mut slow = connect(addr);
    meta(&mut slow);
    // Far more than the queue and the socket buffers hold, while the viewer reads nothing.
    for _ in 0..40_000 {
        b.step(&[&[0.0, 0.0, 0.0, 0.0]]);
    }
    // Its connection ends while the simulation is still streaming.
    let mut n = 0;
    while slow.next_message().unwrap().is_some() {
        n += 1;
    }
    assert!(n > 1000, "{n} messages before the end");
    assert!(b.detach_recorder(0).is_some());
}
