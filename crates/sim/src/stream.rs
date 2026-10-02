//! Live telemetry over TCP: a [`TelemetrySink`] that streams a recorder's messages to any
//! number of viewers (`autonomousim-viewer attach <addr>`), which may connect and leave at
//! any time.
//!
//! Each connection receives an MCAP stream without chunks, index or footer: the magic, a
//! header, every schema and channel registered so far, then the messages a late viewer needs
//! to make sense of what follows (the last `/meta`, the last `/episode` and the `/route`
//! updates since), then the live messages in order. Records are MCAP's own (opcode, `u64`
//! length, content), so the bytes up to any record boundary parse as an MCAP file that ends
//! early ([`StreamReader`] parses them as they arrive).
//!
//! The simulation never waits for a viewer: each connection has a queue of [`QUEUE`]
//! records, and a viewer that falls further behind (or whose connection fails) is
//! disconnected; it reconnects and starts again from the cached messages.

use crate::SimError;
use crate::record::TelemetrySink;
use std::collections::BTreeMap;
use std::io::{BufReader, BufWriter, ErrorKind, Read, Write};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;

/// Records a connection may lag behind before it is disconnected.
pub const QUEUE: usize = 8192;

const MAGIC: &[u8] = b"\x89MCAP0\r\n";
const OP_HEADER: u8 = 0x01;
const OP_SCHEMA: u8 = 0x03;
const OP_CHANNEL: u8 = 0x04;
const OP_MESSAGE: u8 = 0x05;

fn record(op: u8, content: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(9 + content.len());
    out.push(op);
    out.extend((content.len() as u64).to_le_bytes());
    out.extend(content);
    out
}

fn string(out: &mut Vec<u8>, s: &str) {
    out.extend((s.len() as u32).to_le_bytes());
    out.extend(s.as_bytes());
}

fn header_record() -> Vec<u8> {
    let mut c = Vec::new();
    string(&mut c, "");
    string(&mut c, "autonomousim");
    record(OP_HEADER, &c)
}

/// The schema and channel records of channel `id` (schema id = channel id + 1).
fn channel_records(id: u16, topic: &str, schema_name: &str, schema: &str) -> Vec<u8> {
    let mut s = Vec::new();
    s.extend((id + 1).to_le_bytes());
    string(&mut s, schema_name);
    string(&mut s, "jsonschema");
    s.extend((schema.len() as u32).to_le_bytes());
    s.extend(schema.as_bytes());
    let mut c = Vec::new();
    c.extend(id.to_le_bytes());
    c.extend((id + 1).to_le_bytes());
    string(&mut c, topic);
    string(&mut c, "json");
    c.extend(0u32.to_le_bytes()); // no metadata
    let mut out = record(OP_SCHEMA, &s);
    out.extend(record(OP_CHANNEL, &c));
    out
}

fn message_record(channel: u16, sequence: u32, log_time: u64, data: &[u8]) -> Vec<u8> {
    let mut c = Vec::with_capacity(22 + data.len());
    c.extend(channel.to_le_bytes());
    c.extend(sequence.to_le_bytes());
    c.extend(log_time.to_le_bytes());
    c.extend(log_time.to_le_bytes());
    c.extend(data);
    record(OP_MESSAGE, &c)
}

/// Records sent to the connections (shared between them).
type Bytes = Arc<Vec<u8>>;

#[derive(Default)]
struct Shared {
    /// Header, schemas and channels.
    prefix: Vec<u8>,
    topics: Vec<String>,
    /// The last `/meta` and `/episode` messages and the `/route` updates since.
    meta: Option<Bytes>,
    episode: Option<Bytes>,
    routes: Vec<Bytes>,
    clients: Vec<SyncSender<Bytes>>,
    closed: bool,
}

/// A [`TelemetrySink`] that serves the stream to viewers on a TCP port.
pub struct StreamSink {
    shared: Arc<Mutex<Shared>>,
    addr: SocketAddr,
    sequence: u32,
    accept: Option<JoinHandle<()>>,
}

fn lock(shared: &Mutex<Shared>) -> MutexGuard<'_, Shared> {
    shared.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn io_err(e: std::io::Error) -> SimError {
    SimError::Record(format!("stream: {e}"))
}

impl StreamSink {
    /// Listen on `addr` (port 0: any free port, see [`StreamSink::local_addr`]).
    pub fn bind(addr: impl ToSocketAddrs) -> Result<Self, SimError> {
        let listener = TcpListener::bind(addr).map_err(io_err)?;
        let addr = listener.local_addr().map_err(io_err)?;
        let shared = Arc::new(Mutex::new(Shared { prefix: header_record(), ..Shared::default() }));
        let s = shared.clone();
        let accept = std::thread::Builder::new()
            .name("autonomousim-stream".into())
            .spawn(move || accept_loop(&listener, &s))
            .map_err(io_err)?;
        Ok(Self { shared, addr, sequence: 0, accept: Some(accept) })
    }

    /// The address viewers connect to.
    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }

    /// Connected viewers.
    pub fn clients(&self) -> usize {
        lock(&self.shared).clients.len()
    }

    fn close(&mut self) {
        {
            let mut s = lock(&self.shared);
            s.closed = true;
            s.clients.clear(); // their writers flush and close the connections
        }
        if let Some(accept) = self.accept.take() {
            // Wake the accept loop.
            let ip = match self.addr.ip() {
                IpAddr::V4(ip) if ip.is_unspecified() => IpAddr::V4(Ipv4Addr::LOCALHOST),
                IpAddr::V6(ip) if ip.is_unspecified() => IpAddr::V6(Ipv6Addr::LOCALHOST),
                ip => ip,
            };
            let _ = TcpStream::connect((ip, self.addr.port()));
            let _ = accept.join();
        }
    }
}

impl Drop for StreamSink {
    fn drop(&mut self) {
        self.close();
    }
}

fn accept_loop(listener: &TcpListener, shared: &Mutex<Shared>) {
    for stream in listener.incoming() {
        let mut s = lock(shared);
        if s.closed {
            return;
        }
        let Ok(stream) = stream else { continue };
        let _ = stream.set_nodelay(true);
        let (tx, rx) = sync_channel::<Bytes>(QUEUE);
        let mut start = MAGIC.to_vec();
        start.extend(&s.prefix);
        for m in s.meta.iter().chain(&s.episode).chain(&s.routes) {
            start.extend(m.iter());
        }
        let _ = tx.try_send(Arc::new(start));
        let spawned =
            std::thread::Builder::new().name("autonomousim-stream-client".into()).spawn(move || send_loop(stream, &rx));
        if spawned.is_ok() {
            s.clients.push(tx);
        }
    }
}

/// Write queued records until the sink closes or the connection fails.
fn send_loop(stream: TcpStream, rx: &Receiver<Bytes>) {
    let mut w = BufWriter::with_capacity(1 << 16, stream);
    while let Ok(mut bytes) = rx.recv() {
        loop {
            if w.write_all(&bytes).is_err() {
                return;
            }
            match rx.try_recv() {
                Ok(b) => bytes = b,
                Err(_) => break,
            }
        }
        if w.flush().is_err() {
            return;
        }
    }
    let _ = w.flush();
}

impl TelemetrySink for StreamSink {
    fn add_channel(&mut self, topic: &str, schema_name: &str, schema: &str) -> Result<u16, SimError> {
        let mut s = lock(&self.shared);
        let id = u16::try_from(s.topics.len()).map_err(|_| SimError::Record("too many channels".into()))?;
        let records = channel_records(id, topic, schema_name, schema);
        s.prefix.extend(&records);
        s.topics.push(topic.to_string());
        let records = Arc::new(records);
        s.clients.retain(|c| c.try_send(records.clone()).is_ok());
        Ok(id)
    }

    fn write(&mut self, channel: u16, log_time_ns: u64, data: &[u8]) -> Result<(), SimError> {
        self.sequence = self.sequence.wrapping_add(1);
        let mut s = lock(&self.shared);
        let Some(topic) = s.topics.get(usize::from(channel)) else {
            return Err(SimError::Record(format!("no channel {channel}")));
        };
        let kind = match topic.as_str() {
            "/meta" => 1,
            "/episode" => 2,
            "/route" => 3,
            _ if s.clients.is_empty() => return Ok(()),
            _ => 0,
        };
        let record = Arc::new(message_record(channel, self.sequence, log_time_ns, data));
        match kind {
            1 => (s.meta, s.episode, s.routes) = (Some(record.clone()), None, Vec::new()),
            2 => (s.episode, s.routes) = (Some(record.clone()), Vec::new()),
            3 => s.routes.push(record.clone()),
            _ => {}
        }
        s.clients.retain(|c| c.try_send(record.clone()).is_ok());
        Ok(())
    }

    fn finish(&mut self) -> Result<(), SimError> {
        self.close();
        Ok(())
    }

    fn live(&self) -> bool {
        !lock(&self.shared).clients.is_empty()
    }
}

/// A message of a stream.
#[derive(Clone, Debug)]
pub struct StreamMessage {
    pub topic: String,
    pub log_time_ns: u64,
    pub data: Vec<u8>,
}

/// Parses an MCAP stream as it arrives: a [`StreamSink`] connection, or an unchunked MCAP
/// file read front to back. Yields the messages and ends with the input (a record cut short
/// is an error).
pub struct StreamReader<R: Read> {
    r: BufReader<R>,
    topics: BTreeMap<u16, String>,
    started: bool,
}

/// The largest record a reader accepts.
const MAX_RECORD: u64 = 1 << 30;

fn stream_err(what: &str) -> SimError {
    SimError::Record(format!("stream: {what}"))
}

impl<R: Read> StreamReader<R> {
    pub fn new(r: R) -> Self {
        Self { r: BufReader::with_capacity(1 << 16, r), topics: BTreeMap::new(), started: false }
    }

    /// The next message, `None` at the end of the input.
    pub fn next_message(&mut self) -> Result<Option<StreamMessage>, SimError> {
        if !self.started {
            let mut magic = [0; 8];
            self.r.read_exact(&mut magic).map_err(io_err)?;
            if magic != MAGIC {
                return Err(stream_err("not an MCAP stream"));
            }
            self.started = true;
        }
        loop {
            let mut head = [0; 9];
            match self.r.read(&mut head[..1]) {
                Ok(0) => return Ok(None),
                Ok(_) => {}
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) => return Err(io_err(e)),
            }
            self.r.read_exact(&mut head[1..]).map_err(io_err)?;
            let len = u64::from_le_bytes(head[1..].try_into().expect("8 bytes"));
            if len > MAX_RECORD {
                return Err(stream_err("record too large"));
            }
            let mut c = vec![0; len as usize];
            self.r.read_exact(&mut c).map_err(io_err)?;
            match head[0] {
                OP_CHANNEL => {
                    let short = || stream_err("short channel record");
                    let id = u16::from_le_bytes(c.get(..2).ok_or_else(short)?.try_into().expect("2 bytes"));
                    let n = u32::from_le_bytes(c.get(4..8).ok_or_else(short)?.try_into().expect("4 bytes")) as usize;
                    let topic = c.get(8..8 + n).ok_or_else(short)?;
                    let topic = String::from_utf8(topic.to_vec()).map_err(|_| stream_err("topic not UTF-8"))?;
                    self.topics.insert(id, topic);
                }
                OP_MESSAGE => {
                    if c.len() < 22 {
                        return Err(stream_err("short message record"));
                    }
                    let channel = u16::from_le_bytes([c[0], c[1]]);
                    let log_time_ns = u64::from_le_bytes(c[6..14].try_into().expect("8 bytes"));
                    let topic = self.topics.get(&channel).ok_or_else(|| stream_err("message on an unknown channel"))?;
                    return Ok(Some(StreamMessage { topic: topic.clone(), log_time_ns, data: c[22..].to_vec() }));
                }
                _ => {}
            }
        }
    }
}

impl<R: Read> Iterator for StreamReader<R> {
    type Item = Result<StreamMessage, SimError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_message().transpose()
    }
}
