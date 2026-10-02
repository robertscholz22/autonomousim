//! A ROS 2 node (DDS through `ros2-client`) with typed publishers and subscriptions for the
//! [`RosMessage`] types, and the bridge's QoS profiles.

use crate::msgs::RosMessage;
use anyhow::{Context as _, anyhow};
use ros2_client::qos::{Durability, History, WhenFull};
use ros2_client::{Context, ContextOptions, MessageTypeName, Name, Node, NodeName, NodeOptions, QosProfile};

pub use ros2_client::{Publisher, Subscription};

/// QoS profiles (ROS's presets: sensor data, reliable, latched).
pub mod qos {
    use super::*;

    /// Sensor data: best effort, the last 5.
    pub const SENSOR: QosProfile =
        QosProfile::subscription_default().reliability_best_effort().history(History::KeepLast { depth: 5 });
    /// State and commands: reliable, the last 10.
    pub const RELIABLE: QosProfile = QosProfile::subscription_default()
        .reliability_reliable(WhenFull::DEFAULT)
        .history(History::KeepLast { depth: 10 });
    /// Latched (meta, static transforms): reliable and transient local, the last one.
    pub const LATCHED: QosProfile = QosProfile::subscription_default()
        .reliability_reliable(WhenFull::DEFAULT)
        .durability(Durability::TransientLocal)
        .history(History::KeepLast { depth: 1 });
}

/// A node in a DDS domain; its spinner (graph, parameters) runs on its own thread.
pub struct RosNode {
    node: Node,
    _context: Context,
}

impl RosNode {
    /// `name` in `namespace` (e.g. "/") on DDS domain `domain_id` (`ROS_DOMAIN_ID`).
    pub fn new(namespace: &str, name: &str, domain_id: u16) -> anyhow::Result<Self> {
        let context = Context::with_options(ContextOptions::new().domain_id(domain_id))
            .map_err(|e| anyhow!("DDS participant on domain {domain_id}: {e:?}"))?;
        let node_name = NodeName::new(namespace, name).map_err(|e| anyhow!("node name {namespace}/{name}: {e:?}"))?;
        let mut node = context
            .new_node(node_name, NodeOptions::new().enable_rosout(true))
            .map_err(|e| anyhow!("node {name}: {e:?}"))?;
        let spinner = node.spinner().map_err(|e| anyhow!("spinner: {e:?}"))?;
        std::thread::Builder::new()
            .name("ros-spinner".into())
            .spawn(move || futures::executor::block_on(spinner.spin()))
            .context("spinner thread")?;
        Ok(Self { node, _context: context })
    }

    fn topic<M: RosMessage>(
        &mut self,
        topic: &str,
        qos: &QosProfile,
    ) -> anyhow::Result<ros2_client::dds::rustdds::Topic> {
        let (package, name) = M::TYPE.split_once('/').expect("TYPE is package/Name");
        let topic_name = Name::parse(topic).map_err(|e| anyhow!("topic name {topic}: {e:?}"))?;
        self.node
            .create_topic(&topic_name, MessageTypeName::new(package, name), qos)
            .map_err(|e| anyhow!("topic {topic}: {e:?}"))
    }

    pub fn publisher<M: RosMessage>(&mut self, topic: &str, qos: QosProfile) -> anyhow::Result<Publisher<M>> {
        let t = self.topic::<M>(topic, &qos)?;
        self.node.create_publisher(&t, Some(qos)).map_err(|e| anyhow!("publisher {topic}: {e:?}"))
    }

    pub fn subscription<M: RosMessage>(&mut self, topic: &str, qos: QosProfile) -> anyhow::Result<Subscription<M>> {
        let t = self.topic::<M>(topic, &qos)?;
        self.node.create_subscription(&t, Some(qos)).map_err(|e| anyhow!("subscription {topic}: {e:?}"))
    }

    pub fn inner(&mut self) -> &mut Node {
        &mut self.node
    }
}
