//! Kinematic-tree model and per-instance state.

use super::JointType;
use crate::math::{Pose, RigidInertia, Xform};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// One rigid link and the joint connecting it to its parent.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Link {
    pub name: String,
    /// Parent link index (always smaller than this link's index); `None` = world.
    pub parent: Option<usize>,
    pub joint: JointType,
    /// Tree transform `X_T`: parent link frame → joint predecessor frame.
    pub x_tree: Xform,
    /// Inertia in this link's frame.
    pub inertia: RigidInertia,
    /// Prescribed (kinematically driven) joint: its acceleration is an input to the ABA and
    /// the required joint force is an output (hybrid dynamics, RBDA §9.2).
    pub prescribed: bool,
}

/// Immutable kinematic tree with topologically ordered links (`parent < child`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MultibodyModel {
    links: Vec<Link>,
    q_off: Vec<usize>,
    v_off: Vec<usize>,
    nq: usize,
    nv: usize,
}

impl MultibodyModel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a link. `joint_frame` is the pose of the joint (predecessor) frame in the parent
    /// link frame (or in the world for root links). Returns the new link's index.
    pub fn add_link(
        &mut self,
        name: impl Into<String>,
        parent: Option<usize>,
        joint: JointType,
        joint_frame: Pose,
        inertia: RigidInertia,
    ) -> usize {
        let idx = self.links.len();
        if let Some(p) = parent {
            assert!(p < idx, "parent link {p} must be added before child {idx}");
        }
        self.q_off.push(self.nq);
        self.v_off.push(self.nv);
        self.nq += joint.nq();
        self.nv += joint.nv();
        self.links.push(Link {
            name: name.into(),
            parent,
            joint,
            x_tree: joint_frame.to_xform(),
            inertia,
            prescribed: false,
        });
        idx
    }

    /// Mark a joint as prescribed (its acceleration is given instead of its force).
    pub fn set_prescribed(&mut self, link: usize, prescribed: bool) {
        self.links[link].prescribed = prescribed;
    }

    #[inline]
    pub fn links(&self) -> &[Link] {
        &self.links
    }

    #[inline]
    pub fn link(&self, i: usize) -> &Link {
        &self.links[i]
    }

    #[inline]
    pub fn num_links(&self) -> usize {
        self.links.len()
    }

    #[inline]
    pub fn nq(&self) -> usize {
        self.nq
    }

    #[inline]
    pub fn nv(&self) -> usize {
        self.nv
    }

    #[inline]
    pub fn q_offset(&self, link: usize) -> usize {
        self.q_off[link]
    }

    #[inline]
    pub fn v_offset(&self, link: usize) -> usize {
        self.v_off[link]
    }

    /// Position coordinates of `link`'s joint.
    #[inline]
    pub fn q_slice<'a>(&self, link: usize, q: &'a [f64]) -> &'a [f64] {
        let o = self.q_off[link];
        &q[o..o + self.links[link].joint.nq()]
    }

    /// Velocity coordinates of `link`'s joint.
    #[inline]
    pub fn v_slice<'a>(&self, link: usize, v: &'a [f64]) -> &'a [f64] {
        let o = self.v_off[link];
        &v[o..o + self.links[link].joint.nv()]
    }

    pub fn find_link(&self, name: &str) -> Option<usize> {
        self.links.iter().position(|l| l.name == name)
    }

    /// Inertia of the body if this model is a single free-floating rigid body.
    #[inline]
    pub fn single_free_body_inertia(&self) -> Option<&RigidInertia> {
        match self.links.as_slice() {
            [l] if l.joint == JointType::Free => Some(&l.inertia),
            _ => None,
        }
    }

    pub fn total_mass(&self) -> f64 {
        self.links.iter().map(|l| l.inertia.mass).sum()
    }

    /// State at the neutral configuration with zero velocity.
    pub fn neutral_state(&self) -> MbState {
        let mut s = MbState { q: SmallVec::from_elem(0.0, self.nq), v: SmallVec::from_elem(0.0, self.nv) };
        for (i, l) in self.links.iter().enumerate() {
            let o = self.q_off[i];
            l.joint.neutral_q(&mut s.q[o..o + l.joint.nq()]);
        }
        s
    }
}

/// Generalised coordinates of one multibody instance.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MbState {
    pub q: SmallVec<[f64; 16]>,
    pub v: SmallVec<[f64; 16]>,
}

impl MbState {
    pub fn is_finite(&self) -> bool {
        self.q.iter().chain(self.v.iter()).all(|x| x.is_finite())
    }
}
