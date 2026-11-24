//! The ray-tracing structure builder's compute stages, off by default.
//!
//! The builder's stages feed each other through memory: a stage writes offsets that a later
//! stage reads. A stage that does not run (or a stand-in in place of one whose pipeline is
//! still building) leaves the next stages computing their store offsets from whatever the
//! memory holds, and the GPU hangs. So a builder dispatch runs only when every builder
//! pipeline made so far is fully built (all or none), and by default none run:
//! `XCL_BVH_BUILDER=1` runs them under that rule.
//!
//! A builder pipeline is a compute pipeline whose console program has a spilled user-data
//! table: more root arguments than the inline user SGPRs hold (`spills`).

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use super::console_list::Layout;

/// `XCL_BVH_BUILDER=1`: builder dispatches run, all or none.
pub fn enabled() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| std::env::var("XCL_BVH_BUILDER").is_ok_and(|v| v == "1"))
}

/// Whether a compute program's root arguments spill to the user-data table: more user data
/// than the inline user SGPRs hold by the layout's rule, or more than the program's own user
/// SGPRs hold (`console_list::inline_for` applies both).
pub fn spills(l: &Layout, user_sgprs: Option<u32>) -> bool {
    super::console_list::inline_for(l, user_sgprs) < l.size
}

/// What to do with a dispatch of a pipeline.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Step {
    Run,
    Skip,
    /// skip, and run no builder dispatch for the rest of the run: a stage is not built
    SkipAndOff,
}

/// The decision, without side effects (`skip` applies it).
pub fn decide(flag: bool, builder: bool, all_built: bool, off: bool) -> Step {
    if !builder {
        return Step::Run;
    }
    if off || !flag {
        return Step::Skip;
    }
    if all_built {
        Step::Run
    } else {
        Step::SkipAndOff
    }
}

/// Every address a builder pipeline is known by: the stand-in the title holds and the real
/// pipeline built on a worker behind it.
static BUILDER: Mutex<Option<HashSet<usize>>> = Mutex::new(None);
/// builder pipelines handed to the title so far
static CREATED: AtomicU32 = AtomicU32::new(0);
/// a builder pipeline was not built when a builder dispatch came: off for the rest of the run
static OFF: AtomicBool = AtomicBool::new(false);
static SKIPPED: AtomicU64 = AtomicU64::new(0);

fn is_builder(pso: usize) -> bool {
    pso != 0 && BUILDER.lock().unwrap().as_ref().is_some_and(|s| s.contains(&pso))
}

/// A builder pipeline went to the title (its stand-in, or its real pipeline when built in the
/// call): remembered for the all-or-none rule, logged first and at powers of two.
pub fn note_created(pso: usize) {
    if pso == 0 {
        return;
    }
    BUILDER.lock().unwrap().get_or_insert_with(HashSet::new).insert(pso);
    let n = CREATED.fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() {
        let how = if enabled() {
            "run, all or none"
        } else {
            "their dispatches skipped, XCL_BVH_BUILDER=1 runs them"
        };
        crate::xlog_info!("D3D12.X: ray-tracing structure builder pipelines so far: {n} ({how})");
    }
}

/// The real pipeline of a builder, built on a worker behind its stand-in: remembered so a
/// dispatch that resolved to it is recognised.
pub fn note_built(pso: usize) {
    if pso != 0 {
        BUILDER.lock().unwrap().get_or_insert_with(HashSet::new).insert(pso);
    }
}

/// Every builder pipeline made so far is a real, finished pipeline (not still building, not a
/// stand-in whose build failed).
pub fn all_built() -> bool {
    let g = BUILDER.lock().unwrap();
    g.as_ref().map_or(true, |s| s.iter().all(|&p| super::pso_async::is_built(p)))
}

/// Whether this dispatch of `pso` must be skipped (counted and logged here).
pub fn skip(pso: usize) -> bool {
    let p = super::pso_async::resolve(pso);
    if !is_builder(p) {
        return false;
    }
    let step = decide(enabled(), true, all_built(), OFF.load(Ordering::Relaxed));
    if step == Step::Run {
        return false;
    }
    if step == Step::SkipAndOff && !OFF.swap(true, Ordering::Relaxed) {
        crate::xlog_warn!(
            "D3D12.X: ray-tracing structure builder turned off for the rest of the run: a builder pipeline is not built yet, and its stages pass their store offsets through memory"
        );
    }
    let n = SKIPPED.fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() {
        crate::xlog_info!("D3D12.X: ray-tracing structure builder off: {n} dispatches skipped (XCL_BVH_BUILDER=1 runs it)");
    }
    true
}
