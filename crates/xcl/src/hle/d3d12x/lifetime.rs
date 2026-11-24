//! PC object lifetime behind the title's command lists.
//!
//! A title may release an object right after it submits a list that names it. The PC runtime
//! does not keep a command list's own reference to the resources, pipelines, root signatures
//! and heaps recorded in it, so freeing the object at that release lets the GPU read freed
//! memory. XCL therefore holds one PC reference of its own to every object a list names, and
//! gives that reference back only when the title has released the object and every submission
//! that names it has completed. A queue's submissions complete in order: the fence value the
//! title signals covers every submission queued before it on that queue.
//!
//! Release paths ask [`released`] first: it answers `true` when the object is one of ours and
//! its final release must wait, and `false` when the caller keeps its old behaviour.

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::sync::Mutex;

/// Frees the PC object once nothing in flight names it (the object's release path in `real`).
pub(super) type FreeFn = unsafe fn(usize);

struct Entry {
    free: FreeFn,
    /// lists still recording (not yet submitted) that name the object
    open: usize,
    /// the title released its last reference
    gone: bool,
}

struct Submission {
    queue: usize,
    /// the fence and value a later queue Signal covered it with (0: none yet)
    fence: usize,
    value: u64,
    assigned: bool,
    done: bool,
    objects: Vec<usize>,
}

impl Default for Submission {
    fn default() -> Self {
        Submission { queue: 0, fence: 0, value: 0, assigned: false, done: false, objects: Vec::new() }
    }
}

#[derive(Default)]
struct State {
    objects: HashMap<usize, Entry>,
    /// PC command list → the objects its recording names
    lists: HashMap<usize, HashSet<usize>>,
    in_flight: Vec<Submission>,
    deferred: Vec<usize>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

#[cfg(test)]
fn with<R>(f: impl FnOnce(&mut State) -> R) -> R {
    let mut g = STATE.lock().unwrap_or_else(|e| e.into_inner());
    f(g.get_or_insert_with(State::default))
}

/// The completed value of a fleet of PC fences (a fence the runtime does not know counts as done).
fn completed(fence: usize) -> u64 {
    #[cfg(test)]
    if let Some(v) = TEST_FENCES.lock().unwrap().as_ref() {
        return v.get(&fence).copied().unwrap_or(0);
    }
    use windows::core::Interface;
    let p = fence as *mut c_void;
    match unsafe { windows::Win32::Graphics::Direct3D12::ID3D12Fence::from_raw_borrowed(&p) } {
        Some(f) => unsafe { f.GetCompletedValue() },
        None => u64::MAX,
    }
}

#[cfg(test)]
static TEST_FENCES: Mutex<Option<HashMap<usize, u64>>> = Mutex::new(None);

/// Take XCL's own PC reference so the object outlives the title's release.
unsafe fn hold(handle: usize) {
    #[cfg(test)]
    if TEST_FENCES.lock().unwrap().is_some() {
        return;
    }
    let add: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(super::real::orig_fn(handle as *mut c_void, 1));
    add(handle as *mut c_void);
}

/// `list` records a command that names `handle`: hold the object until that submission is done.
pub(super) fn note(list: usize, handle: usize) {
    if list == 0 || handle == 0 {
        return;
    }
    let known = STATE.lock().unwrap_or_else(|e| e.into_inner()).as_ref().is_some_and(|s| s.objects.contains_key(&handle));
    if !known {
        let Some(free) = (unsafe { super::real::free_fn(handle) }) else { return };
        let mut g = STATE.lock().unwrap_or_else(|e| e.into_inner());
        let st = g.get_or_insert_with(State::default);
        if !st.objects.contains_key(&handle) {
            unsafe { hold(handle) };
            st.objects.insert(handle, Entry { free, open: 0, gone: false });
        }
    }
    let mut g = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let st = g.get_or_insert_with(State::default);
    if st.lists.entry(list).or_default().insert(handle) {
        if let Some(e) = st.objects.get_mut(&handle) {
            e.open += 1;
        }
    }
}

/// The list was reset or released: its recording no longer names anything it had not submitted.
pub(super) fn forget_list(list: usize) {
    let drop = {
        let mut g = STATE.lock().unwrap_or_else(|e| e.into_inner());
        let Some(st) = g.as_mut() else { return };
        let Some(refs) = st.lists.remove(&list) else { return };
        for h in refs {
            if let Some(e) = st.objects.get_mut(&h) {
                e.open = e.open.saturating_sub(1);
            }
        }
        true
    };
    if drop {
        poll();
    }
}

/// `lists` went to `queue`: what they recorded now waits on that queue's submissions.
pub(super) fn submit(queue: usize, lists: &[*mut c_void]) {
    let mut objects = Vec::new();
    {
        let mut g = STATE.lock().unwrap_or_else(|e| e.into_inner());
        let Some(st) = g.as_mut() else { return };
        for &l in lists {
            let Some(refs) = st.lists.remove(&(l as usize)) else { continue };
            for h in refs {
                if let Some(e) = st.objects.get_mut(&h) {
                    e.open = e.open.saturating_sub(1);
                }
                if !objects.contains(&h) {
                    objects.push(h);
                }
            }
        }
        if !objects.is_empty() {
            st.in_flight.push(Submission { queue, objects, ..Default::default() });
        }
    }
    poll();
}

/// The queue signalled `fence` with `value`: every submission queued before it on that queue
/// is done no later than that value.
pub(super) fn signal(queue: usize, fence: usize, value: u64) {
    {
        let mut g = STATE.lock().unwrap_or_else(|e| e.into_inner());
        let Some(st) = g.as_mut() else { return };
        for s in st.in_flight.iter_mut().filter(|s| !s.done && !s.assigned && s.queue == queue) {
            s.fence = fence;
            s.value = value;
            s.assigned = true;
        }
    }
    poll();
}

/// Called from an object's release path with the count the runtime returned. `true`: the object
/// is XCL's and must not be freed here (now or yet), so `free` runs from [`poll`] later.
/// `false`: not an object a recorded list holds; the caller does what it did before.
pub(super) fn released(handle: usize, remaining: u32) -> bool {
    poll();
    let mut g = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(st) = g.as_mut() else { return false };
    let Some(e) = st.objects.get_mut(&handle) else { return false };
    if remaining <= 1 && !e.gone {
        e.gone = true;
        st.deferred.push(handle);
    }
    true
}

/// Free every deferred object that no open recording and no incomplete submission names.
pub(super) fn poll() {
    let free: Vec<(usize, FreeFn)> = {
        let mut g = STATE.lock().unwrap_or_else(|e| e.into_inner());
        let Some(st) = g.as_mut() else { return };
        for s in st.in_flight.iter_mut() {
            eprintln!("poll: sub q={:x} fence={:x} val={} assigned={} done={} objs={:x?} completed={}", s.queue, s.fence, s.value, s.assigned, s.done, s.objects, s.assigned.then(|| completed(s.fence)).unwrap_or(0));
            if !s.done && s.assigned && completed(s.fence) >= s.value {
                s.done = true;
            }
        }
        eprintln!("poll: deferred={:x?} objects={:x?}", st.deferred, st.objects.keys().collect::<Vec<_>>());
        let mut free = Vec::new();
        st.deferred.retain(|&h| {
            let busy = st.objects.get(&h).map_or(true, |e| e.open > 0) || st.in_flight.iter().any(|s| !s.done && s.objects.contains(&h));
            if !busy {
                if let Some(e) = st.objects.remove(&h) {
                    free.push((h, e.free));
                }
            }
            !busy
        });
        eprintln!("poll: after retain deferred={:x?} free={:x?}", st.deferred, free.iter().map(|x| x.0).collect::<Vec<_>>());
        let waiting: HashSet<usize> = st.deferred.iter().copied().collect();
        st.in_flight.retain_mut(|s| {
            if s.done {
                s.objects.retain(|o| waiting.contains(o));
            }
            !s.done || !s.objects.is_empty()
        });
        // done submissions that name nothing still waiting no longer matter
        free
    };
    for (h, f) in free {
        unsafe { f(h) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static FREED: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    /// The registers are process-wide.
    static SERIAL: Mutex<()> = Mutex::new(());

    unsafe fn record_free(h: usize) {
        FREED.lock().unwrap().push(h);
    }

    fn reset() {
        *STATE.lock().unwrap() = None;
        *TEST_FENCES.lock().unwrap() = Some(HashMap::new());
        FREED.lock().unwrap().clear();
    }

    fn track(h: usize) {
        with(|st| {
            st.objects.insert(h, Entry { free: record_free, open: 0, gone: false });
        });
    }

    fn set_completed(fence: usize, value: u64) {
        TEST_FENCES.lock().unwrap().as_mut().unwrap().insert(fence, value);
    }

    /// An object queued for release is not freed before its fence value is reached.
    #[test]
    fn a_released_object_waits_for_its_fence_value() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        track(0x1000);
        note(0x100, 0x1000);
        submit(0x700, &[0x100 as *mut c_void]);
        assert!(released(0x1000, 1), "a tracked object's release is deferred");
        poll();
        assert!(FREED.lock().unwrap().is_empty(), "freed before any fence: {:x?}", FREED.lock().unwrap());
        signal(0x700, 0x900, 4);
        poll();
        assert!(FREED.lock().unwrap().is_empty(), "freed before the fence reached its value");
        set_completed(0x900, 3);
        poll();
        assert!(FREED.lock().unwrap().is_empty(), "freed below the value");
        set_completed(0x900, 4);
        poll();
        assert_eq!(*FREED.lock().unwrap(), vec![0x1000], "not freed at the fence value");
    }

    /// A list that still records the object keeps it alive even after the title released it;
    /// the list's submission then covers it.
    #[test]
    fn an_open_recording_keeps_a_released_object() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        track(0x2000);
        note(0x200, 0x2000);
        assert!(released(0x2000, 1));
        poll();
        assert!(FREED.lock().unwrap().is_empty(), "freed while a recording still names it");
        submit(0x710, &[0x200 as *mut c_void]);
        poll();
        assert!(FREED.lock().unwrap().is_empty(), "freed before its submission completed");
        signal(0x710, 0x910, 1);
        set_completed(0x910, 1);
        poll();
        assert_eq!(*FREED.lock().unwrap(), vec![0x2000]);
    }

    /// A reset list drops its recordings, so a released object it named goes at once.
    #[test]
    fn a_reset_recording_releases_its_hold() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        track(0x3000);
        note(0x300, 0x3000);
        assert!(released(0x3000, 1));
        forget_list(0x300);
        assert_eq!(*FREED.lock().unwrap(), vec![0x3000]);
    }

    /// Another queue's signal does not complete this queue's submissions.
    #[test]
    fn submission_order_is_per_queue() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        track(0x4000);
        note(0x400, 0x4000);
        submit(0x720, &[0x400 as *mut c_void]);
        assert!(released(0x4000, 1));
        signal(0x721, 0x920, 1);
        set_completed(0x920, 1);
        poll();
        assert!(FREED.lock().unwrap().is_empty(), "another queue's fence freed it");
        signal(0x720, 0x921, 1);
        set_completed(0x921, 1);
        poll();
        assert_eq!(*FREED.lock().unwrap(), vec![0x4000]);
    }
}
