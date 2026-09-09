//! Quiet-period batching with an explicit monotonic time boundary.
//!
//! The watcher supplies real Instants; tests supply exact logical timestamps.
//! Keys retain the caller's path/kind identity and values retain the latest event.

use std::collections::HashMap;
use std::hash::Hash;
use std::time::{Duration, Instant};

pub(crate) struct DebounceBatch<K, V> {
    pending: HashMap<K, V>,
    last_event_at: Option<Instant>,
    quiet_period: Duration,
}

impl<K: Eq + Hash, V> DebounceBatch<K, V> {
    pub(crate) fn new(quiet_period: Duration) -> Self {
        Self {
            pending: HashMap::new(),
            last_event_at: None,
            quiet_period,
        }
    }

    /// Accumulate a value, replacing an earlier value with the same key.
    /// The caller restarts the quiet interval after processing its raw event.
    pub(crate) fn insert(&mut self, key: K, value: V) {
        self.pending.insert(key, value);
    }

    /// Record activity after processing a classified raw notification.
    /// This remains explicit so filtered notifications keep existing semantics.
    pub(crate) fn restart(&mut self, now: Instant) {
        self.last_event_at = Some(now);
    }

    /// Emit a nonempty batch once, at or after the quiet deadline.
    /// A time before the last activity cannot cause an early flush.
    pub(crate) fn take_due(&mut self, now: Instant) -> Option<Vec<V>> {
        let elapsed = now.checked_duration_since(self.last_event_at?)?;
        if self.pending.is_empty() || elapsed < self.quiet_period {
            return None;
        }
        Some(self.drain())
    }

    /// Drain pending values for shutdown and clear the activity timestamp.
    pub(crate) fn drain(&mut self) -> Vec<V> {
        self.last_event_at = None;
        self.pending.drain().map(|(_, value)| value).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::DebounceBatch;
    use std::time::{Duration, Instant};

    #[test]
    fn repeated_keys_coalesce_but_distinct_paths_and_kinds_survive() {
        let mut batch = DebounceBatch::new(Duration::from_millis(100));
        let now = Instant::now();
        batch.insert(("one.txt", "modify"), 1);
        batch.insert(("one.txt", "modify"), 2);
        batch.insert(("one.txt", "create"), 3);
        batch.insert(("two.txt", "modify"), 4);
        batch.restart(now);
        let mut values = batch.take_due(now + Duration::from_millis(100)).unwrap();
        values.sort_unstable();
        assert_eq!(values, vec![2, 3, 4]);
    }

    #[test]
    fn later_activity_restarts_the_entire_quiet_interval() {
        let mut batch = DebounceBatch::new(Duration::from_millis(100));
        let now = Instant::now();
        batch.insert("path", 1);
        batch.restart(now);
        assert_eq!(batch.take_due(now + Duration::from_millis(89)), None);
        batch.insert("path", 2);
        batch.restart(now + Duration::from_millis(90));
        assert_eq!(batch.take_due(now + Duration::from_millis(100)), None);
        assert_eq!(batch.take_due(now + Duration::from_millis(189)), None);
        assert_eq!(
            batch.take_due(now + Duration::from_millis(190)),
            Some(vec![2])
        );
    }

    #[test]
    fn exact_deadline_flushes_once_and_never_early() {
        let mut batch = DebounceBatch::new(Duration::from_millis(150));
        let now = Instant::now();
        batch.insert("path", 7);
        batch.restart(now);
        assert_eq!(batch.take_due(now + Duration::from_millis(149)), None);
        assert_eq!(
            batch.take_due(now + Duration::from_millis(150)),
            Some(vec![7])
        );
        assert_eq!(batch.take_due(now + Duration::from_millis(300)), None);
    }

    #[test]
    fn empty_batches_do_not_emit_callbacks() {
        let mut batch = DebounceBatch::<&str, i32>::new(Duration::from_millis(10));
        let now = Instant::now();
        assert_eq!(batch.take_due(now), None);
        batch.restart(now);
        assert_eq!(batch.take_due(now + Duration::from_secs(1)), None);
        assert!(batch.drain().is_empty());
    }

    #[test]
    fn classified_notification_without_visible_paths_keeps_existing_timing() {
        // Preserve the watcher's existing restart after a classified raw event,
        // even when its paths are filtered; this repair does not change filtering.
        let mut batch = DebounceBatch::new(Duration::from_millis(100));
        let now = Instant::now();
        batch.insert("visible", 1);
        batch.restart(now);
        batch.restart(now + Duration::from_millis(80));
        assert_eq!(batch.take_due(now + Duration::from_millis(100)), None);
        assert_eq!(
            batch.take_due(now + Duration::from_millis(180)),
            Some(vec![1])
        );
    }

    #[test]
    fn shutdown_drains_pending_changes_once_before_the_deadline() {
        let mut batch = DebounceBatch::new(Duration::from_secs(1));
        let now = Instant::now();
        batch.insert("path", 9);
        batch.restart(now);
        assert_eq!(batch.take_due(now), None);
        assert_eq!(batch.drain(), vec![9]);
        assert!(batch.drain().is_empty());
        assert_eq!(batch.take_due(now + Duration::from_secs(2)), None);
        batch.insert("next", 10);
        batch.restart(now + Duration::from_secs(3));
        assert_eq!(batch.take_due(now + Duration::from_secs(4)), Some(vec![10]));
    }

    #[test]
    fn zero_delay_emits_immediately() {
        let mut batch = DebounceBatch::new(Duration::ZERO);
        let now = Instant::now();
        batch.insert("path", 1);
        batch.restart(now);
        assert_eq!(batch.take_due(now), Some(vec![1]));
        assert_eq!(batch.take_due(now), None);
    }

    #[test]
    fn time_before_last_activity_cannot_flush() {
        let mut batch = DebounceBatch::new(Duration::ZERO);
        let now = Instant::now();
        batch.insert("path", 1);
        batch.restart(now + Duration::from_millis(1));
        assert_eq!(batch.take_due(now), None);
        assert_eq!(
            batch.take_due(now + Duration::from_millis(1)),
            Some(vec![1])
        );
    }
}
