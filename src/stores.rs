/*!
Implementation of various caches

*/

use std::cmp::Eq;
use std::collections::HashMap;
use std::hash::Hash;
use std::time::Instant;

use super::Cached;

use std::collections::hash_map::{
    Entry as HashMapEntry, OccupiedEntry as HashMapOccupiedEntry, VacantEntry as hashMapVacantEntry,
};

/// Default unbounded cache
///
/// This cache has no size limit or eviction policy.
///
/// Note: This cache is in-memory only
#[derive(Clone, Debug)]
pub struct UnboundCache<K, V> {
    store: HashMap<K, V>,
    hits: u64,
    misses: u64,
    initial_capacity: Option<usize>,
}

impl<K, V> PartialEq for UnboundCache<K, V>
where
    K: Eq + Hash,
    V: PartialEq,
{
    fn eq(&self, other: &UnboundCache<K, V>) -> bool {
        self.store.eq(&other.store)
    }
}

impl<K, V> Eq for UnboundCache<K, V>
where
    K: Eq + Hash,
    V: PartialEq,
{
}

impl<K: Hash + Eq, V> UnboundCache<K, V> {
    /// Creates an empty `UnboundCache`
    #[allow(clippy::new_without_default)]
    pub fn new() -> UnboundCache<K, V> {
        UnboundCache {
            store: Self::new_store(None),
            hits: 0,
            misses: 0,
            initial_capacity: None,
        }
    }

    /// Creates an empty `UnboundCache` with a given pre-allocated capacity
    pub fn with_capacity(size: usize) -> UnboundCache<K, V> {
        UnboundCache {
            store: Self::new_store(Some(size)),
            hits: 0,
            misses: 0,
            initial_capacity: Some(size),
        }
    }

    fn new_store(capacity: Option<usize>) -> HashMap<K, V> {
        capacity.map_or_else(HashMap::new, HashMap::with_capacity)
    }
}

impl<K: Hash + Eq, V> Cached<K, V> for UnboundCache<K, V> {
    fn get(&mut self, key: &K) -> Option<&V> {
        match self.store.get(key) {
            Some(v) => {
                self.hits += 1;
                Some(v)
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    fn get_mut(&mut self, key: &K) -> std::option::Option<&mut V> {
        match self.store.get_mut(key) {
            Some(v) => {
                self.hits += 1;
                Some(v)
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    fn insert(&mut self, key: K, val: V) -> (Option<V>, &V) {
        let (old, new) = self.insert_mut(key, val);
        (old, new)
    }

    fn insert_mut(&mut self, key: K, val: V) -> (Option<V>, &mut V) {
        match self.store.entry(key) {
            std::collections::hash_map::Entry::Occupied(mut occupied) => {
                let old = occupied.insert(val);
                (Some(old), occupied.into_mut())
            }
            std::collections::hash_map::Entry::Vacant(vacant) => (None, vacant.insert(val)),
        }
    }

    fn remove(&mut self, k: &K) -> Option<V> {
        self.store.remove(k)
    }

    fn remove_entry(&mut self, k: &K) -> Option<(K, V)> {
        self.store.remove_entry(k)
    }

    fn clear(&mut self) {
        self.store.clear();
    }

    fn reset(&mut self) {
        self.store = Self::new_store(self.initial_capacity);
    }

    fn size(&self) -> usize {
        self.store.len()
    }

    fn hits(&self) -> Option<u64> {
        Some(self.hits)
    }

    fn misses(&self) -> Option<u64> {
        Some(self.misses)
    }
}

/// Limited functionality doubly linked list using Vec as storage.
#[derive(Clone, Debug)]
struct LRUList<T> {
    values: Vec<ListEntry<T>>,
}

#[derive(Clone, Debug)]
struct ListEntry<T> {
    value: Option<T>,
    next: usize,
    prev: usize,
}

/// Free and occupied cells are each linked into a cyclic list with one auxiliary cell.
/// Cell #0 is on the list of free cells, element #1 is on the list of occupied cells.
///
impl<T> LRUList<T> {
    const FREE: usize = 0;
    const OCCUPIED: usize = 1;

    fn with_capacity(capacity: usize) -> LRUList<T> {
        let mut values = Vec::with_capacity(capacity + 2);
        values.push(ListEntry::<T> {
            value: None,
            next: 0,
            prev: 0,
        });
        values.push(ListEntry::<T> {
            value: None,
            next: 1,
            prev: 1,
        });
        LRUList { values }
    }

    fn unlink(&mut self, index: usize) {
        let prev = self.values[index].prev;
        let next = self.values[index].next;
        self.values[prev].next = next;
        self.values[next].prev = prev;
    }

    fn link_after(&mut self, index: usize, prev: usize) {
        let next = self.values[prev].next;
        self.values[index].prev = prev;
        self.values[index].next = next;
        self.values[prev].next = index;
        self.values[next].prev = index;
    }

    fn move_to_front(&mut self, index: usize) {
        self.unlink(index);
        self.link_after(index, Self::OCCUPIED);
    }

    fn push_front(&mut self, value: Option<T>) -> usize {
        if self.values[Self::FREE].next == Self::FREE {
            self.values.push(ListEntry::<T> {
                value: None,
                next: Self::FREE,
                prev: Self::FREE,
            });
            self.values[Self::FREE].next = self.values.len() - 1;
        }
        let index = self.values[Self::FREE].next;
        self.values[index].value = value;
        self.unlink(index);
        self.link_after(index, Self::OCCUPIED);
        index
    }

    fn remove(&mut self, index: usize) -> T {
        self.unlink(index);
        self.link_after(index, Self::FREE);
        self.values[index].value.take().expect("invalid index")
    }

    fn back(&self) -> usize {
        self.values[Self::OCCUPIED].prev
    }

    fn pop_back(&mut self) -> T {
        let index = self.back();
        self.remove(index)
    }

    fn get_back(&self) -> &T {
        self.values[self.back()]
            .value
            .as_ref()
            .expect("invalid index")
    }

    // fn get(&self, index: usize) -> &T {
    //     self.values[index].value.as_ref().expect("invalid index")
    // }

    fn get_mut(&mut self, index: usize) -> &mut T {
        self.values[index].value.as_mut().expect("invalid index")
    }

    // fn insert(&mut self, index: usize, value: T) -> (Option<T>, &T) {
    //     let (old, new) = self.insert_mut(index, value);
    //     (old, new)
    // }

    fn insert_mut(&mut self, index: usize, value: T) -> (Option<T>, &mut T) {
        let old = core::mem::replace(&mut self.values[index].value, Some(value));
        (old, self.get_mut(index))
    }

    fn clear(&mut self) {
        self.values.clear();
        self.values.push(ListEntry::<T> {
            value: None,
            next: 0,
            prev: 0,
        });
        self.values.push(ListEntry::<T> {
            value: None,
            next: 1,
            prev: 1,
        });
    }

    fn iter(&self) -> LRUListIterator<T> {
        LRUListIterator::<T> {
            list: self,
            index: Self::OCCUPIED,
        }
    }
}

#[derive(Debug)]
struct LRUListIterator<'a, T> {
    list: &'a LRUList<T>,
    index: usize,
}

impl<'a, T> Iterator for LRUListIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.list.values[self.index].next;
        if next == LRUList::<T>::OCCUPIED {
            None
        } else {
            let value = self.list.values[next].value.as_ref();
            self.index = next;
            value
        }
    }
}

/// Least Recently Used / `Sized` Cache
///
/// Stores up to a specified size before beginning
/// to evict the least recently used keys
///
/// Note: This cache is in-memory only
#[derive(Clone, Debug)]
pub struct SizedCache<K, V> {
    store: HashMap<K, usize>,
    order: LRUList<(K, V)>,
    capacity: usize,
    hits: u64,
    misses: u64,
}

impl<K, V> PartialEq for SizedCache<K, V>
where
    K: Eq + Hash,
    V: PartialEq,
{
    fn eq(&self, other: &SizedCache<K, V>) -> bool {
        self.store.eq(&other.store)
    }
}

impl<K, V> Eq for SizedCache<K, V>
where
    K: Eq + Hash,
    V: PartialEq,
{
}

pub enum Entry<'a, K: 'a, V: 'a> {
    Occupied(OccupiedEntry<'a, K, V>),
    Vacant(VacantEntry<'a, K, V>),
}

pub struct OccupiedEntry<'a, K: 'a, V: 'a> {
    inner: HashMapOccupiedEntry<'a, K, usize>,
    order: &'a mut LRUList<(K, V)>,
    hits: &'a mut u64,
}

impl<'a, K, V> OccupiedEntry<'a, K, V> {
    pub fn get_mut(&mut self) -> &mut V {
        let index = *self.inner.get();
        self.order.move_to_front(index);
        *self.hits += 1;
        &mut self.order.get_mut(index).1
    }

    pub fn into_mut(self) -> &'a mut V {
        let index = *self.inner.get();
        self.order.move_to_front(index);
        *self.hits += 1;
        &mut self.order.get_mut(index).1
    }

    pub fn key(&self) -> &K {
        self.inner.key()
    }
}

pub struct VacantEntry<'a, K: 'a, V: 'a> {
    inner: hashMapVacantEntry<'a, K, usize>,
    order: &'a mut LRUList<(K, V)>,
    misses: &'a mut u64,
}

impl<'a, K: Clone, V> VacantEntry<'a, K, V> {
    pub fn insert(self, value: V) -> &'a mut V {
        *self.misses += 1;
        let key = self.inner.key().clone();
        let index = *self.inner.insert(self.order.push_front(None));
        let (_, new) = self.order.insert_mut(index, (key, value));
        &mut new.1
    }

    pub fn key(&self) -> &K {
        self.inner.key()
    }
}

impl<'a, K: Clone, V> Entry<'a, K, V> {
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(occupied) => occupied.into_mut(),
            Entry::Vacant(vacant) => vacant.insert(default),
        }
    }

    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        match self {
            Entry::Occupied(occupied) => occupied.into_mut(),
            Entry::Vacant(vacant) => vacant.insert(default()),
        }
    }

    pub fn key(&self) -> &K {
        match self {
            Entry::Occupied(occupied) => occupied.key(),
            Entry::Vacant(vacant) => vacant.key(),
        }
    }

    pub fn and_modify<F>(self, f: F) -> Self
    where
        F: FnOnce(&mut V),
    {
        match self {
            Entry::Occupied(mut occupied) => {
                f(occupied.get_mut());
                Entry::Occupied(occupied)
            }
            Entry::Vacant(vacant) => Entry::Vacant(vacant),
        }
    }
}

impl<'a, K: Clone, V: Default> Entry<'a, K, V> {
    pub fn or_default(self) -> &'a mut V {
        match self {
            Entry::Occupied(occupied) => occupied.into_mut(),
            Entry::Vacant(vacant) => vacant.insert(Default::default()),
        }
    }
}

impl<K: Hash + Eq, V> SizedCache<K, V> {
    #[deprecated(since = "0.5.1", note = "method renamed to `with_size`")]
    pub fn with_capacity(size: usize) -> SizedCache<K, V> {
        Self::with_size(size)
    }

    /// Creates a new `SizedCache` with a given size limit and pre-allocated backing data
    pub fn with_size(size: usize) -> SizedCache<K, V> {
        if size == 0 {
            panic!("`size` of `SizedCache` must be greater than zero.")
        }
        SizedCache {
            store: HashMap::with_capacity(size),
            order: LRUList::<(K, V)>::with_capacity(size),
            capacity: size,
            hits: 0,
            misses: 0,
        }
    }

    /// Return an iterator of keys in the current order from most
    /// to least recently used.
    pub fn key_order(&self) -> impl Iterator<Item = &K> {
        self.order.iter().map(|(k, _v)| k)
    }

    /// Return an iterator of values in the current order from most
    /// to least recently used.
    pub fn value_order(&self) -> impl Iterator<Item = &V> {
        self.order.iter().map(|(_k, v)| v)
    }

    pub fn entry<'a>(&'a mut self, k: K) -> Entry<'a, K, V> {
        if self.store.len() >= self.capacity {
            // store has reached capacity, evict the oldest item.
            // store capacity cannot be zero, so there must be content in `self.order`.
            let (key, _value) = self.order.get_back();
            if key != &k {
                let (key, _value) = self.order.pop_back();
                self.store
                    .remove(&key)
                    .expect("SizedCache::set failed evicting cache key");
            }
        }
        match self.store.entry(k) {
            HashMapEntry::Occupied(inner) => Entry::Occupied(OccupiedEntry {
                inner,
                order: &mut self.order,
                hits: &mut self.hits,
            }),
            HashMapEntry::Vacant(inner) => Entry::Vacant(VacantEntry {
                inner,
                order: &mut self.order,
                misses: &mut self.misses,
            }),
        }
    }
}

impl<K: Hash + Eq + Clone, V> Cached<K, V> for SizedCache<K, V> {
    fn get(&mut self, key: &K) -> Option<&V> {
        self.get_mut(key).map(|v| &*v)
    }

    fn get_mut(&mut self, key: &K) -> std::option::Option<&mut V> {
        let val = self.store.get(key);
        match val {
            Some(&index) => {
                self.order.move_to_front(index);
                self.hits += 1;
                Some(&mut self.order.get_mut(index).1)
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    fn insert(&mut self, key: K, val: V) -> (Option<V>, &V) {
        let (old, new) = self.insert_mut(key, val);
        (old, new)
    }

    fn insert_mut(&mut self, key: K, val: V) -> (Option<V>, &mut V) {
        if self.store.len() >= self.capacity {
            // store has reached capacity, evict the oldest item.
            // store capacity cannot be zero, so there must be content in `self.order`.
            let (key, _value) = self.order.pop_back();
            self.store
                .remove(&key)
                .expect("SizedCache::set failed evicting cache key");
        }
        let Self { store, order, .. } = self;
        let index = *store
            .entry(key.clone())
            .or_insert_with(|| order.push_front(None));
        let (old, new) = order.insert_mut(index, (key, val));
        (old.map(|(_, v)| v), &mut new.1)
    }

    fn remove(&mut self, k: &K) -> Option<V> {
        self.remove_entry(k).map(|(_, v)| v)
    }

    fn remove_entry(&mut self, k: &K) -> Option<(K, V)> {
        // try and remove item from mapping, and then from order list if it was in mapping
        if let Some(index) = self.store.remove(k) {
            // need to remove the key in the order list
            let (key, value) = self.order.remove(index);
            Some((key, value))
        } else {
            None
        }
    }

    fn clear(&mut self) {
        // clear both the store and the order list
        self.store.clear();
        self.order.clear();
    }
    fn reset(&mut self) {
        // SizedCache uses clear because capacity is fixed.
        self.clear();
    }
    fn size(&self) -> usize {
        self.store.len()
    }
    fn hits(&self) -> Option<u64> {
        Some(self.hits)
    }
    fn misses(&self) -> Option<u64> {
        Some(self.misses)
    }
    fn capacity(&self) -> Option<usize> {
        Some(self.capacity)
    }
}

/// Enum used for defining the status of time-cached values
#[derive(Debug)]
enum Status {
    NotFound,
    Found,
    Expired,
}

/// Cache store bound by time
///
/// Values are timestamped when inserted and are
/// evicted if expired at time of retrieval.
///
/// Note: This cache is in-memory only
#[derive(Clone, Debug)]
pub struct TimedCache<K, V> {
    store: HashMap<K, (Instant, V)>,
    seconds: u64,
    hits: u64,
    misses: u64,
    initial_capacity: Option<usize>,
}

impl<K: Hash + Eq, V> TimedCache<K, V> {
    /// Creates a new `TimedCache` with a specified lifespan
    pub fn with_lifespan(seconds: u64) -> TimedCache<K, V> {
        TimedCache {
            store: Self::new_store(None),
            seconds,
            hits: 0,
            misses: 0,
            initial_capacity: None,
        }
    }

    /// Creates a new `TimedCache` with a specified lifespan and
    /// cache-store with the specified pre-allocated capacity
    pub fn with_lifespan_and_capacity(seconds: u64, size: usize) -> TimedCache<K, V> {
        TimedCache {
            store: Self::new_store(Some(size)),
            seconds,
            hits: 0,
            misses: 0,
            initial_capacity: Some(size),
        }
    }

    fn new_store(capacity: Option<usize>) -> HashMap<K, (Instant, V)> {
        capacity.map_or_else(HashMap::new, HashMap::with_capacity)
    }
}

impl<K: Hash + Eq, V> Cached<K, V> for TimedCache<K, V> {
    fn get(&mut self, key: &K) -> Option<&V> {
        let status = {
            let val = self.store.get(key);
            if let Some(&(instant, _)) = val {
                if instant.elapsed().as_secs() < self.seconds {
                    Status::Found
                } else {
                    Status::Expired
                }
            } else {
                Status::NotFound
            }
        };
        match status {
            Status::NotFound => {
                self.misses += 1;
                None
            }
            Status::Found => {
                self.hits += 1;
                self.store.get(key).map(|stamped| &stamped.1)
            }
            Status::Expired => {
                self.misses += 1;
                self.store.remove(key).unwrap();
                None
            }
        }
    }

    fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let status = {
            let val = self.store.get(key);
            if let Some(&(instant, _)) = val {
                if instant.elapsed().as_secs() < self.seconds {
                    Status::Found
                } else {
                    Status::Expired
                }
            } else {
                Status::NotFound
            }
        };
        match status {
            Status::NotFound => {
                self.misses += 1;
                None
            }
            Status::Found => {
                self.hits += 1;
                self.store.get_mut(key).map(|stamped| &mut stamped.1)
            }
            Status::Expired => {
                self.misses += 1;
                self.store.remove(key).unwrap();
                None
            }
        }
    }

    fn insert(&mut self, key: K, val: V) -> (Option<V>, &V) {
        let (old, new) = self.insert_mut(key, val);
        (old, new)
    }

    fn insert_mut(&mut self, key: K, val: V) -> (Option<V>, &mut V) {
        let stamped = (Instant::now(), val);
        match self.store.entry(key) {
            std::collections::hash_map::Entry::Occupied(mut occupied) => {
                let old = occupied.insert(stamped);
                (Some(old.1), &mut occupied.into_mut().1)
            }
            std::collections::hash_map::Entry::Vacant(vacant) => {
                (None, &mut vacant.insert(stamped).1)
            }
        }
    }

    fn remove(&mut self, k: &K) -> Option<V> {
        self.store.remove(k).map(|(_, v)| v)
    }

    fn remove_entry(&mut self, k: &K) -> Option<(K, V)> {
        self.store.remove_entry(k).map(|(k, (_, v))| (k, v))
    }

    fn clear(&mut self) {
        self.store.clear();
    }
    fn reset(&mut self) {
        self.store = Self::new_store(self.initial_capacity);
    }
    fn size(&self) -> usize {
        self.store.len()
    }
    fn hits(&self) -> Option<u64> {
        Some(self.hits)
    }
    fn misses(&self) -> Option<u64> {
        Some(self.misses)
    }
    fn lifespan(&self) -> Option<u64> {
        Some(self.seconds)
    }
}

#[cfg(test)]
/// Cache store tests
mod tests {
    use std::thread::sleep;
    use std::time::Duration;

    use super::Cached;

    use super::SizedCache;
    use super::TimedCache;
    use super::UnboundCache;

    #[test]
    fn basic_cache() {
        let mut c = UnboundCache::new();
        assert!(c.get(&1).is_none());
        let misses = c.misses().unwrap();
        assert_eq!(1, misses);

        c.insert(1, 100);
        assert!(c.get(&1).is_some());
        let hits = c.hits().unwrap();
        let misses = c.misses().unwrap();
        assert_eq!(1, hits);
        assert_eq!(1, misses);
    }

    #[test]
    fn sized_cache() {
        let mut c = SizedCache::with_size(5);
        assert!(c.get(&1).is_none());
        let misses = c.misses().unwrap();
        assert_eq!(1, misses);

        c.insert(1, 100);
        assert!(c.get(&1).is_some());
        let hits = c.hits().unwrap();
        let misses = c.misses().unwrap();
        assert_eq!(1, hits);
        assert_eq!(1, misses);

        c.insert(2, 100);
        c.insert(3, 100);
        c.insert(4, 100);
        c.insert(5, 100);

        assert_eq!(c.key_order().cloned().collect::<Vec<_>>(), [5, 4, 3, 2, 1]);

        c.insert(6, 100);
        c.insert(7, 100);

        assert_eq!(c.key_order().cloned().collect::<Vec<_>>(), [7, 6, 5, 4, 3]);

        assert!(c.get(&2).is_none());
        assert!(c.get(&3).is_some());

        assert_eq!(c.key_order().cloned().collect::<Vec<_>>(), [3, 7, 6, 5, 4]);

        assert_eq!(2, c.misses().unwrap());
        let size = c.size();
        assert_eq!(5, size);
    }

    #[test]
    /// This is a regression test to confirm that racing cache sets on a SizedCache
    /// do not cause duplicates to exist in the internal `order`. See issue #7
    fn size_racing_keys_eviction_regression() {
        let mut c = SizedCache::with_size(2);
        c.insert(1, 100);
        c.insert(1, 100);
        // size would be 1, but internal ordered would be [1, 1]
        c.insert(2, 100);
        c.insert(3, 100);
        // this next set would fail because a duplicate key would be evicted
        c.insert(4, 100);
    }

    #[test]
    fn timed_cache() {
        let mut c = TimedCache::with_lifespan(2);
        assert!(c.get(&1).is_none());
        let misses = c.misses().unwrap();
        assert_eq!(1, misses);

        c.insert(1, 100);
        assert!(c.get(&1).is_some());
        let hits = c.hits().unwrap();
        let misses = c.misses().unwrap();
        assert_eq!(1, hits);
        assert_eq!(1, misses);

        sleep(Duration::new(2, 0));
        assert!(c.get(&1).is_none());
        let misses = c.misses().unwrap();
        assert_eq!(2, misses);
    }

    #[test]
    fn clear() {
        let mut c = UnboundCache::new();

        c.insert(1, 100);
        c.insert(2, 200);
        c.insert(3, 300);

        // register some hits and misses
        c.get(&1);
        c.get(&2);
        c.get(&3);
        c.get(&10);
        c.get(&20);
        c.get(&30);

        assert_eq!(3, c.size());
        assert_eq!(3, c.hits().unwrap());
        assert_eq!(3, c.misses().unwrap());
        assert_eq!(3, c.store.capacity());

        // clear the cache, should have no more elements
        // hits and misses will still be kept
        c.clear();

        assert_eq!(0, c.size());
        assert_eq!(3, c.hits().unwrap());
        assert_eq!(3, c.misses().unwrap());
        assert_eq!(3, c.store.capacity()); // Keeps the allocated memory for reuse.

        let capacity = 1;
        let mut c = UnboundCache::with_capacity(capacity);
        assert_eq!(capacity, c.store.capacity());

        c.insert(1, 100);
        c.insert(2, 200);
        c.insert(3, 300);

        assert_eq!(3, c.store.capacity());

        c.clear();

        assert_eq!(3, c.store.capacity()); // Keeps the allocated memory for reuse.

        let mut c = SizedCache::with_size(3);

        c.insert(1, 100);
        c.insert(2, 200);
        c.insert(3, 300);
        c.clear();

        assert_eq!(0, c.size());

        let mut c = TimedCache::with_lifespan(3600);

        c.insert(1, 100);
        c.insert(2, 200);
        c.insert(3, 300);
        c.clear();

        assert_eq!(0, c.size());
    }

    #[test]
    fn reset() {
        let mut c = UnboundCache::new();
        c.insert(1, 100);
        c.insert(2, 200);
        c.insert(3, 300);
        assert_eq!(3, c.store.capacity());

        c.reset();

        assert_eq!(0, c.store.capacity());

        let init_capacity = 1;
        let mut c = UnboundCache::with_capacity(init_capacity);
        c.insert(1, 100);
        c.insert(2, 200);
        c.insert(3, 300);
        assert_eq!(3, c.store.capacity());

        c.reset();

        assert_eq!(init_capacity, c.store.capacity());

        let mut c = SizedCache::with_size(init_capacity);
        c.insert(1, 100);
        c.insert(2, 200);
        c.insert(3, 300);
        assert_eq!(init_capacity, c.store.capacity());

        c.reset();

        assert_eq!(init_capacity, c.store.capacity());

        let mut c = TimedCache::with_lifespan(100);
        c.insert(1, 100);
        c.insert(2, 200);
        c.insert(3, 300);
        assert_eq!(3, c.store.capacity());

        c.reset();

        assert_eq!(0, c.store.capacity());

        let mut c = TimedCache::with_lifespan_and_capacity(100, init_capacity);
        c.insert(1, 100);
        c.insert(2, 200);
        c.insert(3, 300);
        assert_eq!(3, c.store.capacity());

        c.reset();

        assert_eq!(init_capacity, c.store.capacity());
    }

    #[test]
    fn remove() {
        let mut c = UnboundCache::new();

        c.insert(1, 100);
        c.insert(2, 200);
        c.insert(3, 300);

        // register some hits and misses
        c.get(&1);
        c.get(&2);
        c.get(&3);
        c.get(&10);
        c.get(&20);
        c.get(&30);

        assert_eq!(3, c.size());
        assert_eq!(3, c.hits().unwrap());
        assert_eq!(3, c.misses().unwrap());

        // remove some items from cache
        // hits and misses will still be kept
        assert_eq!(Some(100), c.remove(&1));

        assert_eq!(2, c.size());
        assert_eq!(3, c.hits().unwrap());
        assert_eq!(3, c.misses().unwrap());

        assert_eq!(Some(200), c.remove(&2));

        assert_eq!(1, c.size());

        // removing extra is ok
        assert_eq!(None, c.remove(&2));

        assert_eq!(1, c.size());

        let mut c = SizedCache::with_size(3);

        c.insert(1, 100);
        c.insert(2, 200);
        c.insert(3, 300);

        assert_eq!(Some(100), c.remove(&1));
        assert_eq!(2, c.size());

        assert_eq!(Some(200), c.remove(&2));
        assert_eq!(1, c.size());

        assert_eq!(None, c.remove(&2));
        assert_eq!(1, c.size());

        assert_eq!(Some(300), c.remove(&3));
        assert_eq!(0, c.size());

        let mut c = TimedCache::with_lifespan(3600);

        c.insert(1, 100);
        c.insert(2, 200);
        c.insert(3, 300);

        assert_eq!(Some(100), c.remove(&1));
        assert_eq!(2, c.size());
    }

    #[test]
    fn sized_get_mut() {
        let mut c = SizedCache::with_size(5);
        assert!(c.get_mut(&1).is_none());
        let misses = c.misses().unwrap();
        assert_eq!(1, misses);

        c.insert(1, 100);
        assert_eq!(*c.get_mut(&1).unwrap(), 100);
        let hits = c.hits().unwrap();
        let misses = c.misses().unwrap();
        assert_eq!(1, hits);
        assert_eq!(1, misses);

        let value = c.get_mut(&1).unwrap();
        *value = 10;

        let hits = c.hits().unwrap();
        let misses = c.misses().unwrap();
        assert_eq!(2, hits);
        assert_eq!(1, misses);
        assert_eq!(*c.get_mut(&1).unwrap(), 10);
    }
}
