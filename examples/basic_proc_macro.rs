use cached::proc_macro::cached;
use std::thread::sleep;
use std::time::{Duration, Instant};

#[cached(size = 50)]
fn slow_fn(n: u32) -> String {
    if n == 0 {
        return "done".to_string();
    }
    sleep(Duration::new(1, 0));
    slow_fn(n - 1)
}

pub fn main() {
    println!("Initial run...");
    let now = Instant::now();
    let _ = slow_fn(10);
    println!("Elapsed: {}\n", now.elapsed().as_secs());

    println!("Cached run...");
    let now = Instant::now();
    let _ = slow_fn(10);
    println!("Elapsed: {}\n", now.elapsed().as_secs());

    // Inspect the cache
    {
        use cached::Cached; // must be in scope to access cache

        println!(" ** Cache info **");
        let cache = SLOW_FN.lock().unwrap();
        println!("hits=1 -> {:?}", cache.hits().unwrap() == 1);
        println!("misses=11 -> {:?}", cache.misses().unwrap() == 11);
        // make sure the cache-lock is dropped
    }
}
