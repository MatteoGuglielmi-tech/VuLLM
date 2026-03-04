use std::io::{Write, stdout};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

pub struct Loader {
    thread_handle: Option<JoinHandle<()>>,
    // shared flag to signal the animation thread to stop.
    done_flag: Arc<AtomicBool>,
    end_msg: String,
}

impl Loader {
    /// Creates a new `Loader` and starts the animation in a background thread.
    ///
    /// # Arguments
    ///
    /// * `desc_msg` - The description message to display during the animation.
    /// * `end_msg` - The message to display when the animation is done.
    /// * `timeout` - The refresh interval for the animation spinner.
    pub fn new(desc_msg: &str, end_msg: &str, timeout: Duration) -> Self {
        let done_flag = Arc::new(AtomicBool::new(false));
        let done_flag_clone = Arc::clone(&done_flag);
        let desc_msg_clone = desc_msg.to_string();
        let full_end_msg = format!("\r{} -> {}\n", desc_msg, end_msg);

        let thread_handle = thread::spawn(move || {
            const SPINNER_CHARS: &[&str] = &["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"];
            let mut i = 0;

            // Loop until the main thread sets the `done_flag` to true.
            while !done_flag_clone.load(Ordering::Relaxed) {
                let char = SPINNER_CHARS[i % SPINNER_CHARS.len()];
                let mut stdout = stdout();
                print!("\r{} {}", desc_msg_clone, char);
                stdout.flush().unwrap_or_default();

                thread::sleep(timeout);
                i += 1;
            }
        });

        Loader {
            thread_handle: Some(thread_handle),
            done_flag,
            end_msg: full_end_msg,
        }
    }
}

/// The `Drop` trait is Rust's equivalent of a destructor.
/// This block is automatically executed when a `Loader` instance goes out of scope.
impl Drop for Loader {
    fn drop(&mut self) {
        // 1. Signal the animation thread to stop.
        self.done_flag.store(true, Ordering::Relaxed);

        // 2. Wait for the thread to finish its work.
        // We use `take()` to move the JoinHandle out of the Option.
        if let Some(handle) = self.thread_handle.take() {
            handle.join().expect("Failed to join the animation thread.");
        }

        // 3. Print the final "done" message.
        let mut stdout = stdout();
        print!("{}", self.end_msg);
        stdout.flush().unwrap_or_default();
    }
}
