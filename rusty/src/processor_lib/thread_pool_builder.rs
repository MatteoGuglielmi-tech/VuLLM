use log::error;
// use log::info;

#[cfg(target_os = "macos")]
use std::convert::TryInto;

pub fn setup_global_thread_pool() {
    // --- Configure the global Rayon thread pool ---
    rayon::ThreadPoolBuilder::new()
        // .num_threads(6) // limit the num_threads to #P-cores
        .start_handler(|thread_index| {
            // info!("Rayon thread {} started", thread_index);

            #[cfg(target_os = "macos")]
            {
                use thread_priority::{ThreadPriority, set_current_thread_priority};

                if let Err(e) = set_current_thread_priority(ThreadPriority::Crossplatform(
                    47u8.try_into().unwrap(),
                )) {
                    error!("Failed to set QoS for thread {}: {:?}", thread_index, e);
                }
            }

            #[cfg(any(target_os = "linux", target_os = "windows"))]
            {
                if let Some(core_ids) = core_affinity::get_core_ids()
                    && let Some(core) = core_ids.get(thread_index)
                    && !core_affinity::set_for_current(*core)
                {
                    error!("Failed to set core affinity for thread {}", thread_index);
                }
            }
        })
        .build_global()
        .expect("Unable to set thread pool priority")
}
