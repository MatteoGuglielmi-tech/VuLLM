use log::LevelFilter;
use data_processing::processor_lib::logger::setup_logger;
use std::sync::Once;

static INIT: Once = Once::new();

/// This function will be called by each test to ensure the logger is initialized.
/// The `Once` guard guarantees that the setup code runs only a single time,
/// even if multiple tests call this function concurrently.
pub fn setup() {
    INIT.call_once(|| {
        setup_logger(LevelFilter::Debug).expect("Failed to initialize logger for tests.");
    });
}
