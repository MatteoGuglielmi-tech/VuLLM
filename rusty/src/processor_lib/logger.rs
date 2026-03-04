use colored::*;
use fern;
use log::{Level, LevelFilter};

/// Configures and initializes a global logger with custom colors and format.
///
/// # Arguments
///
/// * `level` - The minimum log level to display (e.g., `LevelFilter::Debug`).
///
/// # Returns
///
/// * `Result<(), fern::InitError>` - An empty Ok on success, or an error if
///   the logger fails to initialize (e.g., if one is already set).
pub fn setup_logger(level: LevelFilter) -> Result<(), fern::InitError> {
    fern::Dispatch::new()
        .format(move |out, message, record| {
            // First, create the full, uncolored log string.
            let formatted_line = format!(
                "| {level: <8} | {timestamp} | {target}:{line} | \"{message}\"",
                level = record.level(),
                timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
                target = record.target(),
                line = record.line().unwrap_or(0),
                message = message,
            );

            // Now, apply a color to the entire string based on the log level.
            let colored_line = match record.level() {
                Level::Error => formatted_line.red(),
                Level::Warn => formatted_line.yellow(),
                Level::Info => formatted_line.magenta(), // Magenta for Purple
                Level::Debug => formatted_line.blue(),
                Level::Trace => formatted_line.dimmed(),
            };

            out.finish(format_args!("{}", colored_line));
        })
        // Set the minimum log level to display.
        .level(level)
        // Direct the output to stdout.
        .chain(std::io::stdout())
        // Apply the configuration to the global logger.
        .apply()?;

    Ok(())
}
