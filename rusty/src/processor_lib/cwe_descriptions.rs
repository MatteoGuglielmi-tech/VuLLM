use anyhow::Result;
use indicatif::ProgressIterator;
use std::collections::HashMap;
use std::fs::{File, create_dir_all};
use std::io::{BufReader, BufWriter};
use std::path::Path;

use crate::processor_lib::{feature_extractor::ProcessedEntry, utils::create_progress_bar};

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct CweInfo {
    pub name: String,
    pub description: String,
}

/// A temporary struct to represent a row during CSV deserialization.
#[derive(serde::Deserialize)]
struct CweRecord {
    #[serde(rename = "CWE-ID")]
    id: String,
    #[serde(rename = "Name")]
    name: String,
    #[serde(rename = "Description")]
    description: String,
}

/// Reads a CSV file and maps CWE IDs to their name and description.
///
/// # Arguments
/// * `path` - The path to the CWE CSV file.
///
/// # Returns
/// A `HashMap` where the key is the CWE ID (e.g., "CWE-79") and the value
/// is a `CweInfo` struct containing the name and description.
pub fn load_cwe_data<P: AsRef<Path>>(path: P) -> Result<HashMap<String, CweInfo>> {
    let file = File::open(path.as_ref())?;
    let file_size = file.metadata()?.len();

    let pb = create_progress_bar(file_size, "🗺️ Building CWE IDs map");
    let reader = pb.wrap_read(file);

    // let mut reader = csv::ReaderBuilder::new().flexible(true).from_path(path)?;
    let mut reader = csv::ReaderBuilder::new().flexible(true).from_reader(reader);
    let mut cwe_map = HashMap::new();

    for result in reader.deserialize() {
        let record: CweRecord = result?;
        cwe_map.insert(
            record.id,
            CweInfo {
                name: record.name,
                description: record.description,
            },
        );
    }

    pb.finish_with_message("✅ CSV parsing complete!");

    Ok(cwe_map)
}

/// This function modifies the entries in-place.
///
/// # Arguments
/// * `cwe_map` - A reference to the HashMap containing CWE ID -> CweInfo mappings.
/// * `data` - A mutable slice of `ProcessedEntry` structs to be enriched.
pub fn enrich_data(cwe_map: &HashMap<String, CweInfo>, data: &mut [ProcessedEntry]) {
    let pb = create_progress_bar(cwe_map.len() as u64, "💰 Add CWE ID descriptions ");
    for entry in data.iter_mut().progress_with(pb) {
        entry.cwe_desc = entry
            .cwe
            .iter()
            .map(|cwe_id| {
                let lookup_key = cwe_id.trim().strip_prefix("CWE-").unwrap_or(cwe_id.trim());
                cwe_map
                    .get(lookup_key)
                    .map(|info| info.description.clone())
                    .unwrap_or_else(|| "Description not found.".to_string())
            })
            .collect();
    }
}


/// Saves the CWE map to a JSON file.
pub fn save_cwe_map_as_json<P: AsRef<Path>>(
    path: P,
    map: &HashMap<String, CweInfo>,
) -> Result<()> {
    if let Some(parent) = path.as_ref().parent() {
        create_dir_all(parent)?;
    }
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, map)?;
    Ok(())
}

/// Loads the CWE map from a JSON file.
pub fn load_cwe_map_from_json<P: AsRef<Path>>(path: P) -> Result<HashMap<String, CweInfo>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let map = serde_json::from_reader(reader)?;
    Ok(map)
}
