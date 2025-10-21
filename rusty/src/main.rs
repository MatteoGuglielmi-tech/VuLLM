use anyhow::Result;
use clap::{Arg, ArgAction, Command};
use log::{LevelFilter, debug};
use std::path::PathBuf;

use data_processing::processor_lib::cwe_descriptions::{enrich_data, load_cwe_data};
use data_processing::processor_lib::cwe_descriptions::{
    load_cwe_map_from_json, save_cwe_map_as_json,
};
use data_processing::processor_lib::thread_pool_builder::setup_global_thread_pool;
use data_processing::processor_lib::{feature_extractor::FeatureExtractor, logger::setup_logger};

fn main() -> Result<()> {
    setup_logger(LevelFilter::Debug).expect("Issue encountered when initializing the logger");

    let args = Command::new("Balancer")
        .version("1.0")
        .about("A Rust program to:\n\t- filter out empty, comments only, invalid and C++ functions\n\t- parse, clean and fix dataset samples.")
        .arg(Arg::new("input_fp")
            .short('i')
            .long("input_fp")
            .value_parser(clap::value_parser!(PathBuf))
            .required(true)
            .help("Absolute/relative path to the raw dataset jsonl file."))
        .arg(Arg::new("output_fp")
            .long("output_fp")
            .short('o')
            .value_parser(clap::value_parser!(PathBuf))
            .required(true)
            .help("Absolute/relative path wherein saving the balanced dataset."))
        .arg(Arg::new("mitre_cwes")
            .long("mitre_cwes")
            .short('m')
            .value_parser(clap::value_parser!(PathBuf))
            .required(true)
            .help("Absolute/relative path to the MITRE CSV file containing all CWE IDs metadata."))
        .arg(Arg::new("cached_cwe")
            .long("cached_cwe")
            .short('c')
            .value_parser(clap::value_parser!(PathBuf))
            .default_value("./src/cache/cache.json")
            .help("Absolute/relative path to the MITRE CSV file containing all CWE IDs metadata."))
        .arg(Arg::new("debug")
            .long("debug")
            .action(ArgAction::SetTrue)
            .help("Activate debug CLI logs."))
        .get_matches();

    let input_fp = args
        .get_one::<PathBuf>("input_fp")
        .expect("`input_file_path` is required");
    let output_fp = args
        .get_one::<PathBuf>("output_fp")
        .expect("`output_file_path` is required");
    let mitre_fp = args
        .get_one::<PathBuf>("mitre_cwes")
        .expect("`mitre_cwes` argument is required");
    let cache_fp = args
        .get_one::<PathBuf>("cached_cwe")
        .expect("`cached_cwe` argument is required");

    if args.get_flag("debug") {
        debug!("Input file: {}", input_fp.display());
        debug!("Output file: {}", output_fp.display());
        debug!("MITRE file: {}", mitre_fp.display());
    }

    setup_global_thread_pool();

    let cwe_map = match load_cwe_map_from_json(cache_fp) {
        Ok(map) => {
            println!(" ⬇️ CWE map loaded from JSON cache '{:#?}'.", cache_fp);
            map
        }
        Err(_) => {
            println!("⚠️ Cache not found. Parsing CSV file...");
            let map = load_cwe_data(mitre_fp)?;

            save_cwe_map_as_json(cache_fp, &map)?;
            println!(
                "💾 CWE map created and saved to JSON cache '{:#?}'.",
                cache_fp
            );
            map
        }
    };

    let mut enriched_data = FeatureExtractor::extract_features(input_fp)?;
    enrich_data(&cwe_map, &mut enriched_data);
    FeatureExtractor::count_vulnerable(&enriched_data)?;
    FeatureExtractor::save_dataset(&enriched_data, output_fp)?;
    // println!("Miss clang cache rate: {:#?}", data_processing::processor_lib::processor::CACHE_MISSES);
    // println!("Hit clang cache rate: {:#?}", data_processing::processor_lib::processor::CACHE_HITS);

    Ok(())
}
