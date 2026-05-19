use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use data_processor_core::{convert, inspect, preview};

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Inspect {
        path: PathBuf,
    },
    Preview {
        path: PathBuf,
        #[arg(long, default_value_t = 100)]
        rows: usize,
        #[arg(long)]
        columns: Option<String>,
    },
    Convert {
        input: PathBuf,
        output: PathBuf,
        #[arg(long = "format", default_value = "csv")]
        output_format: String,
        #[arg(long)]
        columns: Option<String>,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Commands::Inspect { path } => inspect(&path).and_then(to_json),
        Commands::Preview {
            path,
            rows,
            columns,
        } => preview(&path, rows, columns.as_deref()).and_then(to_json),
        Commands::Convert {
            input,
            output,
            output_format,
            columns,
        } => convert(&input, &output, &output_format, columns.as_deref()).and_then(to_json),
    };

    match result {
        Ok(json) => {
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(2)
        }
    }
}

fn to_json<T: serde::Serialize>(
    value: T,
) -> Result<String, data_processor_core::DataProcessorError> {
    serde_json::to_string(&value).map_err(|error| {
        data_processor_core::DataProcessorError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            error,
        ))
    })
}
