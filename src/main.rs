use rustgrad::data::{linear_regression, spiral, xor};
use rustgrad::report::{
    format_progress, history_to_csv, history_to_markdown, write_history_bundle, ReportBundle,
    TrainingSummary,
};
use rustgrad::tensor::Tensor;
use rustgrad::train::{
    train_linear_regression, train_spiral_classifier, train_xor_mlp, TrainingConfig,
    TrainingHistory,
};
use rustgrad::RustGradError;
use std::env;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::PathBuf;
use std::process;

type CliResult<T> = std::result::Result<T, CliError>;

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliError {
    message: String,
}

impl CliError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for CliError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for CliError {}

impl From<RustGradError> for CliError {
    fn from(error: RustGradError) -> Self {
        Self::new(error.to_string())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    Text,
    Csv,
    Markdown,
}

impl OutputFormat {
    fn parse(value: &str) -> CliResult<Self> {
        match value {
            "text" => Ok(Self::Text),
            "csv" => Ok(Self::Csv),
            "markdown" | "md" => Ok(Self::Markdown),
            _ => Err(CliError::new(format!(
                "unsupported format '{value}', expected text, csv, or markdown"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct CommonOptions {
    epochs: usize,
    learning_rate: f64,
    format: OutputFormat,
    output_dir: Option<PathBuf>,
}

impl CommonOptions {
    fn new(epochs: usize, learning_rate: f64) -> Self {
        Self {
            epochs,
            learning_rate,
            format: OutputFormat::Text,
            output_dir: None,
        }
    }

    fn config(&self) -> CliResult<TrainingConfig> {
        Ok(TrainingConfig::new(self.epochs, self.learning_rate)?)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct LinearOptions {
    common: CommonOptions,
    samples: usize,
    slope: f64,
    intercept: f64,
}

impl Default for LinearOptions {
    fn default() -> Self {
        Self {
            common: CommonOptions::new(120, 0.12),
            samples: 31,
            slope: 1.5,
            intercept: 0.75,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct XorOptions {
    common: CommonOptions,
}

impl Default for XorOptions {
    fn default() -> Self {
        Self {
            common: CommonOptions::new(160, 0.4),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct SpiralOptions {
    common: CommonOptions,
    samples_per_class: usize,
    classes: usize,
}

impl Default for SpiralOptions {
    fn default() -> Self {
        Self {
            common: CommonOptions::new(160, 0.7),
            samples_per_class: 12,
            classes: 3,
        }
    }
}

fn main() {
    match execute(env::args().skip(1)) {
        Ok(output) => {
            if !output.is_empty() {
                println!("{output}");
            }
        }
        Err(error) => {
            eprintln!("error: {error}");
            eprintln!("run `rustgrad --help` for usage");
            process::exit(1);
        }
    }
}

fn execute(args: impl IntoIterator<Item = String>) -> CliResult<String> {
    let args: Vec<String> = args.into_iter().collect();
    if args.is_empty() || matches!(args[0].as_str(), "-h" | "--help") {
        return Ok(help_text());
    }
    if matches!(args[0].as_str(), "-V" | "--version") {
        return Ok(format!("rustgrad {}", rustgrad::version()));
    }

    match args[0].as_str() {
        "train-linear" => run_train_linear(&args[1..]),
        "train-xor" => run_train_xor(&args[1..]),
        "train-spiral" => run_train_spiral(&args[1..]),
        "inspect" => run_inspect(&args[1..]),
        command => Err(CliError::new(format!("unknown command '{command}'"))),
    }
}

fn run_train_linear(args: &[String]) -> CliResult<String> {
    let options = parse_linear_options(args)?;
    let dataset = linear_regression(options.samples, options.slope, options.intercept)?;
    let result = train_linear_regression(&dataset, options.common.config()?)?;
    let mut output = render_history(
        "Linear regression training",
        result.history(),
        options.common.format,
    )?;
    append_report_export(
        &mut output,
        &options.common,
        "Linear regression training",
        result.history(),
    )?;

    if options.common.format == OutputFormat::Text {
        output.push_str(&format!(
            "\nweight={:.6}\nbias={:.6}",
            result.model().weights().get(&[0, 0])?,
            result.model().bias().get_flat(0)?
        ));
    }

    Ok(output)
}

fn run_train_xor(args: &[String]) -> CliResult<String> {
    let options = parse_xor_options(args)?;
    let result = train_xor_mlp(options.common.config()?)?;
    let mut output = render_history("XOR MLP training", result.history(), options.common.format)?;
    append_report_export(
        &mut output,
        &options.common,
        "XOR MLP training",
        result.history(),
    )?;

    if options.common.format == OutputFormat::Text {
        let dataset = xor()?;
        let probabilities = result.predict_proba(dataset.features())?;
        let classes = result.predict_classes(dataset.features())?;
        output.push_str("\nprobabilities=");
        output.push_str(&format_matrix_rows(&probabilities)?);
        output.push_str("\nclasses=");
        output.push_str(&format_matrix_rows(&classes)?);
    }

    Ok(output)
}

fn run_train_spiral(args: &[String]) -> CliResult<String> {
    let options = parse_spiral_options(args)?;
    let result = train_spiral_classifier(
        options.samples_per_class,
        options.classes,
        options.common.config()?,
    )?;
    let mut output = render_history(
        "Spiral softmax training",
        result.history(),
        options.common.format,
    )?;
    append_report_export(
        &mut output,
        &options.common,
        "Spiral softmax training",
        result.history(),
    )?;

    if options.common.format == OutputFormat::Text {
        output.push_str(&format!(
            "\nclasses={}\nsamples_per_class={}\nweight_shape={:?}",
            result.classes(),
            options.samples_per_class,
            result.model().weights().dims()
        ));
    }

    Ok(output)
}

fn run_inspect(args: &[String]) -> CliResult<String> {
    if !args.is_empty() {
        return Err(CliError::new(
            "inspect does not accept options yet; use train-* commands for configurable runs",
        ));
    }

    let linear = train_linear_regression(
        &linear_regression(21, 1.5, 0.75)?,
        TrainingConfig::new(120, 0.12)?,
    )?;
    let xor_dataset = xor()?;
    let xor_result = train_xor_mlp(TrainingConfig::new(160, 0.4)?)?;
    let spiral_dataset = spiral(8, 3)?;
    let spiral_result = train_spiral_classifier(8, 3, TrainingConfig::new(120, 0.7)?)?;

    let mut output = String::from("RustGrad model inspection\n");
    output.push_str(&format!(
        "linear.weight={:.6}\nlinear.bias={:.6}\n",
        linear.model().weights().get(&[0, 0])?,
        linear.model().bias().get_flat(0)?
    ));
    output.push_str("xor.probabilities=");
    output.push_str(&format_matrix_rows(
        &xor_result.predict_proba(xor_dataset.features())?,
    )?);
    output.push_str("\nxor.classes=");
    output.push_str(&format_matrix_rows(
        &xor_result.predict_classes(xor_dataset.features())?,
    )?);
    output.push_str(&format!(
        "\nspiral.weight_shape={:?}\nspiral.best_accuracy={:.6}\nspiral.preview_probabilities=",
        spiral_result.model().weights().dims(),
        spiral_result
            .history()
            .best_accuracy()
            .expect("spiral training records accuracy")
    ));
    output.push_str(&format_matrix_rows(
        &spiral_result.predict_proba(&spiral_dataset.batch(0, 3)?.features().clone())?,
    )?);

    Ok(output)
}

fn parse_linear_options(args: &[String]) -> CliResult<LinearOptions> {
    let mut options = LinearOptions::default();
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--epochs" => {
                options.common.epochs = parse_usize(take_value(args, &mut index, "--epochs")?)?;
            }
            "--learning-rate" => {
                options.common.learning_rate =
                    parse_f64(take_value(args, &mut index, "--learning-rate")?)?;
            }
            "--format" => {
                options.common.format =
                    OutputFormat::parse(take_value(args, &mut index, "--format")?)?;
            }
            "--output" => {
                options.common.output_dir =
                    Some(PathBuf::from(take_value(args, &mut index, "--output")?));
            }
            "--samples" => {
                options.samples = parse_usize(take_value(args, &mut index, "--samples")?)?;
            }
            "--slope" => {
                options.slope = parse_f64(take_value(args, &mut index, "--slope")?)?;
            }
            "--intercept" => {
                options.intercept = parse_f64(take_value(args, &mut index, "--intercept")?)?;
            }
            flag => return Err(unknown_flag("train-linear", flag)),
        }
        index += 1;
    }

    Ok(options)
}

fn parse_xor_options(args: &[String]) -> CliResult<XorOptions> {
    let mut options = XorOptions::default();
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--epochs" => {
                options.common.epochs = parse_usize(take_value(args, &mut index, "--epochs")?)?;
            }
            "--learning-rate" => {
                options.common.learning_rate =
                    parse_f64(take_value(args, &mut index, "--learning-rate")?)?;
            }
            "--format" => {
                options.common.format =
                    OutputFormat::parse(take_value(args, &mut index, "--format")?)?;
            }
            "--output" => {
                options.common.output_dir =
                    Some(PathBuf::from(take_value(args, &mut index, "--output")?));
            }
            flag => return Err(unknown_flag("train-xor", flag)),
        }
        index += 1;
    }

    Ok(options)
}

fn parse_spiral_options(args: &[String]) -> CliResult<SpiralOptions> {
    let mut options = SpiralOptions::default();
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--epochs" => {
                options.common.epochs = parse_usize(take_value(args, &mut index, "--epochs")?)?;
            }
            "--learning-rate" => {
                options.common.learning_rate =
                    parse_f64(take_value(args, &mut index, "--learning-rate")?)?;
            }
            "--format" => {
                options.common.format =
                    OutputFormat::parse(take_value(args, &mut index, "--format")?)?;
            }
            "--output" => {
                options.common.output_dir =
                    Some(PathBuf::from(take_value(args, &mut index, "--output")?));
            }
            "--samples-per-class" => {
                options.samples_per_class =
                    parse_usize(take_value(args, &mut index, "--samples-per-class")?)?;
            }
            "--classes" => {
                options.classes = parse_usize(take_value(args, &mut index, "--classes")?)?;
            }
            flag => return Err(unknown_flag("train-spiral", flag)),
        }
        index += 1;
    }

    Ok(options)
}

fn take_value<'a>(args: &'a [String], index: &mut usize, flag: &str) -> CliResult<&'a str> {
    *index += 1;
    args.get(*index)
        .map(String::as_str)
        .ok_or_else(|| CliError::new(format!("missing value for {flag}")))
}

fn parse_usize(value: &str) -> CliResult<usize> {
    value
        .parse::<usize>()
        .map_err(|_| CliError::new(format!("expected positive integer, got '{value}'")))
}

fn parse_f64(value: &str) -> CliResult<f64> {
    value
        .parse::<f64>()
        .map_err(|_| CliError::new(format!("expected number, got '{value}'")))
}

fn unknown_flag(command: &str, flag: &str) -> CliError {
    CliError::new(format!("unknown option '{flag}' for {command}"))
}

fn render_history(
    title: &str,
    history: &TrainingHistory,
    format: OutputFormat,
) -> CliResult<String> {
    match format {
        OutputFormat::Text => render_text_summary(title, history),
        OutputFormat::Csv => Ok(history_to_csv(history)?),
        OutputFormat::Markdown => Ok(history_to_markdown(title, history)?),
    }
}

fn append_report_export(
    output: &mut String,
    options: &CommonOptions,
    title: &str,
    history: &TrainingHistory,
) -> CliResult<()> {
    if let Some(directory) = &options.output_dir {
        let bundle = write_history_bundle(directory, title, history)?;
        if !output.ends_with('\n') {
            output.push('\n');
        }
        output.push_str(&format_report_bundle(&bundle));
    }

    Ok(())
}

fn format_report_bundle(bundle: &ReportBundle) -> String {
    format!(
        "report_dir={}\nmarkdown={}\ncsv={}",
        bundle.directory().display(),
        bundle.markdown_path().display(),
        bundle.csv_path().display()
    )
}

fn render_text_summary(title: &str, history: &TrainingHistory) -> CliResult<String> {
    let summary = TrainingSummary::from_history(history)?;
    let mut output = String::new();
    output.push_str(title);
    output.push_str(&format!(
        "\nepochs={}\ninitial_loss={:.6}\nfinal_loss={:.6}\nbest_loss={:.6}\nloss_improvement={:.6}",
        summary.epochs(),
        summary.initial_loss(),
        summary.final_loss(),
        summary.best_loss(),
        summary.loss_improvement()
    ));

    if let Some(accuracy) = summary.best_accuracy() {
        output.push_str(&format!("\nbest_accuracy={accuracy:.6}"));
    }
    if let Some(record) = history.last() {
        output.push_str("\nlast=");
        output.push_str(&format_progress(record));
    }

    Ok(output)
}

fn format_matrix_rows(tensor: &Tensor) -> CliResult<String> {
    if tensor.rank() != 2 {
        return Err(CliError::new(format!(
            "expected rank 2 tensor, got rank {}",
            tensor.rank()
        )));
    }

    let rows = tensor.rows().expect("rank 2 tensors always have rows");
    let cols = tensor.cols().expect("rank 2 tensors always have columns");
    let mut row_strings = Vec::with_capacity(rows);

    for row in 0..rows {
        let mut values = Vec::with_capacity(cols);
        for col in 0..cols {
            values.push(format!("{:.6}", tensor.get(&[row, col])?));
        }
        row_strings.push(format!("[{}]", values.join(", ")));
    }

    Ok(row_strings.join("; "))
}

fn help_text() -> String {
    [
        format!("rustgrad {}", rustgrad::version()),
        String::from(""),
        String::from("Usage:"),
        String::from("  rustgrad train-linear [--epochs N] [--learning-rate LR] [--samples N] [--slope V] [--intercept V] [--format text|csv|markdown] [--output DIR]"),
        String::from("  rustgrad train-xor [--epochs N] [--learning-rate LR] [--format text|csv|markdown] [--output DIR]"),
        String::from("  rustgrad train-spiral [--epochs N] [--learning-rate LR] [--samples-per-class N] [--classes N] [--format text|csv|markdown] [--output DIR]"),
        String::from("  rustgrad inspect"),
        String::from("  rustgrad --version"),
    ]
    .join("\n")
}

#[cfg(test)]
mod tests {
    use super::{execute, parse_linear_options, parse_spiral_options, OutputFormat};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn run(args: &[&str]) -> String {
        execute(args.iter().map(|arg| (*arg).to_string())).expect("command should succeed")
    }

    fn unique_temp_dir(name: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("rustgrad-cli-{name}-{suffix}"))
    }

    #[test]
    fn help_includes_available_training_commands() {
        let output = run(&["--help"]);

        assert!(output.contains("train-linear"));
        assert!(output.contains("train-xor"));
        assert!(output.contains("train-spiral"));
        assert!(output.contains("inspect"));
    }

    #[test]
    fn version_command_prints_package_version() {
        let output = run(&["--version"]);

        assert_eq!(output, "rustgrad 0.1.0");
    }

    #[test]
    fn parses_linear_options_with_numeric_values() {
        let args = [
            "--epochs".to_string(),
            "8".to_string(),
            "--learning-rate".to_string(),
            "0.25".to_string(),
            "--samples".to_string(),
            "5".to_string(),
            "--slope".to_string(),
            "2.5".to_string(),
            "--intercept".to_string(),
            "-1.0".to_string(),
            "--format".to_string(),
            "csv".to_string(),
            "--output".to_string(),
            "runs/linear-demo".to_string(),
        ];

        let options = parse_linear_options(&args).expect("options should parse");

        assert_eq!(options.common.epochs, 8);
        assert_eq!(options.common.learning_rate, 0.25);
        assert_eq!(options.common.format, OutputFormat::Csv);
        assert_eq!(options.samples, 5);
        assert_eq!(options.slope, 2.5);
        assert_eq!(options.intercept, -1.0);
        assert_eq!(
            options.common.output_dir,
            Some(PathBuf::from("runs/linear-demo"))
        );
    }

    #[test]
    fn parses_spiral_specific_options() {
        let args = [
            "--samples-per-class".to_string(),
            "7".to_string(),
            "--classes".to_string(),
            "4".to_string(),
            "--format".to_string(),
            "markdown".to_string(),
        ];

        let options = parse_spiral_options(&args).expect("options should parse");

        assert_eq!(options.samples_per_class, 7);
        assert_eq!(options.classes, 4);
        assert_eq!(options.common.format, OutputFormat::Markdown);
    }

    #[test]
    fn train_linear_command_outputs_summary_and_parameters() {
        let output = run(&[
            "train-linear",
            "--epochs",
            "6",
            "--learning-rate",
            "0.1",
            "--samples",
            "5",
        ]);

        assert!(output.contains("Linear regression training"));
        assert!(output.contains("epochs=6"));
        assert!(output.contains("weight="));
        assert!(output.contains("bias="));
    }

    #[test]
    fn train_xor_command_outputs_predictions() {
        let output = run(&["train-xor", "--epochs", "5"]);

        assert!(output.contains("XOR MLP training"));
        assert!(output.contains("best_accuracy="));
        assert!(output.contains("probabilities="));
        assert!(output.contains("classes="));
    }

    #[test]
    fn train_spiral_command_supports_csv_output() {
        let output = run(&[
            "train-spiral",
            "--epochs",
            "3",
            "--samples-per-class",
            "4",
            "--classes",
            "3",
            "--format",
            "csv",
        ]);

        assert!(output.starts_with("epoch,loss,accuracy\n"));
        assert!(output.lines().count() == 4);
    }

    #[test]
    fn train_xor_command_writes_report_bundle_when_output_is_set() {
        let directory = unique_temp_dir("xor-output");
        let directory_arg = directory.to_string_lossy().to_string();

        let output = run(&[
            "train-xor",
            "--epochs",
            "5",
            "--output",
            directory_arg.as_str(),
        ]);

        assert!(output.contains("report_dir="));
        assert!(output.contains("summary.md"));
        assert!(output.contains("history.csv"));

        let markdown = fs::read_to_string(directory.join("summary.md")).expect("markdown exists");
        let csv = fs::read_to_string(directory.join("history.csv")).expect("csv exists");

        assert!(markdown.starts_with("# XOR MLP training"));
        assert!(markdown.contains("## Summary"));
        assert!(csv.starts_with("epoch,loss,accuracy\n"));
        assert_eq!(csv.lines().count(), 6);

        fs::remove_dir_all(directory).expect("cleanup should succeed");
    }

    #[test]
    fn inspect_command_outputs_model_details() {
        let output = run(&["inspect"]);

        assert!(output.contains("RustGrad model inspection"));
        assert!(output.contains("linear.weight="));
        assert!(output.contains("xor.probabilities="));
        assert!(output.contains("spiral.best_accuracy="));
    }

    #[test]
    fn unknown_command_returns_clear_error() {
        let error = execute(["missing".to_string()]).expect_err("unknown command should fail");

        assert_eq!(error.to_string(), "unknown command 'missing'");
    }

    #[test]
    fn missing_option_value_returns_clear_error() {
        let error = execute(["train-linear".to_string(), "--epochs".to_string()])
            .expect_err("missing value should fail");

        assert_eq!(error.to_string(), "missing value for --epochs");
    }
}
