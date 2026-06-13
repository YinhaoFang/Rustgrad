use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

fn rustgrad_command() -> Command {
    Command::new(env!("CARGO_BIN_EXE_rustgrad"))
}

fn run_success(args: &[&str]) -> String {
    let output = rustgrad_command()
        .args(args)
        .output()
        .expect("command should run");

    assert_success(&output);
    String::from_utf8(output.stdout).expect("stdout should be utf8")
}

fn run_failure(args: &[&str]) -> Output {
    let output = rustgrad_command()
        .args(args)
        .output()
        .expect("command should run");

    assert!(
        !output.status.success(),
        "expected command to fail, stdout={}, stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "expected success, stdout={}, stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn unique_temp_dir(name: &str) -> PathBuf {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("rustgrad-cli-integration-{name}-{suffix}"))
}

#[test]
fn help_command_lists_training_commands() {
    let stdout = run_success(&["--help"]);

    assert!(stdout.contains("rustgrad 0.1.0"));
    assert!(stdout.contains("train-linear"));
    assert!(stdout.contains("train-xor"));
    assert!(stdout.contains("train-spiral"));
    assert!(stdout.contains("inspect"));
}

#[test]
fn train_linear_command_outputs_summary_and_parameters() {
    let stdout = run_success(&[
        "train-linear",
        "--epochs",
        "8",
        "--learning-rate",
        "0.1",
        "--samples",
        "7",
    ]);

    assert!(stdout.contains("Linear regression training"));
    assert!(stdout.contains("epochs=8"));
    assert!(stdout.contains("initial_loss="));
    assert!(stdout.contains("final_loss="));
    assert!(stdout.contains("weight="));
    assert!(stdout.contains("bias="));
}

#[test]
fn train_spiral_csv_output_is_machine_readable() {
    let stdout = run_success(&[
        "train-spiral",
        "--epochs",
        "4",
        "--samples-per-class",
        "4",
        "--classes",
        "3",
        "--format",
        "csv",
    ]);

    let lines: Vec<&str> = stdout.lines().filter(|line| !line.is_empty()).collect();
    assert_eq!(lines.len(), 5);
    assert_eq!(lines[0], "epoch,loss,accuracy");
    assert!(lines[1].starts_with("1,"));
    assert!(lines[4].starts_with("4,"));
}

#[test]
fn train_xor_output_option_writes_report_bundle() {
    let directory = unique_temp_dir("xor-report");
    let directory_arg = directory.to_string_lossy().to_string();

    let stdout = run_success(&[
        "train-xor",
        "--epochs",
        "6",
        "--output",
        directory_arg.as_str(),
    ]);

    assert!(stdout.contains("XOR MLP training"));
    assert!(stdout.contains("report_dir="));
    assert!(stdout.contains("summary.md"));
    assert!(stdout.contains("history.csv"));

    let markdown_path = directory.join("summary.md");
    let csv_path = directory.join("history.csv");
    let markdown = fs::read_to_string(&markdown_path).expect("summary should exist");
    let csv = fs::read_to_string(&csv_path).expect("history should exist");

    assert!(markdown.starts_with("# XOR MLP training"));
    assert!(markdown.contains("## Summary"));
    assert!(markdown.contains("## History"));
    assert_eq!(csv.lines().count(), 7);
    assert!(csv.starts_with("epoch,loss,accuracy\n"));

    fs::remove_dir_all(directory).expect("cleanup should succeed");
}

#[test]
fn invalid_command_exits_with_error_message() {
    let output = run_failure(&["does-not-exist"]);
    let stderr = String::from_utf8(output.stderr).expect("stderr should be utf8");

    assert!(stderr.contains("error: unknown command 'does-not-exist'"));
    assert!(stderr.contains("run `rustgrad --help` for usage"));
}

#[test]
fn missing_option_value_exits_with_error_message() {
    let output = run_failure(&["train-linear", "--epochs"]);
    let stderr = String::from_utf8(output.stderr).expect("stderr should be utf8");

    assert!(stderr.contains("error: missing value for --epochs"));
    assert!(stderr.contains("run `rustgrad --help` for usage"));
}
