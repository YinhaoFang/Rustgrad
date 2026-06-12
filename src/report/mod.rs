//! Training report export utilities.

use crate::train::{TrainingHistory, TrainingRecord};
use crate::{Result, RustGradError};

/// Summary statistics derived from a training history.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainingSummary {
    epochs: usize,
    initial_loss: f64,
    final_loss: f64,
    best_loss: f64,
    best_accuracy: Option<f64>,
}

impl TrainingSummary {
    /// Creates a summary from a non-empty training history.
    pub fn from_history(history: &TrainingHistory) -> Result<Self> {
        ensure_non_empty(history)?;

        Ok(Self {
            epochs: history.len(),
            initial_loss: history
                .initial_loss()
                .expect("non-empty history has an initial loss"),
            final_loss: history
                .final_loss()
                .expect("non-empty history has a final loss"),
            best_loss: history
                .best_loss()
                .expect("non-empty history has a best loss"),
            best_accuracy: history.best_accuracy(),
        })
    }

    /// Returns the number of recorded epochs.
    #[must_use]
    pub fn epochs(&self) -> usize {
        self.epochs
    }

    /// Returns the first recorded loss.
    #[must_use]
    pub fn initial_loss(&self) -> f64 {
        self.initial_loss
    }

    /// Returns the final recorded loss.
    #[must_use]
    pub fn final_loss(&self) -> f64 {
        self.final_loss
    }

    /// Returns the lowest recorded loss.
    #[must_use]
    pub fn best_loss(&self) -> f64 {
        self.best_loss
    }

    /// Returns the best recorded accuracy, if any record includes accuracy.
    #[must_use]
    pub fn best_accuracy(&self) -> Option<f64> {
        self.best_accuracy
    }

    /// Returns the absolute loss improvement from first to final record.
    #[must_use]
    pub fn loss_improvement(&self) -> f64 {
        self.initial_loss - self.final_loss
    }

    /// Returns true when the final loss is lower than the initial loss.
    #[must_use]
    pub fn loss_decreased(&self) -> bool {
        self.final_loss < self.initial_loss
    }
}

/// Formats a compact Markdown summary for reports.
pub fn history_to_markdown(title: &str, history: &TrainingHistory) -> Result<String> {
    validate_title(title)?;
    let summary = TrainingSummary::from_history(history)?;

    let mut output = String::new();
    output.push_str(&format!("# {title}\n\n"));
    output.push_str("## Summary\n\n");
    output.push_str(&format!("- Epochs: {}\n", summary.epochs()));
    output.push_str(&format!(
        "- Initial loss: {}\n",
        format_metric(summary.initial_loss())
    ));
    output.push_str(&format!(
        "- Final loss: {}\n",
        format_metric(summary.final_loss())
    ));
    output.push_str(&format!(
        "- Best loss: {}\n",
        format_metric(summary.best_loss())
    ));
    output.push_str(&format!(
        "- Loss improvement: {}\n",
        format_metric(summary.loss_improvement())
    ));

    if let Some(accuracy) = summary.best_accuracy() {
        output.push_str(&format!("- Best accuracy: {}\n", format_metric(accuracy)));
    }

    output.push_str("\n## History\n\n");
    output.push_str("| epoch | loss | accuracy |\n");
    output.push_str("| ---: | ---: | ---: |\n");
    for record in history.records() {
        output.push_str(&format!(
            "| {} | {} | {} |\n",
            record.epoch(),
            format_metric(record.loss()),
            format_optional_metric(record.accuracy())
        ));
    }

    Ok(output)
}

/// Formats training history as CSV text.
pub fn history_to_csv(history: &TrainingHistory) -> Result<String> {
    ensure_non_empty(history)?;

    let mut output = String::from("epoch,loss,accuracy\n");
    for record in history.records() {
        output.push_str(&format!(
            "{},{},{}\n",
            record.epoch(),
            format_metric(record.loss()),
            format_optional_metric(record.accuracy())
        ));
    }

    Ok(output)
}

/// Formats one progress line for terminal output.
#[must_use]
pub fn format_progress(record: &TrainingRecord) -> String {
    match record.accuracy() {
        Some(accuracy) => format!(
            "epoch={} loss={} accuracy={}",
            record.epoch(),
            format_metric(record.loss()),
            format_metric(accuracy)
        ),
        None => format!(
            "epoch={} loss={}",
            record.epoch(),
            format_metric(record.loss())
        ),
    }
}

fn ensure_non_empty(history: &TrainingHistory) -> Result<()> {
    if history.is_empty() {
        Err(RustGradError::InvalidArgument {
            name: "history",
            reason: "training history must not be empty".to_string(),
        })
    } else {
        Ok(())
    }
}

fn validate_title(title: &str) -> Result<()> {
    if title.trim().is_empty() {
        Err(RustGradError::InvalidArgument {
            name: "title",
            reason: "report title must not be empty".to_string(),
        })
    } else {
        Ok(())
    }
}

fn format_optional_metric(value: Option<f64>) -> String {
    value.map_or_else(String::new, format_metric)
}

fn format_metric(value: f64) -> String {
    format!("{value:.6}")
}

#[cfg(test)]
mod tests {
    use super::{format_progress, history_to_csv, history_to_markdown, TrainingSummary};
    use crate::train::{TrainingHistory, TrainingRecord};
    use crate::RustGradError;

    const EPSILON: f64 = 1e-12;

    fn sample_history() -> TrainingHistory {
        TrainingHistory::from_records(vec![
            TrainingRecord::new(1, 2.0, Some(0.25)).expect("valid first record"),
            TrainingRecord::new(2, 1.25, Some(0.5)).expect("valid second record"),
            TrainingRecord::new(3, 0.75, Some(0.875)).expect("valid third record"),
        ])
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < EPSILON,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn summary_extracts_history_statistics() {
        let history = sample_history();

        let summary = TrainingSummary::from_history(&history).expect("summary should compute");

        assert_eq!(summary.epochs(), 3);
        assert_close(summary.initial_loss(), 2.0);
        assert_close(summary.final_loss(), 0.75);
        assert_close(summary.best_loss(), 0.75);
        assert_eq!(summary.best_accuracy(), Some(0.875));
        assert_close(summary.loss_improvement(), 1.25);
        assert!(summary.loss_decreased());
    }

    #[test]
    fn summary_detects_non_decreasing_loss() {
        let history = TrainingHistory::from_records(vec![
            TrainingRecord::new(1, 0.5, None).expect("valid first record"),
            TrainingRecord::new(2, 0.75, None).expect("valid second record"),
        ]);

        let summary = TrainingSummary::from_history(&history).expect("summary should compute");

        assert!(!summary.loss_decreased());
        assert_close(summary.loss_improvement(), -0.25);
        assert_eq!(summary.best_accuracy(), None);
    }

    #[test]
    fn summary_rejects_empty_history() {
        let error = TrainingSummary::from_history(&TrainingHistory::new())
            .expect_err("empty history should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "history",
                reason: "training history must not be empty".to_string(),
            }
        );
    }

    #[test]
    fn markdown_report_contains_summary_and_history_table() {
        let history = sample_history();

        let markdown =
            history_to_markdown("XOR training", &history).expect("markdown should format");

        assert!(markdown.starts_with("# XOR training\n\n"));
        assert!(markdown.contains("## Summary"));
        assert!(markdown.contains("- Epochs: 3"));
        assert!(markdown.contains("- Initial loss: 2.000000"));
        assert!(markdown.contains("- Final loss: 0.750000"));
        assert!(markdown.contains("- Best loss: 0.750000"));
        assert!(markdown.contains("- Loss improvement: 1.250000"));
        assert!(markdown.contains("- Best accuracy: 0.875000"));
        assert!(markdown.contains("| epoch | loss | accuracy |"));
        assert!(markdown.contains("| 1 | 2.000000 | 0.250000 |"));
        assert!(markdown.contains("| 3 | 0.750000 | 0.875000 |"));
    }

    #[test]
    fn markdown_report_allows_records_without_accuracy() {
        let history = TrainingHistory::from_records(vec![
            TrainingRecord::new(1, 1.0, None).expect("valid first record"),
            TrainingRecord::new(2, 0.5, None).expect("valid second record"),
        ]);

        let markdown =
            history_to_markdown("Linear training", &history).expect("markdown should format");

        assert!(!markdown.contains("- Best accuracy:"));
        assert!(markdown.contains("| 1 | 1.000000 |  |"));
        assert!(markdown.contains("| 2 | 0.500000 |  |"));
    }

    #[test]
    fn markdown_report_rejects_empty_title() {
        let history = sample_history();

        let error = history_to_markdown(" ", &history).expect_err("empty title should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "title",
                reason: "report title must not be empty".to_string(),
            }
        );
    }

    #[test]
    fn markdown_report_rejects_empty_history() {
        let error = history_to_markdown("Empty", &TrainingHistory::new())
            .expect_err("empty history should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "history",
                reason: "training history must not be empty".to_string(),
            }
        );
    }

    #[test]
    fn csv_report_formats_records_with_header() {
        let history = sample_history();

        let csv = history_to_csv(&history).expect("csv should format");

        assert_eq!(
            csv,
            "epoch,loss,accuracy\n1,2.000000,0.250000\n2,1.250000,0.500000\n3,0.750000,0.875000\n"
        );
    }

    #[test]
    fn csv_report_leaves_missing_accuracy_blank() {
        let history = TrainingHistory::from_records(vec![
            TrainingRecord::new(1, 1.0, None).expect("valid first record"),
            TrainingRecord::new(2, 0.5, Some(0.8)).expect("valid second record"),
        ]);

        let csv = history_to_csv(&history).expect("csv should format");

        assert_eq!(
            csv,
            "epoch,loss,accuracy\n1,1.000000,\n2,0.500000,0.800000\n"
        );
    }

    #[test]
    fn csv_report_rejects_empty_history() {
        let error = history_to_csv(&TrainingHistory::new()).expect_err("empty history should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "history",
                reason: "training history must not be empty".to_string(),
            }
        );
    }

    #[test]
    fn progress_line_includes_accuracy_when_present() {
        let record = TrainingRecord::new(7, 0.125, Some(0.9375)).expect("valid record");

        assert_eq!(
            format_progress(&record),
            "epoch=7 loss=0.125000 accuracy=0.937500"
        );
    }

    #[test]
    fn progress_line_omits_accuracy_when_missing() {
        let record = TrainingRecord::new(7, 0.125, None).expect("valid record");

        assert_eq!(format_progress(&record), "epoch=7 loss=0.125000");
    }
}
