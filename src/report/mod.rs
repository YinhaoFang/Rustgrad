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
