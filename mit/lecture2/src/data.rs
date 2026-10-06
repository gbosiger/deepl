//! TODO: implement the data handling yourself.
//
// TODO 1: Inspect data/heart.csv and decide how to represent a raw row in Rust.
// TODO 2: Load the CSV and report the row count and a few example rows.
// TODO 3: Separate the target from the input features.
// TODO 4: Decide which features are numeric and which are categorical.
// TODO 5: Create reproducible training, validation, and test splits.
// TODO 6: Fit normalization statistics using the training data.
// TODO 7: Implement one-hot encoding with a consistent category/feature order.
// TODO 8: Combine the processed features into 29-value input vectors.
// TODO 9: Decide how to store and reload preprocessing information for prediction.
// TODO 10: Decide how to handle invalid rows and unfamiliar categories.

use serde::de::Error;
use serde::Deserialize;
use std::path::Path;

// The CSV also contains "1" and "2"; keep them as distinct categories.
#[derive(Debug, Deserialize)]
pub enum ThalKind {
    #[serde(rename = "1")]
    Code1,
    #[serde(rename = "2")]
    Code2,
    #[serde(rename = "normal")]
    Normal,
    #[serde(rename = "fixed")]
    Fixed,
    #[serde(rename = "reversible")]
    Reversible,
}

#[derive(Debug, Deserialize)]
pub struct HeartData {
    pub age: u8,
    #[serde(deserialize_with = "deserialize_sex")]
    pub sex: [u8; 2],
    pub cp: u8,
    pub trestbps: u16,
    pub chol: u16,
    pub fbs: u8,
    pub restecg: u8,
    pub thalach: u16,
    pub exang: u8,
    pub oldpeak: f32,
    pub slope: u8,
    pub ca: u8,
    pub thal: ThalKind,
    pub target: u8,
}

fn deserialize_sex<'de, D>(deserializer: D) -> Result<[u8; 2], D::Error>
where
    D: serde::Deserializer<'de>,
{
    let val = u8::deserialize(deserializer)?;
    match val {
        0 => Ok([1, 0]),
        1 => Ok([0, 1]),
        _ => Err(D::Error::custom("sex must be 0 or 1")),
    }
}

pub fn read_heart_data(path: &Path) -> Result<Vec<HeartData>, csv::Error> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut entries = Vec::new();

    for result in reader.deserialize::<HeartData>() {
        let entry = result?;
        entries.push(entry);
    }

    Ok(entries)
}
