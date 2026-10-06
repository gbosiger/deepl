//! TODO: implement the data handling yourself.
//
// TODO 1: Inspect data/heart.csv and decide how to represent a raw row in Rust.
// TODO 2: Load the CSV and report the row count and a few example rows.
// TODO 3: Separate the target from the input features.
// TODO 4: Decide which features are numeric and which are categorical.
// TODO 5: Create reproducible training, validation, and test splits.
//
// TODO 6: Fit normalization statistics using the training data.
// TODO 7: Implement one-hot encoding with a consistent category/feature order.
// TODO 8: Combine the processed features into 29-value input vectors.
// TODO 9: Decide how to store and reload preprocessing information for prediction.
// TODO 10: Decide how to handle invalid rows and unfamiliar categories.

use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use serde::de::Error;
use serde::Deserialize;
use serde::Deserializer;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct HeartData {
    pub age: u8,
    #[serde(deserialize_with = "deserialize_sex")]
    pub sex: [u8; 2],
    #[serde(deserialize_with = "deserialize_cp")]
    pub cp: [u8; 5],
    pub trestbps: u16,
    pub chol: u16,
    #[serde(deserialize_with = "deserialize_fbs")]
    pub fbs: [u8; 2],
    #[serde(deserialize_with = "deserialize_restecg")]
    pub restecg: [u8; 3],
    pub thalach: u16,
    pub exang: u8,
    pub oldpeak: f32,
    pub slope: u8,
    #[serde(deserialize_with = "deserialize_ca")]
    pub ca: [u8; 4],
    #[serde(deserialize_with = "deserialize_thal")]
    pub thal: [u8; 5],
    pub target: u8,
}

fn deserialize_sex<'de, D>(deserializer: D) -> Result<[u8; 2], D::Error>
where
    D: Deserializer<'de>,
{
    let val = u8::deserialize(deserializer)?;
    match val {
        0 => Ok([1, 0]),
        1 => Ok([0, 1]),
        _ => Err(D::Error::custom("sex must be 0 or 1")),
    }
}

fn deserialize_cp<'de, D>(deserializer: D) -> Result<[u8; 5], D::Error>
where
    D: Deserializer<'de>,
{
    let val = u8::deserialize(deserializer)?;
    match val {
        0 => Ok([1, 0, 0, 0, 0]),
        1 => Ok([0, 1, 0, 0, 0]),
        2 => Ok([0, 0, 1, 0, 0]),
        3 => Ok([0, 0, 0, 1, 0]),
        4 => Ok([0, 0, 0, 0, 1]),
        _ => Err(D::Error::custom("cp must be between 0..=4")),
    }
}

fn deserialize_fbs<'de, D>(deserializer: D) -> Result<[u8; 2], D::Error>
where
    D: Deserializer<'de>,
{
    let val = u8::deserialize(deserializer)?;
    match val {
        0 => Ok([1, 0]),
        1 => Ok([0, 1]),
        _ => Err(D::Error::custom("fbs must be 0 or 1")),
    }
}

fn deserialize_restecg<'de, D>(deserializer: D) -> Result<[u8; 3], D::Error>
where
    D: Deserializer<'de>,
{
    let val = u8::deserialize(deserializer)?;
    match val {
        0 => Ok([1, 0, 0]),
        1 => Ok([0, 1, 0]),
        2 => Ok([0, 0, 1]),
        _ => Err(D::Error::custom("restecg must be between 0..=2")),
    }
}

fn deserialize_ca<'de, D>(deserializer: D) -> Result<[u8; 4], D::Error>
where
    D: Deserializer<'de>,
{
    let val = u8::deserialize(deserializer)?;
    match val {
        0 => Ok([1, 0, 0, 0]),
        1 => Ok([0, 1, 0, 0]),
        2 => Ok([0, 0, 1, 0]),
        3 => Ok([0, 0, 0, 1]),
        _ => Err(D::Error::custom("ca must be between 0..=3")),
    }
}

fn deserialize_thal<'de, D>(deserializer: D) -> Result<[u8; 5], D::Error>
where
    D: Deserializer<'de>,
{
    let val = String::deserialize(deserializer)?;
    match val.as_str() {
        "1" => Ok([1, 0, 0, 0, 0]),
        "2" => Ok([0, 1, 0, 0, 0]),
        "normal" => Ok([0, 0, 1, 0, 0]),
        "fixed" => Ok([0, 0, 0, 1, 0]),
        "reversible" => Ok([0, 0, 0, 0, 1]),
        _ => Err(D::Error::custom(
            "thal must be between 1 2, normal, fixed, or reversible",
        )),
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

// Prepare slices for training and testing by shuffling and spitting
pub fn shuffle_and_split_heart_data(data: &[HeartData]) -> (Vec<&HeartData>, Vec<&HeartData>) {
    let mut shuffled_data: Vec<&HeartData> = data.iter().collect();
    let mut rng = StdRng::seed_from_u64(42);
    shuffled_data.shuffle(&mut rng);

    let count = data.len() * 80 / 100;
    // We take part of the shuffled_data vector here
    // It is a copy, which I do not like, but otherwise the user would need to hold the shuffled vector
    // as well
    let test_data = shuffled_data.split_off(count);

    (shuffled_data, test_data)
}

// Normalize slices for training
//pub fn normalize_heart_data(mut data: &[HeartData]) {}
