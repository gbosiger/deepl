use anyhow::{ensure, Result};
use num_traits::FromPrimitive;
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use serde::de::Error;
use serde::Deserialize;
use serde::Deserializer;
use serde::Serialize;
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
    #[serde(deserialize_with = "deserialize_exang")]
    pub exang: [u8; 2],
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

fn deserialize_exang<'de, D>(deserializer: D) -> Result<[u8; 2], D::Error>
where
    D: Deserializer<'de>,
{
    let val = u8::deserialize(deserializer)?;
    match val {
        0 => Ok([1, 0]),
        1 => Ok([0, 1]),
        _ => Err(D::Error::custom("exang must be 0 or 1")),
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

#[derive(Debug)]
pub struct HeartNormalizedData {
    age: f64,
    trestbps: f64,
    chol: f64,
    thalach: f64,
    oldpeak: f64,
    slope: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Normalization {
    pub means: [f64; 6],
    pub std_devs: [f64; 6],
}

// Calculate means and std deviations
pub fn fit_normalization(data: &[&HeartData]) -> Normalization {
    assert!(!data.is_empty());

    let means = data
        .iter()
        .map(|row| {
            [
                f64::from(row.age),
                f64::from(row.trestbps),
                f64::from(row.chol),
                f64::from(row.thalach),
                f64::from(row.oldpeak),
                f64::from(row.slope),
            ]
        })
        .fold([0.0_f64; 6], |sum, row| {
            [
                sum[0] + row[0],
                sum[1] + row[1],
                sum[2] + row[2],
                sum[3] + row[3],
                sum[4] + row[4],
                sum[5] + row[5],
            ]
        })
        .map(|sum| sum / data.len() as f64);

    let std_devs = data
        .iter()
        .map(|row| {
            [
                (row.age as f64 - means[0]).powi(2),
                (row.trestbps as f64 - means[1]).powi(2),
                (row.chol as f64 - means[2]).powi(2),
                (row.thalach as f64 - means[3]).powi(2),
                (row.oldpeak as f64 - means[4]).powi(2),
                (row.slope as f64 - means[5]).powi(2),
            ]
        })
        .fold([0.0_f64; 6], |mut sum, row| {
            for i in 0..6 {
                sum[i] += row[i];
            }
            sum
        })
        .map(|sum| {
            let std_dev = (sum / data.len() as f64).sqrt();
            // Constant features normalize to zero without division by zero.
            if std_dev == 0.0 {
                1.0
            } else {
                std_dev
            }
        });

    Normalization { means, std_devs }
}

// Normalize slices for training
pub fn normalize_heart_data(
    data: &[&HeartData],
    normalization: &Normalization,
) -> Vec<HeartNormalizedData> {
    assert!(!data.is_empty());

    data.iter()
        .map(|row| HeartNormalizedData {
            age: ((row.age as f64) - normalization.means[0]) / normalization.std_devs[0],
            trestbps: (row.trestbps as f64 - normalization.means[1]) / normalization.std_devs[1],
            chol: (row.chol as f64 - normalization.means[2]) / normalization.std_devs[2],
            thalach: (row.thalach as f64 - normalization.means[3]) / normalization.std_devs[3],
            oldpeak: (row.oldpeak as f64 - normalization.means[4]) / normalization.std_devs[4],
            slope: (row.slope as f64 - normalization.means[5]) / normalization.std_devs[5],
        })
        .collect()
}

#[derive(Debug)]
pub struct EncodedData<F, I> {
    pub inputs: Vec<F>,
    pub targets: Vec<I>,
}

pub fn encode_samples<F, I>(
    raw: Vec<&HeartData>,
    normalized: Vec<HeartNormalizedData>,
) -> EncodedData<F, I>
where
    F: FromPrimitive + From<u8>,
    I: From<u8>,
{
    assert_eq!(raw.len(), normalized.len());
    let mut result = EncodedData {
        inputs: Vec::with_capacity(29 * raw.len()),
        targets: Vec::with_capacity(raw.len()),
    };
    for (raw, normalized) in raw.into_iter().zip(normalized) {
        result.inputs.extend(
            [
                normalized.age,
                normalized.trestbps,
                normalized.chol,
                normalized.thalach,
                normalized.oldpeak,
                normalized.slope,
            ]
            .map(|value| F::from_f64(value).expect("Normalized input must be convertible")),
        );
        result.inputs.extend(raw.sex.map(F::from));
        result.inputs.extend(raw.cp.map(F::from));
        result.inputs.extend(raw.fbs.map(F::from));
        result.inputs.extend(raw.restecg.map(F::from));
        result.inputs.extend(raw.exang.map(F::from));
        result.inputs.extend(raw.ca.map(F::from));
        result.inputs.extend(raw.thal.map(F::from));
        result.targets.push(I::from(raw.target));
    }
    result
}

// This order matches the numeric arrays used during normalization.
const NUMERIC_FEATURES: [&str; 6] = ["age", "trestbps", "chol", "thalach", "oldpeak", "slope"];

// CSV rows have scalar fields; the in-memory Normalization keeps its arrays.
#[derive(Debug, Serialize, Deserialize)]
struct NormalizationRow {
    feature: String,
    mean: f64,
    std_dev: f64,
}

pub fn write_normalization_params(path: &Path, normalization: &Normalization) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    for (i, feature) in NUMERIC_FEATURES.iter().enumerate() {
        writer.serialize(NormalizationRow {
            feature: (*feature).to_owned(),
            mean: normalization.means[i],
            std_dev: normalization.std_devs[i],
        })?;
    }
    writer.flush()?;
    Ok(())
}

pub fn read_normalization_params(path: &Path) -> Result<Normalization> {
    let mut reader = csv::Reader::from_path(path)?;
    let rows: Vec<NormalizationRow> = reader.deserialize().collect::<Result<_, _>>()?;

    ensure!(
        rows.len() == 6,
        "Normalization CSV must contain exactly six rows"
    );

    // Fixed order: age, trestbps, chol, thalach, oldpeak, slope.
    let mut normalization = Normalization {
        means: [0.0; 6],
        std_devs: [0.0; 6],
    };
    for (i, row) in rows.iter().enumerate() {
        normalization.means[i] = row.mean;
        normalization.std_devs[i] = row.std_dev;
    }
    Ok(normalization)
}
