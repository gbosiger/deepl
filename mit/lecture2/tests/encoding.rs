use burn::backend::Flex;
use burn::tensor::backend::BackendTypes;
use lecture2::batch::HeartBatch;
use lecture2::data::{EncodedData, HeartData, Normalization, encode_samples, normalize_heart_data};

fn patients() -> Vec<HeartData> {
    let csv = "age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target\n\
               10,0,4,20,30,1,2,40,0,1.5,2,3,reversible,1\n\
               11,1,0,21,31,0,1,41,1,2.5,3,0,normal,0\n";
    csv::Reader::from_reader(csv.as_bytes())
        .deserialize()
        .collect::<Result<_, _>>()
        .unwrap()
}

#[test]
fn flat_encoding_preserves_feature_order_and_tensor_values() {
    let patients = patients();
    let raw = patients.iter().collect::<Vec<_>>();
    let normalized = normalize_heart_data(
        &raw,
        &Normalization {
            means: [0.0; 6],
            std_devs: [1.0; 6],
        },
    );
    // This annotation checks that the concrete encoding types match Flex's associated types.
    let encoded: EncodedData<<Flex as BackendTypes>::FloatElem, <Flex as BackendTypes>::IntElem> =
        encode_samples::<f32, i32>(raw, normalized);
    let expected = vec![
        10.0, 20.0, 30.0, 40.0, 1.5, 2.0, 1.0, 0.0, // sex
        0.0, 0.0, 0.0, 0.0, 1.0, // cp
        0.0, 1.0, // fbs
        0.0, 0.0, 1.0, // restecg
        1.0, 0.0, // exang
        0.0, 0.0, 0.0, 1.0, // ca
        0.0, 0.0, 0.0, 0.0, 1.0, // thal
        11.0, 21.0, 31.0, 41.0, 2.5, 3.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    ];
    assert_eq!(encoded.inputs, expected);
    assert_eq!(encoded.targets, vec![1, 0]);

    let batch = HeartBatch::<Flex>::from_encoded(encoded, &Default::default()).unwrap();
    assert_eq!(batch.inputs.dims(), [2, 29]);
    assert_eq!(batch.targets.dims(), [2, 1]);
    assert_eq!(batch.inputs.into_data().to_vec::<f32>().unwrap(), expected);
    assert_eq!(
        batch.targets.into_data().to_vec::<i32>().unwrap(),
        vec![1, 0]
    );
    // Consuming the reference vector leaves the original records available.
    assert_eq!(patients[0].age, 10);
}

#[test]
fn encoding_supports_f64_and_i64_without_rounding_through_f32() {
    let patients = patients();
    let raw = patients.iter().collect::<Vec<_>>();
    let normalized = normalize_heart_data(
        &raw,
        &Normalization {
            means: [0.1; 6],
            std_devs: [1.0; 6],
        },
    );
    let encoded = encode_samples::<f64, i64>(raw, normalized);
    assert_eq!(encoded.inputs.len(), 58);
    assert_eq!(encoded.inputs[0], 10.0_f64 - 0.1);
    assert_eq!(encoded.targets, vec![1_i64, 0]);
}

#[test]
#[should_panic(expected = "assertion `left == right` failed")]
fn encoding_rejects_mismatched_counts_before_zip_can_truncate() {
    let patients = patients();
    let _ = encode_samples::<f32, i32>(patients.iter().collect(), Vec::new());
}
