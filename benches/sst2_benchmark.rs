#[macro_use]
extern crate criterion;

use criterion::Criterion;
use rust_bert::pipelines::sentiment::{ss2_processor, SentimentModel};
use rust_bert::pipelines::sequence_classification::SequenceClassificationConfig;
use std::env;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tch::Device;
use torch_sys::dummy_cuda_dependency;

static BATCH_SIZE: usize = 64;

fn create_sentiment_model() -> SentimentModel {
    let config = SequenceClassificationConfig {
        device: Device::cuda_if_available(),
        ..Default::default()
    };
    SentimentModel::new(config).unwrap()
}

fn sst2_forward_pass(iters: u64, model: &SentimentModel, sst2_data: &Vec<String>) -> Duration {
    let mut duration = Duration::new(0, 0);
    let batch_size = BATCH_SIZE;
    let mut output = vec![];
    for _i in 0..iters {
        let start = Instant::now();
        for batch in sst2_data.chunks(batch_size) {
            output.push(
                model.predict(
                    batch
                        .iter()
                        .map(|v| v.as_str())
                        .collect::<Vec<&str>>()
                        .as_slice(),
                ),
            );
        }
        duration = duration.checked_add(start.elapsed()).unwrap();
    }
    duration
}

fn sst2_load_model(iters: u64) -> Duration {
    let mut duration = Duration::new(0, 0);
    for _i in 0..iters {
        let start = Instant::now();
        let config = SequenceClassificationConfig {
            device: Device::cuda_if_available(),
            ..Default::default()
        };
        let _ = SentimentModel::new(config).unwrap();
        duration = duration.checked_add(start.elapsed()).unwrap();
    }
    duration
}

fn bench_sst2(c: &mut Criterion) {
    //    Set-up classifier
    let model = create_sentiment_model();
    unsafe {
        torch_sys::dummy_cuda_dependency();
    }
    //    Define input
    let mut sst2_path = PathBuf::from(env::var("SST2_PATH")
        .expect("Please set the \"squad_dataset\" environment variable pointing to the SQuAD dataset folder"));
    sst2_path.push("train.tsv");
    let mut inputs = ss2_processor(sst2_path).unwrap();
    inputs.truncate(5000);

    c.bench_function("SST2 forward pass", |b| {
        b.iter_custom(|iters| sst2_forward_pass(iters, &model, &inputs))
    });

    c.bench_function("Load model", |b| {
        b.iter_custom(|iters| sst2_load_model(iters))
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_sst2
}

criterion_main!(benches);
