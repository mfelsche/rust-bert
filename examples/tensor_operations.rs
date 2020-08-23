use std::time::{Duration, Instant};
use tch::kind::Kind::Float;
use tch::{Device, Tensor};

fn matrix_multiply(iters: u64, input: &Tensor, weights: &Tensor) -> Duration {
    let mut duration = Duration::new(0, 0);
    for _i in 0..iters {
        let start = Instant::now();
        let _ = input.matmul(weights);
        duration = duration.checked_add(start.elapsed()).unwrap();
    }
    duration
}

fn main() {
    unsafe {
        torch_sys::dummy_cuda_dependency();
    }
    let n_iter = 15000;
    let input = Tensor::rand(&[32, 128, 512], (Float, Device::cuda_if_available()));
    let weights = Tensor::rand(&[512, 512], (Float, Device::cuda_if_available()));

    let duration = matrix_multiply(n_iter, &input, &weights);
    println!("{:?}", duration / n_iter as u32)
}
