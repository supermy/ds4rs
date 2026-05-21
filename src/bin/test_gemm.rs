fn main() {
    let device = ds4rs::GpuTensor::new_device(0).unwrap();
    let cublas = ds4rs::CublasHandle::new().unwrap();

    let m = 2;
    let n = 3;
    let k = 2;

    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b_data: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

    let a_cpu = ds4rs::CpuTensor::new(
        bytemuck::cast_slice(&a_data).to_vec(),
        vec![m, k],
        ds4rs::DType::FP32,
    );
    let b_cpu = ds4rs::CpuTensor::new(
        bytemuck::cast_slice(&b_data).to_vec(),
        vec![k, n],
        ds4rs::DType::FP32,
    );

    let a_gpu = ds4rs::GpuTensor::from_host(device.clone(), &a_cpu).unwrap();
    let b_gpu = ds4rs::GpuTensor::from_host(device.clone(), &b_cpu).unwrap();
    let mut c_gpu = ds4rs::GpuTensor::zeros(device.clone(), vec![m, n], ds4rs::DType::FP32).unwrap();

    cublas.gemm_f32_nn_strided_batched(
        m, n, k,
        &a_gpu, &b_gpu, &mut c_gpu,
        (k) as i64, (n) as i64, (n) as i64,
        1,
        1.0, 0.0,
    ).unwrap();

    let c_cpu = c_gpu.to_host().unwrap();
    let c_data: &[f32] = bytemuck::cast_slice(&c_cpu.data);

    eprintln!("A = {:?}", a_data);
    eprintln!("B = {:?}", b_data);
    eprintln!("C = {:?}", c_data);

    let expected: Vec<f32> = (0..m).flat_map(|i| {
        (0..n).map(move |j| {
            (0..k).map(|kk| a_data[i * k + kk] * b_data[kk * n + j]).sum::<f32>()
        })
    }).collect();
    eprintln!("Expected C = A * B = {:?}", expected);
}
