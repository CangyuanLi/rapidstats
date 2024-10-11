pub fn create_rayon_pool(n_jobs: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_jobs)
        .build()
        .unwrap()
}
