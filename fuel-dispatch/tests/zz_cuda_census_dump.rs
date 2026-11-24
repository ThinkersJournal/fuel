// TEMPORARY (6.2 oracle, not committed): dump the CUDA binding table at the base commit.
#![cfg(feature = "cuda")]
use fuel_dispatch::kernel::KernelBindingTable;
use fuel_ir::probe::BackendId;

#[test]
fn dump_cuda_census() {
    let mut table = KernelBindingTable::new();
    fuel_dispatch::dispatch::register_cuda_kernels(&mut table);
    fuel_dispatch::baracuda_dispatch::register_baracuda_cuda_kernels(&mut table);
    let mut lines: Vec<String> = table
        .iter_entries()
        .filter(|(_, _, backend, _)| *backend == BackendId::Cuda)
        .map(|(op, dtypes, backend, entry)| {
            format!(
                "{op:?} {dtypes:?} {backend:?} src={} generic={} rev={:016x}",
                entry.kernel_source, entry.is_generic, entry.kernel_revision_hash
            )
        })
        .collect();
    lines.sort();
    let path = std::env::var_os("FUEL_CENSUS_OUT").expect("FUEL_CENSUS_OUT");
    std::fs::write(path, format!("{}\n", lines.join("\n"))).unwrap();
    assert!(lines.len() > 100, "only {} entries", lines.len());
}
