//! Python bindings for the section parser: `filing_signals._parser`.

mod core;
mod entities;

use std::io::Read;

use flate2::read::GzDecoder;
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use rayon::prelude::*;

fn to_dict<'py>(py: Python<'py>, doc: &core::Document) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("n_lines", doc.n_lines)?;
    out.set_item("n_chars", doc.n_chars)?;
    for (name, s) in &doc.sections {
        let d = PyDict::new(py);
        d.set_item("status", s.status)?;
        d.set_item("start_line", s.start_line)?;
        d.set_item("end_line", s.end_line)?;
        d.set_item("text", &s.text)?;
        out.set_item(*name, d)?;
    }
    Ok(out)
}

/// Parse one document's raw bytes. Same output as textparse.parse_document.
#[pyfunction]
fn parse_document<'py>(py: Python<'py>, raw: &Bound<'py, PyBytes>) -> PyResult<Bound<'py, PyDict>> {
    let bytes = raw.as_bytes().to_vec();
    let doc = py.detach(|| core::parse_document(&bytes));
    to_dict(py, &doc)
}

/// Clean lines, exactly as textparse.to_lines.
#[pyfunction]
fn to_lines(py: Python<'_>, raw: &Bound<'_, PyBytes>) -> Vec<String> {
    let bytes = raw.as_bytes().to_vec();
    py.detach(|| core::to_lines(&bytes))
}

/// Read and parse many gzipped documents in parallel, without the GIL.
/// Returns one dict per path, in input order.
#[pyfunction]
fn parse_gz_files<'py>(py: Python<'py>, paths: Vec<String>) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let parsed: Vec<Result<core::Document, String>> = py.detach(|| {
        paths
            .par_iter()
            .map(|path| {
                let file = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
                let mut raw = Vec::new();
                GzDecoder::new(file)
                    .read_to_end(&mut raw)
                    .map_err(|e| format!("{path}: {e}"))?;
                Ok(core::parse_document(&raw))
            })
            .collect()
    });
    parsed
        .into_iter()
        .map(|r| r.map_err(PyIOError::new_err).and_then(|doc| to_dict(py, &doc)))
        .collect()
}

#[pymodule]
fn _parser(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_document, m)?)?;
    m.add_function(wrap_pyfunction!(to_lines, m)?)?;
    m.add_function(wrap_pyfunction!(parse_gz_files, m)?)?;
    Ok(())
}
