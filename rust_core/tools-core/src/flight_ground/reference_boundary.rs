//! Shared strict JSON boundary used by native, PyO3, and WASM callers.

use super::reference_runtime::run_normalized_ground_reference_v1;
use super::{
    canonical_ground_reference_runtime_error_v1_json, canonical_result_v1_json,
    parse_ground_reference_execution_v1_json, parse_request_v1_json, GroundReferenceExecutionV1,
    GroundReferenceExecutionV1Error, GroundReferenceRuntimeErrorV1, GroundRequestV1Error,
    GroundResultV1Error,
};

#[derive(Debug, Clone, PartialEq)]
pub enum GroundReferenceBoundaryErrorV1 {
    Request(GroundRequestV1Error),
    Execution(GroundReferenceExecutionV1Error),
    Runtime(GroundReferenceRuntimeErrorV1),
    Result(GroundResultV1Error),
}

impl GroundReferenceBoundaryErrorV1 {
    #[must_use]
    pub fn payload(&self) -> String {
        match self {
            Self::Request(error) => error.code().to_owned(),
            Self::Execution(error) => error.code().to_owned(),
            Self::Runtime(error) => canonical_ground_reference_runtime_error_v1_json(error),
            Self::Result(error) => error.code().to_owned(),
        }
    }

    #[must_use]
    pub const fn is_cancelled(&self) -> bool {
        matches!(
            self,
            Self::Runtime(error)
                if matches!(error.code, super::GroundReferenceRuntimeCodeV1::Cancelled)
        )
    }
}

pub fn run_ground_reference_v1_json<C>(
    request_payload: &str,
    execution_payload: Option<&str>,
    is_cancelled: C,
) -> Result<String, GroundReferenceBoundaryErrorV1>
where
    C: FnMut() -> bool,
{
    let request =
        parse_request_v1_json(request_payload).map_err(GroundReferenceBoundaryErrorV1::Request)?;
    let execution = match execution_payload {
        Some(payload) => parse_ground_reference_execution_v1_json(payload)
            .map_err(GroundReferenceBoundaryErrorV1::Execution)?,
        None => GroundReferenceExecutionV1::default(),
    };
    let result = run_normalized_ground_reference_v1(&request, &execution, is_cancelled)
        .map_err(GroundReferenceBoundaryErrorV1::Runtime)?;
    canonical_result_v1_json(&result).map_err(GroundReferenceBoundaryErrorV1::Result)
}
