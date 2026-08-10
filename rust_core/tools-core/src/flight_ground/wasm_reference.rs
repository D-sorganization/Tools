//! wasm-bindgen execution boundary for the compiled ground reference runtime.

use std::cell::RefCell;

use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = "runFlightToGroundReferenceV1")]
pub fn run_ground_reference_v1(
    request_json: String,
    execution_json: Option<String>,
    is_cancelled: Option<js_sys::Function>,
) -> Result<String, JsValue> {
    let callback_error = RefCell::new(None);
    let result =
        super::run_ground_reference_v1_json(&request_json, execution_json.as_deref(), || {
            callback_cancelled(is_cancelled.as_ref(), &callback_error)
        });
    if let Some(error) = callback_error.into_inner() {
        return Err(error);
    }
    result.map_err(|error| JsValue::from_str(&error.payload()))
}

fn callback_cancelled(
    callback: Option<&js_sys::Function>,
    callback_error: &RefCell<Option<JsValue>>,
) -> bool {
    let Some(callback) = callback else {
        return false;
    };
    match callback.call0(&JsValue::NULL) {
        Ok(value) => match value.as_bool() {
            Some(cancelled) => cancelled,
            None => {
                *callback_error.borrow_mut() = Some(JsValue::from_str("is_cancelled_result"));
                true
            }
        },
        Err(error) => {
            *callback_error.borrow_mut() = Some(error);
            true
        }
    }
}
