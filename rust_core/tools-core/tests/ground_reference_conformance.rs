//! Cross-runtime scientific conformance against independent analytic oracles.

use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Number, Value};
use std::collections::HashSet;
use std::fmt;
use tools_core::flight_ground::{
    canonical_result_v1_json, parse_ground_reference_execution_v1_json, parse_request_v1_json,
    run_ground_reference_v1,
};

const TEMPLATE: &str = include_str!(
    "../../../src/rate_of_closure/web/src/model/__fixtures__/ground_reference_pipeline_golden_v1.json"
);
const CORPUS: &str = include_str!(
    "../../../src/rate_of_closure/web/src/model/__fixtures__/ground_reference_conformance_v1.json"
);

struct UniqueValue(Value);

struct UniqueValueVisitor;

impl<'de> Visitor<'de> for UniqueValueVisitor {
    type Value = UniqueValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value without duplicate object keys")
    }

    fn visit_bool<E: de::Error>(self, value: bool) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Bool(value)))
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Number(Number::from(value))))
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Number(Number::from(value))))
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Self::Value, E> {
        Number::from_f64(value)
            .map(Value::Number)
            .map(UniqueValue)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::String(value.to_owned())))
    }

    fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Null))
    }

    fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Null))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
        let mut values = Vec::new();
        while let Some(UniqueValue(value)) = sequence.next_element()? {
            values.push(value);
        }
        Ok(UniqueValue(Value::Array(values)))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut object: A) -> Result<Self::Value, A::Error> {
        let mut values = Map::new();
        let mut keys = HashSet::new();
        while let Some(key) = object.next_key::<String>()? {
            if !keys.insert(key.clone()) {
                return Err(de::Error::custom(format!("duplicate JSON key: {key}")));
            }
            let UniqueValue(value) = object.next_value()?;
            values.insert(key, value);
        }
        Ok(UniqueValue(Value::Object(values)))
    }
}

impl<'de> Deserialize<'de> for UniqueValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(UniqueValueVisitor)
    }
}

fn parse_unique_json(payload: &str) -> Result<Value, serde_json::Error> {
    serde_json::from_str::<UniqueValue>(payload).map(|value| value.0)
}

fn assert_template_fixture(corpus: &Value) -> Result<(), String> {
    let expected = "ground_reference_pipeline_golden_v1.json";
    match corpus["template_fixture"].as_str() {
        Some(actual) if actual == expected => Ok(()),
        Some(actual) => Err(format!("untrusted conformance template: {actual}")),
        None => Err("conformance template_fixture must be a string".to_owned()),
    }
}

fn decode_pointer_token(token: &str) -> Result<String, String> {
    let mut decoded = String::new();
    let mut characters = token.chars();
    while let Some(character) = characters.next() {
        if character != '~' {
            decoded.push(character);
            continue;
        }
        match characters.next() {
            Some('0') => decoded.push('~'),
            Some('1') => decoded.push('/'),
            _ => return Err(format!("invalid JSON pointer escape: {token}")),
        }
    }
    Ok(decoded)
}

fn pointer_tokens(pointer: &str) -> Result<Vec<String>, String> {
    if pointer.is_empty() {
        return Ok(Vec::new());
    }
    let suffix = pointer
        .strip_prefix('/')
        .ok_or_else(|| format!("invalid JSON pointer: {pointer}"))?;
    suffix.split('/').map(decode_pointer_token).collect()
}

fn array_index(token: &str, length: usize) -> Result<usize, String> {
    let canonical = !token.is_empty()
        && token.bytes().all(|value| value.is_ascii_digit())
        && (token == "0" || !token.starts_with('0'));
    if !canonical {
        return Err(format!("noncanonical JSON pointer array index: {token}"));
    }
    let index = token
        .parse::<usize>()
        .map_err(|_| format!("invalid JSON pointer array index: {token}"))?;
    if index >= length {
        return Err(format!("JSON pointer array index is out of range: {token}"));
    }
    Ok(index)
}

fn resolve_tokens_mut<'a>(
    mut current: &'a mut Value,
    tokens: &[String],
) -> Result<&'a mut Value, String> {
    for token in tokens {
        current = match current {
            Value::Array(values) => {
                let index = array_index(token, values.len())?;
                &mut values[index]
            }
            Value::Object(values) => values
                .get_mut(token)
                .ok_or_else(|| format!("JSON pointer key does not exist: {token}"))?,
            _ => return Err(format!("JSON pointer traverses a scalar at: {token}")),
        };
    }
    Ok(current)
}

fn apply_overrides(document: &mut Value, overrides: &Value) -> Result<(), String> {
    for (pointer, replacement) in overrides.as_object().unwrap() {
        let tokens = pointer_tokens(pointer)?;
        let (leaf, parent_tokens) = tokens
            .split_last()
            .ok_or_else(|| "an override cannot replace the document root".to_owned())?;
        let parent = resolve_tokens_mut(document, parent_tokens)?;
        match parent {
            Value::Array(values) => {
                let index = array_index(leaf, values.len())?;
                values[index] = replacement.clone();
            }
            Value::Object(values) if values.contains_key(leaf) => {
                values.insert(leaf.clone(), replacement.clone());
            }
            _ => {
                return Err(format!(
                    "override does not replace an existing leaf: {pointer}"
                ))
            }
        }
    }
    Ok(())
}

fn vector(value: &Value) -> [f64; 3] {
    let values = value.as_array().unwrap();
    std::array::from_fn(|index| values[index].as_f64().unwrap())
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    (0..3).map(|index| left[index] * right[index]).sum()
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn assert_close(actual: f64, expected: f64, check: &Value) {
    let absolute = check["absolute_tolerance"].as_f64().unwrap();
    let relative = check["relative_tolerance"].as_f64().unwrap();
    let error = (actual - expected).abs();
    assert!(
        error <= absolute.max(relative * actual.abs().max(expected.abs())),
        "{}: actual={actual}, expected={expected}, error={error}",
        check["description"].as_str().unwrap()
    );
}

fn assert_rolling_constraint(result: &Value, request: &Value, check: &Value) {
    let event = &result["events"][check["event_index"].as_u64().unwrap() as usize];
    let normal = vector(&request["surface"]["normal_unit"]);
    let radius = request["ball_radius_m"].as_f64().unwrap();
    let arm = normal.map(|component| -radius * component);
    let velocity = vector(&event["velocity_after_m_s"]);
    let spin_velocity = cross(vector(&event["angular_velocity_after_rad_s"]), arm);
    let surface_velocity = vector(&request["surface"]["surface_velocity_m_s"]);
    let contact = std::array::from_fn(|index| {
        velocity[index] + spin_velocity[index] - surface_velocity[index]
    });
    let normal_speed = dot(contact, normal);
    let tangent = std::array::from_fn(|index| contact[index] - normal_speed * normal[index]);
    let slip = dot(tangent, tangent).sqrt();
    assert!(slip <= check["absolute_tolerance"].as_f64().unwrap());
}

fn kinetic_energy(event: &Value, request: &Value, after: bool) -> f64 {
    let suffix = if after { "after" } else { "before" };
    let velocity = vector(&event[format!("velocity_{suffix}_m_s")]);
    let spin = vector(&event[format!("angular_velocity_{suffix}_rad_s")]);
    let mass = request["ball_mass_kg"].as_f64().unwrap();
    let radius = request["ball_radius_m"].as_f64().unwrap();
    let factor = request["rotational_inertia_factor"].as_f64().unwrap();
    0.5 * mass * dot(velocity, velocity) + 0.5 * factor * mass * radius.powi(2) * dot(spin, spin)
}

fn assert_vector_values(actual: &Value, check: &Value) {
    let actual = actual.as_array().unwrap();
    let expected = check["expected"].as_array().unwrap();
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_close(actual.as_f64().unwrap(), expected.as_f64().unwrap(), check);
    }
}

fn assert_event_types(result: &Value, check: &Value) {
    let actual: Vec<&str> = result["events"]
        .as_array()
        .unwrap()
        .iter()
        .map(|event| event["event_type"].as_str().unwrap())
        .collect();
    let expected: Vec<&str> = check["expected"]
        .as_array()
        .unwrap()
        .iter()
        .map(|event| event.as_str().unwrap())
        .collect();
    assert_eq!(actual, expected);
}

fn assert_restitution(result: &Value, request: &Value, check: &Value) {
    let event = &result["events"][check["event_index"].as_u64().unwrap() as usize];
    let normal = vector(&request["surface"]["normal_unit"]);
    let before = dot(vector(&event["velocity_before_m_s"]), normal);
    let after = dot(vector(&event["velocity_after_m_s"]), normal);
    assert_close(after / -before, check["expected"].as_f64().unwrap(), check);
}

fn assert_impact_energy(result: &Value, request: &Value, check: &Value) {
    let event = &result["events"][check["event_index"].as_u64().unwrap() as usize];
    let tolerance = check["absolute_tolerance_j"].as_f64().unwrap();
    assert!(
        kinetic_energy(event, request, true) <= kinetic_energy(event, request, false) + tolerance
    );
}

fn assert_check(result: &Value, request: &Value, check: &Value) {
    match check["kind"].as_str().unwrap() {
        "value_equal" => assert_eq!(
            result.pointer(check["path"].as_str().unwrap()).unwrap(),
            &check["expected"]
        ),
        "scalar_close" => assert_close(
            result
                .pointer(check["path"].as_str().unwrap())
                .unwrap()
                .as_f64()
                .unwrap(),
            check["expected"].as_f64().unwrap(),
            check,
        ),
        "vector_close" => assert_vector_values(
            result.pointer(check["path"].as_str().unwrap()).unwrap(),
            check,
        ),
        "terminal_vector_close" => {
            let terminal = result["trajectory"].as_array().unwrap().last().unwrap();
            assert_vector_values(&terminal[check["field"].as_str().unwrap()], check);
        }
        "event_types_equal" => assert_event_types(result, check),
        "restitution_ratio" => assert_restitution(result, request, check),
        "rolling_constraint" => assert_rolling_constraint(result, request, check),
        "impact_energy_nonincrease" => assert_impact_energy(result, request, check),
        kind => panic!("unsupported conformance check: {kind}"),
    }
}

#[test]
fn native_reference_satisfies_shared_scientific_corpus() {
    let template = parse_unique_json(TEMPLATE).unwrap();
    let corpus = parse_unique_json(CORPUS).unwrap();
    assert_eq!(corpus["schema_version"], "ground-reference-conformance/v1");
    assert_template_fixture(&corpus).unwrap();
    for case in corpus["cases"].as_array().unwrap() {
        assert!(case["platforms"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value == "native"));
        let mut request_value = template["request"].clone();
        apply_overrides(&mut request_value, &case["request_overrides"]).unwrap();
        let request = parse_request_v1_json(&request_value.to_string()).unwrap();
        let mut execution_value = template["execution"].clone();
        execution_value["schema_version"] = template["execution_schema_version"].clone();
        let execution =
            parse_ground_reference_execution_v1_json(&execution_value.to_string()).unwrap();
        let result = run_ground_reference_v1(&request, &execution, || false).unwrap();
        let result_value: Value =
            serde_json::from_str(&canonical_result_v1_json(&result).unwrap()).unwrap();
        for check in case["checks"].as_array().unwrap() {
            assert_check(&result_value, &request_value, check);
        }
    }
}

#[test]
fn native_override_pointer_semantics_are_strict() {
    let mut document = serde_json::json!({"a/b": 1, "a~b": 2, "items": [3, 4]});
    let overrides = serde_json::json!({"/a~1b": 10, "/a~0b": 20, "/items/1": 40});
    apply_overrides(&mut document, &overrides).unwrap();
    assert_eq!(
        document,
        serde_json::json!({"a/b": 10, "a~b": 20, "items": [3, 40]})
    );

    for pointer in ["/items/-1", "/items/01", "/items/2", "/items/~2"] {
        let mut candidate = serde_json::json!({"items": [1, 2]});
        let overrides = serde_json::json!({pointer: 9});
        assert!(apply_overrides(&mut candidate, &overrides).is_err());
    }
}

#[test]
fn native_fixture_parser_rejects_duplicate_keys_at_any_depth() {
    assert!(parse_unique_json(r#"{"outer":{"value":1,"value":2}}"#).is_err());
}

#[test]
fn native_corpus_pins_the_compiled_template_reference() {
    let valid = serde_json::json!({"template_fixture": "ground_reference_pipeline_golden_v1.json"});
    assert!(assert_template_fixture(&valid).is_ok());
    let escaped = serde_json::json!({"template_fixture": "../outside.json"});
    assert!(assert_template_fixture(&escaped).is_err());
}
