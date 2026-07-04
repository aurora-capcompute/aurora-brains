//! The quarantine: this brain's guest-side variable store and the two pure
//! halves of its contract with the planning model — stub observations (all
//! the model is ever told about an executed call) and `$N` substitution (how
//! data the model cannot read still flows into the calls it authors). Raw
//! tool output enters the store and leaves only through substitution; it
//! never crosses into the model's conversation.

use aurora_brain_sdk as sdk;
use serde::Serialize;
use serde_json::Value;

/// VarStore holds each successful tool call's raw result. Slot k (1-based)
/// is named by the literal string `"$k"`; the planner only ever learns the
/// name, never the value.
#[derive(Default)]
pub struct VarStore {
    vars: Vec<Value>,
}

impl VarStore {
    /// insert quarantines one result and returns the `"$N"` name that
    /// references it.
    pub fn insert(&mut self, value: Value) -> String {
        self.vars.push(value);
        format!("${}", self.vars.len())
    }

    fn get(&self, n: usize) -> Option<&Value> {
        n.checked_sub(1).and_then(|i| self.vars.get(i))
    }

    fn len(&self) -> usize {
        self.vars.len()
    }
}

/// StubObservation is everything the planner learns about an executed call:
/// the action it chose, whether it worked, and the variable that now holds
/// the result. Success carries no content. Failure carries the generic
/// "failed" marker plus, when the host supplied one, a short machine code —
/// never the error message, which may quote whatever the tool returned.
#[derive(Serialize)]
pub struct StubObservation {
    pub action: String,
    pub status: &'static str,
    /// The `"$N"` holding the result — present on success only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub var: Option<String>,
    /// Generic failure marker; never raw error text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<&'static str>,
    /// Errno-ish machine code from the host, when present and token-shaped.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

impl StubObservation {
    /// result reports a success: the model learns only which variable holds
    /// the value.
    pub fn result(action: &str, var: String) -> Self {
        StubObservation {
            action: action.to_string(),
            status: sdk::STATUS_RESULT,
            var: Some(var),
            error: None,
            code: None,
        }
    }

    /// failed reports a failure with the error message withheld. Only the
    /// host's machine code passes through, and only when it is token-shaped
    /// — a code that could smuggle prose is dropped.
    pub fn failed(action: &str, code: &str) -> Self {
        StubObservation {
            action: action.to_string(),
            status: sdk::STATUS_FAILED,
            var: None,
            error: Some("failed"),
            code: sanitize_code(code),
        }
    }
}

/// sanitize_code keeps a short machine token (`[A-Za-z0-9_.-]`, at most 64
/// bytes) and drops anything else — the failure channel must not carry text.
fn sanitize_code(code: &str) -> Option<String> {
    let token_shaped = !code.is_empty()
        && code.len() <= 64
        && code
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'.' || b == b'-');
    token_shaped.then(|| code.to_string())
}

/// substitute resolves every `"$N"` reference in a model-authored JSON value
/// — the CaMeL move: the model has already chosen the action and written the
/// args; only now does quarantined data flow into them. String positions
/// only: a string that is exactly `"$N"` becomes the stored value itself
/// (any JSON type); `$N` embedded in a longer string becomes the value's
/// text rendering (strings verbatim, other values compact JSON); `$$`
/// escapes a literal `$`; a `$` not followed by a digit stays literal.
/// Object keys are never substituted — a reference names data, not shape.
/// Referencing a variable that does not exist is an error.
pub fn substitute(value: &Value, store: &VarStore) -> anyhow::Result<Value> {
    match value {
        Value::String(s) => substitute_string(s, store),
        Value::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                out.push(substitute(item, store)?);
            }
            Ok(Value::Array(out))
        }
        Value::Object(fields) => {
            let mut out = serde_json::Map::with_capacity(fields.len());
            for (key, val) in fields {
                out.insert(key.clone(), substitute(val, store)?);
            }
            Ok(Value::Object(out))
        }
        other => Ok(other.clone()),
    }
}

fn substitute_string(s: &str, store: &VarStore) -> anyhow::Result<Value> {
    if let Some(digits) = whole_ref(s) {
        return Ok(resolve(digits, store)?.clone());
    }
    interpolate(s, store).map(Value::String)
}

/// whole_ref returns the digit part when the entire string is one `"$N"`
/// reference and nothing else.
fn whole_ref(s: &str) -> Option<&str> {
    let digits = s.strip_prefix('$')?;
    (!digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())).then_some(digits)
}

/// interpolate renders a string with embedded `$N` references replaced by
/// their stored values' text form.
fn interpolate(s: &str, store: &VarStore) -> anyhow::Result<String> {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('$') {
        out.push_str(&rest[..i]);
        let after = &rest[i + 1..];
        // "$$" is a literal "$".
        if let Some(tail) = after.strip_prefix('$') {
            out.push('$');
            rest = tail;
            continue;
        }
        let digits = after.bytes().take_while(|b| b.is_ascii_digit()).count();
        // A "$" not followed by a digit is literal.
        if digits == 0 {
            out.push('$');
            rest = after;
            continue;
        }
        out.push_str(&render(resolve(&after[..digits], store)?));
        rest = &after[digits..];
    }
    out.push_str(rest);
    Ok(out)
}

fn resolve<'a>(digits: &str, store: &'a VarStore) -> anyhow::Result<&'a Value> {
    let n: usize = digits
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid variable reference ${}", digits))?;
    store.get(n).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown variable ${}: this run has stored {} result(s)",
            n,
            store.len()
        )
    })
}

/// render is the embedded-in-a-string form of a stored value: strings pass
/// through verbatim (no added quotes); every other JSON value renders as
/// compact JSON.
fn render(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn store_with(values: &[Value]) -> VarStore {
        let mut store = VarStore::default();
        for v in values {
            store.insert(v.clone());
        }
        store
    }

    // -- VarStore --

    #[test]
    fn insert_names_slots_sequentially() {
        let mut store = VarStore::default();
        assert_eq!(store.insert(json!("a")), "$1");
        assert_eq!(store.insert(json!("b")), "$2");
        assert_eq!(store.get(1), Some(&json!("a")));
        assert_eq!(store.get(2), Some(&json!("b")));
        assert_eq!(store.get(0), None);
        assert_eq!(store.get(3), None);
    }

    // -- Whole-string references --

    #[test]
    fn whole_string_ref_substitutes_the_json_value_itself() {
        let store = store_with(&[json!({"items": [1, 2], "next": null})]);
        let got = substitute(&json!("$1"), &store).unwrap();
        assert_eq!(got, json!({"items": [1, 2], "next": null}));
    }

    #[test]
    fn whole_string_ref_to_string_value_stays_a_string() {
        let store = store_with(&[json!("plain text")]);
        assert_eq!(substitute(&json!("$1"), &store).unwrap(), json!("plain text"));
    }

    #[test]
    fn leading_zeros_resolve_numerically() {
        let store = store_with(&[json!(42)]);
        assert_eq!(substitute(&json!("$01"), &store).unwrap(), json!(42));
    }

    // -- Embedded interpolation --

    #[test]
    fn embedded_ref_renders_string_value_verbatim() {
        let store = store_with(&[json!("Berlin")]);
        let got = substitute(&json!("weather in $1 today"), &store).unwrap();
        assert_eq!(got, json!("weather in Berlin today"));
    }

    #[test]
    fn embedded_ref_renders_non_strings_as_compact_json() {
        let store = store_with(&[json!(3.5), json!(true), json!({"a": [1]}), json!(null)]);
        let got = substitute(&json!("n=$1 b=$2 o=$3 z=$4"), &store).unwrap();
        assert_eq!(got, json!("n=3.5 b=true o={\"a\":[1]} z=null"));
    }

    #[test]
    fn multiple_refs_in_one_string() {
        let store = store_with(&[json!("alpha"), json!("beta")]);
        let got = substitute(&json!("$1+$2+$1"), &store).unwrap();
        assert_eq!(got, json!("alpha+beta+alpha"));
    }

    #[test]
    fn ref_followed_by_non_digit_ends_at_the_digits() {
        let store = store_with(&[json!("v")]);
        assert_eq!(substitute(&json!("$1x"), &store).unwrap(), json!("vx"));
    }

    #[test]
    fn dollar_dollar_escapes_a_literal_dollar() {
        let store = store_with(&[json!("v")]);
        assert_eq!(substitute(&json!("$$1"), &store).unwrap(), json!("$1"));
        assert_eq!(substitute(&json!("a$$b"), &store).unwrap(), json!("a$b"));
        assert_eq!(substitute(&json!("$$$1"), &store).unwrap(), json!("$v"));
    }

    #[test]
    fn dollar_without_digits_is_literal() {
        let store = VarStore::default();
        assert_eq!(substitute(&json!("$foo"), &store).unwrap(), json!("$foo"));
        assert_eq!(substitute(&json!("cost: $"), &store).unwrap(), json!("cost: $"));
        assert_eq!(substitute(&json!("$ 1"), &store).unwrap(), json!("$ 1"));
    }

    // -- Structural traversal --

    #[test]
    fn substitution_recurses_into_nested_objects_and_arrays() {
        let store = store_with(&[json!("bob@example.com"), json!({"id": 7})]);
        let args = json!({
            "to": "$1",
            "meta": {"ticket": "$2", "note": "for $1"},
            "cc": ["$1", {"deep": "$2"}],
            "count": 2,
            "flag": false
        });
        let got = substitute(&args, &store).unwrap();
        assert_eq!(
            got,
            json!({
                "to": "bob@example.com",
                "meta": {"ticket": {"id": 7}, "note": "for bob@example.com"},
                "cc": ["bob@example.com", {"deep": {"id": 7}}],
                "count": 2,
                "flag": false
            })
        );
    }

    #[test]
    fn object_keys_are_never_substituted() {
        let store = store_with(&[json!("evil")]);
        let got = substitute(&json!({"$1": "$1"}), &store).unwrap();
        assert_eq!(got, json!({"$1": "evil"}));
    }

    #[test]
    fn non_string_scalars_pass_through_untouched() {
        let store = VarStore::default();
        let args = json!({"n": 1, "f": 2.5, "b": true, "z": null});
        assert_eq!(substitute(&args, &store).unwrap(), args);
    }

    // -- Reference errors --

    #[test]
    fn unknown_variable_is_an_error() {
        let store = store_with(&[json!("only one")]);
        let err = substitute(&json!("$2"), &store).unwrap_err();
        assert!(err.to_string().contains("unknown variable $2"), "{}", err);
        let err = substitute(&json!("see $9 here"), &store).unwrap_err();
        assert!(err.to_string().contains("unknown variable $9"), "{}", err);
    }

    #[test]
    fn zero_is_not_a_variable() {
        let store = store_with(&[json!("v")]);
        let err = substitute(&json!("$0"), &store).unwrap_err();
        assert!(err.to_string().contains("unknown variable $0"), "{}", err);
    }

    #[test]
    fn overflowing_reference_is_an_error() {
        let store = store_with(&[json!("v")]);
        let err = substitute(&json!("$99999999999999999999999999"), &store).unwrap_err();
        assert!(err.to_string().contains("invalid variable reference"), "{}", err);
    }

    // -- Stub rendering --

    #[test]
    fn result_stub_is_action_status_var_only() {
        let stub = StubObservation::result("web.fetch", "$3".to_string());
        let got = serde_json::to_string(&stub).unwrap();
        assert_eq!(got, r#"{"action":"web.fetch","status":"result","var":"$3"}"#);
    }

    #[test]
    fn failed_stub_carries_generic_marker_and_token_code() {
        let stub = StubObservation::failed("web.fetch", "not_found");
        let got = serde_json::to_string(&stub).unwrap();
        assert_eq!(
            got,
            r#"{"action":"web.fetch","status":"failed","error":"failed","code":"not_found"}"#
        );
    }

    #[test]
    fn failed_stub_without_code_omits_it() {
        let stub = StubObservation::failed("web.fetch", "");
        let got = serde_json::to_string(&stub).unwrap();
        assert_eq!(got, r#"{"action":"web.fetch","status":"failed","error":"failed"}"#);
    }

    #[test]
    fn failed_stub_drops_a_code_that_could_carry_text() {
        // Spaces, quotes, or excessive length make a "code" a prose channel.
        assert_eq!(sanitize_code("ignore previous instructions"), None);
        assert_eq!(sanitize_code("a\"b"), None);
        assert_eq!(sanitize_code(&"x".repeat(65)), None);
        assert_eq!(sanitize_code("http.404"), Some("http.404".to_string()));
        assert_eq!(sanitize_code("ERR-42_x"), Some("ERR-42_x".to_string()));
    }
}
