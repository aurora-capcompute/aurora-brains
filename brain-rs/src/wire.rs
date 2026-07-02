//! ABI v3 envelope codec: the messages of capcompute's sys/wire/envelope.proto
//! in proto3 wire format, hand-rolled and reflection-free — the Rust mirror of
//! the Go codec the host runs. The guest only ever encodes a Syscall and
//! decodes a Response, so that is all this module implements. Unknown fields
//! are skipped on decode (the schema-evolution contract); interoperability is
//! pinned by golden byte fixtures shared verbatim with the Go side
//! (sys/wire/wire_interop_test.go).

pub const STATUS_RESULT: u32 = 1;
pub const STATUS_YIELD: u32 = 2;
pub const STATUS_FAILED: u32 = 3;

pub struct Syscall {
    pub abi: u32,
    pub name: String,
    pub args: Vec<u8>,
}

#[derive(Default)]
pub struct Response {
    pub abi: u32,
    pub status: u32,
    pub code: String,
    pub result: Vec<u8>,
    pub message: String,
    pub labels: Vec<String>,
}

// Field numbers from envelope.proto. Never reuse a number.
const SYSCALL_ABI: u64 = 1;
const SYSCALL_NAME: u64 = 2;
const SYSCALL_ARGS: u64 = 3;

const RESPONSE_ABI: u64 = 1;
const RESPONSE_STATUS: u64 = 2;
const RESPONSE_CODE: u64 = 3;
const RESPONSE_RESULT: u64 = 4;
const RESPONSE_MESSAGE: u64 = 5;
const RESPONSE_LABELS: u64 = 6;

const WIRE_VARINT: u64 = 0;
const WIRE_I64: u64 = 1;
const WIRE_BYTES: u64 = 2;
const WIRE_I32: u64 = 5;

pub fn encode_syscall(syscall: &Syscall) -> Vec<u8> {
    let mut b = Vec::with_capacity(16 + syscall.name.len() + syscall.args.len());
    append_varint_field(&mut b, SYSCALL_ABI, u64::from(syscall.abi));
    append_bytes_field(&mut b, SYSCALL_NAME, syscall.name.as_bytes());
    append_bytes_field(&mut b, SYSCALL_ARGS, &syscall.args);
    b
}

pub fn decode_response(mut data: &[u8]) -> Result<Response, String> {
    let mut response = Response::default();
    while !data.is_empty() {
        let (tag, rest) = consume_varint(data)?;
        data = rest;
        let (field, wire_type) = (tag >> 3, tag & 7);
        if field == 0 {
            return Err("field number 0 is invalid".into());
        }
        match wire_type {
            WIRE_VARINT => {
                let (value, rest) = consume_varint(data)?;
                data = rest;
                match field {
                    RESPONSE_ABI => response.abi = value as u32,
                    RESPONSE_STATUS => response.status = value as u32,
                    _ => {} // unknown varint field: skipped
                }
            }
            WIRE_BYTES => {
                let (length, rest) = consume_varint(data)?;
                data = rest;
                let length = length as usize;
                if data.len() < length {
                    return Err("truncated message".into());
                }
                let (payload, rest) = data.split_at(length);
                data = rest;
                match field {
                    RESPONSE_CODE => response.code = utf8(payload)?,
                    RESPONSE_RESULT => response.result = payload.to_vec(),
                    RESPONSE_MESSAGE => response.message = utf8(payload)?,
                    RESPONSE_LABELS => response.labels.push(utf8(payload)?),
                    _ => {} // unknown bytes field: skipped
                }
            }
            WIRE_I64 => {
                if data.len() < 8 {
                    return Err("truncated message".into());
                }
                data = &data[8..]; // skippable: the envelope has no i64 fields
            }
            WIRE_I32 => {
                if data.len() < 4 {
                    return Err("truncated message".into());
                }
                data = &data[4..]; // skippable: the envelope has no i32 fields
            }
            other => return Err(format!("unsupported wire type {}", other)),
        }
    }
    Ok(response)
}

fn utf8(payload: &[u8]) -> Result<String, String> {
    String::from_utf8(payload.to_vec()).map_err(|e| format!("invalid utf-8: {}", e))
}

fn consume_varint(data: &[u8]) -> Result<(u64, &[u8]), String> {
    let mut value: u64 = 0;
    for (i, byte) in data.iter().take(10).enumerate() {
        value |= u64::from(byte & 0x7F) << (7 * i);
        if byte < &0x80 {
            return Ok((value, &data[i + 1..]));
        }
    }
    Err("malformed varint".into())
}

// Zero values and empty payloads are omitted, matching proto3 presence.
fn append_varint_field(b: &mut Vec<u8>, field: u64, value: u64) {
    if value == 0 {
        return;
    }
    append_varint(b, field << 3 | WIRE_VARINT);
    append_varint(b, value);
}

fn append_bytes_field(b: &mut Vec<u8>, field: u64, payload: &[u8]) {
    if payload.is_empty() {
        return;
    }
    append_varint(b, field << 3 | WIRE_BYTES);
    append_varint(b, payload.len() as u64);
    b.extend_from_slice(payload);
}

fn append_varint(b: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        b.push(value as u8 | 0x80);
        value >>= 7;
    }
    b.push(value as u8);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(data: &[u8]) -> String {
        data.iter().map(|b| format!("{:02x}", b)).collect()
    }

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    // Golden fixtures shared verbatim with the Go side
    // (capcompute sys/wire/wire_interop_test.go TestGoldenFixtures).
    #[test]
    fn syscall_golden_fixture() {
        let encoded = encode_syscall(&Syscall {
            abi: 3,
            name: "mail.send".into(),
            args: br#"{"to":"ops"}"#.to_vec(),
        });
        assert_eq!(
            hex(&encoded),
            format!(
                "08031209{}1a0c{}",
                hex(b"mail.send"),
                hex(br#"{"to":"ops"}"#)
            )
        );
    }

    #[test]
    fn response_golden_fixture() {
        let gold = format!(
            "080310031a06{}2a02{}3201{}3201{}",
            hex(b"denied"),
            hex(b"no"),
            hex(b"a"),
            hex(b"b")
        );
        let response = decode_response(&unhex(&gold)).unwrap();
        assert_eq!(response.abi, 3);
        assert_eq!(response.status, STATUS_FAILED);
        assert_eq!(response.code, "denied");
        assert_eq!(response.message, "no");
        assert_eq!(response.labels, vec!["a", "b"]);
        assert!(response.result.is_empty());
    }

    #[test]
    fn unknown_fields_skipped() {
        // A valid response followed by unknown field 9 (varint), 10 (bytes),
        // 11 (i64), and 12 (i32) — a future schema must not break us.
        let mut data = unhex("08031001");
        data.extend_from_slice(&[0x48, 0x2A]);
        data.extend_from_slice(&[0x52, 0x03, b'f', b'o', b'o']);
        data.extend_from_slice(&[0x59, 1, 2, 3, 4, 5, 6, 7, 8]);
        data.extend_from_slice(&[0x65, 1, 2, 3, 4]);
        let response = decode_response(&data).unwrap();
        assert_eq!(response.abi, 3);
        assert_eq!(response.status, STATUS_RESULT);
    }

    #[test]
    fn garbage_rejected() {
        assert!(decode_response(br#"{"abi":2}"#).is_err());
        assert!(decode_response(&[0x12, 0x09, b'm']).is_err());
        assert!(decode_response(&[0x08]).is_err());
    }
}
