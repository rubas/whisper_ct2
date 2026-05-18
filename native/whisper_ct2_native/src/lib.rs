//! Rustler NIF wrapping `ct2rs::Whisper` for Whisper speech-to-text via
//! `CTranslate2`.
//!
//! NIF surface (all functions return a 2-tuple of `{:ok, value}` or
//! `{:error, %{type, message, details}}`):
//!
//! - [`nif_available_devices`]    — runtime device discovery.
//! - [`nif_load_model`]           — load a CT2-converted Whisper directory.
//! - [`nif_model_info`]           — model metadata snapshot.
//! - [`nif_transcribe`]           — run inference on a buffer of f32 PCM.
//!
//! `samples` is a BEAM binary of little-endian IEEE-754 `f32` PCM samples
//! (mono, 16 kHz). `ct2rs::Whisper` internally splits audio longer than its
//! 30 s window, so callers do not need to chunk first.

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::Mutex;

use ct2rs::sys::{get_device_count, ComputeType, Config, Device};
use ct2rs::{Whisper, WhisperOptions};
use rustler::types::binary::Binary;
use rustler::{Encoder, Env, NifMap, ResourceArc, Term};

#[allow(missing_docs)]
mod atoms {
    rustler::atoms! {
        ok,
        error,
    }
}
use atoms::{error, ok};

/// `true` when this build was compiled with any CUDA cargo feature. Used as a
/// compile-time gate so non-CUDA builds neither link nor call CUDA paths.
const CUDA_SUPPORTED: bool = cfg!(any(feature = "cuda", feature = "cuda-dynamic"));

/// Structured error returned to Elixir as a `NifMap`. The Elixir side maps
/// `type` to a `WhisperCt2.Error.reason` atom.
#[derive(Debug, NifMap)]
struct NativeError {
    r#type: String,
    message: String,
    details: HashMap<String, String>,
}

impl NativeError {
    fn new(type_name: &str, message: impl Into<String>) -> Self {
        Self {
            r#type: type_name.to_owned(),
            message: message.into(),
            details: HashMap::new(),
        }
    }

    fn with_detail(mut self, key: &str, value: impl Into<String>) -> Self {
        self.details.insert(key.to_owned(), value.into());
        self
    }
}

/// Opaque BEAM resource holding a loaded Whisper model.
///
/// `ct2rs::Whisper` is not documented as `Sync`, so inference calls are
/// serialised through a [`Mutex`]. The `CTranslate2` engine itself is
/// thread-safe; load multiple models if you need parallel inference.
struct WhisperResource {
    whisper: Mutex<Whisper>,
    sampling_rate: usize,
    n_samples: usize,
    multilingual: bool,
    device: &'static str,
    compute_type: &'static str,
}

impl rustler::Resource for WhisperResource {}

#[derive(NifMap)]
struct LoadOpts {
    device: Option<String>,
    compute_type: Option<String>,
    device_indices: Option<Vec<i32>>,
    num_threads_per_replica: Option<u32>,
    max_queued_batches: Option<i32>,
    cpu_core_offset: Option<i32>,
}

#[derive(NifMap)]
struct TranscribeOpts {
    language: Option<String>,
    timestamp: bool,
    beam_size: Option<u32>,
    patience: Option<f32>,
    length_penalty: Option<f32>,
    repetition_penalty: Option<f32>,
    no_repeat_ngram_size: Option<u32>,
    sampling_temperature: Option<f32>,
    sampling_topk: Option<u32>,
    suppress_blank: Option<bool>,
    max_length: Option<u32>,
    num_hypotheses: Option<u32>,
    return_scores: Option<bool>,
    return_logits_vocab: Option<bool>,
    return_no_speech_prob: Option<bool>,
    max_initial_timestamp_index: Option<u32>,
    suppress_tokens: Option<Vec<i32>>,
}

#[derive(NifMap)]
struct ModelInfo {
    sampling_rate: usize,
    n_samples: usize,
    multilingual: bool,
    device: String,
    compute_type: String,
}

#[derive(NifMap)]
struct AvailableDevices {
    cpu: i32,
    cuda: i32,
    cuda_supported: bool,
}

fn run_with_panic_protection<T, F>(f: F) -> Result<T, NativeError>
where
    F: FnOnce() -> Result<T, NativeError>,
{
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|panic_info| {
        let message = panic_info
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic_info.downcast_ref::<&str>().copied())
            .unwrap_or("unknown panic");
        Err(NativeError::new("nif_panic", message))
    })
}

fn encode_result<T: Encoder>(env: Env<'_>, result: Result<T, NativeError>) -> Term<'_> {
    match result {
        Ok(value) => (ok(), value).encode(env),
        Err(err) => (error(), err).encode(env),
    }
}

fn parse_device(s: &str) -> Result<Device, NativeError> {
    match s.to_ascii_lowercase().as_str() {
        "cpu" => Ok(Device::CPU),
        "cuda" | "gpu" => Ok(Device::CUDA),
        other => Err(NativeError::new("invalid_request", "unknown device")
            .with_detail("device", other)
            .with_detail("supported", "cpu,cuda")),
    }
}

#[inline]
fn device_label(d: Device) -> &'static str {
    match d {
        Device::CPU => "cpu",
        Device::CUDA => "cuda",
        _ => "unknown",
    }
}

fn parse_compute_type(s: &str) -> Result<ComputeType, NativeError> {
    match s.to_ascii_lowercase().as_str() {
        "default" => Ok(ComputeType::DEFAULT),
        "auto" => Ok(ComputeType::AUTO),
        "float32" | "fp32" => Ok(ComputeType::FLOAT32),
        "float16" | "fp16" => Ok(ComputeType::FLOAT16),
        "bfloat16" | "bf16" => Ok(ComputeType::BFLOAT16),
        "int8" => Ok(ComputeType::INT8),
        "int8_float32" => Ok(ComputeType::INT8_FLOAT32),
        "int8_float16" => Ok(ComputeType::INT8_FLOAT16),
        "int8_bfloat16" => Ok(ComputeType::INT8_BFLOAT16),
        "int16" => Ok(ComputeType::INT16),
        other => Err(NativeError::new("invalid_request", "unknown compute_type")
            .with_detail("compute_type", other)),
    }
}

#[inline]
fn compute_type_label(c: ComputeType) -> &'static str {
    match c {
        ComputeType::DEFAULT => "default",
        ComputeType::AUTO => "auto",
        ComputeType::FLOAT32 => "float32",
        ComputeType::FLOAT16 => "float16",
        ComputeType::BFLOAT16 => "bfloat16",
        ComputeType::INT8 => "int8",
        ComputeType::INT8_FLOAT32 => "int8_float32",
        ComputeType::INT8_FLOAT16 => "int8_float16",
        ComputeType::INT8_BFLOAT16 => "int8_bfloat16",
        ComputeType::INT16 => "int16",
        _ => "unknown",
    }
}

/// Resolves `:auto` to CUDA when CUDA is built in and a device is visible.
/// Explicit `:cuda` returns an error if either condition fails.
fn resolve_device(requested: Option<&str>) -> Result<Device, NativeError> {
    let lowered = requested.map(str::to_ascii_lowercase);
    match lowered.as_deref() {
        None | Some("auto") => {
            if CUDA_SUPPORTED && get_device_count(Device::CUDA) > 0 {
                Ok(Device::CUDA)
            } else {
                Ok(Device::CPU)
            }
        }
        Some(other) => {
            let device = parse_device(other)?;
            if matches!(device, Device::CUDA) {
                if !CUDA_SUPPORTED {
                    return Err(NativeError::new(
                        "invalid_request",
                        "this build of whisper_ct2 was not compiled with CUDA support",
                    )
                    .with_detail(
                        "rebuild_with",
                        "WHISPER_CT2_FEATURES=cuda-dynamic mix compile",
                    ));
                }
                if get_device_count(Device::CUDA) == 0 {
                    return Err(NativeError::new(
                        "invalid_request",
                        "no CUDA devices visible to CTranslate2",
                    ));
                }
            }
            Ok(device)
        }
    }
}

/// Reports `CTranslate2` device support for this build.
#[rustler::nif]
fn nif_available_devices(env: Env<'_>) -> Term<'_> {
    let result = run_with_panic_protection(|| {
        let cuda = if CUDA_SUPPORTED {
            get_device_count(Device::CUDA)
        } else {
            0
        };
        Ok(AvailableDevices {
            cpu: get_device_count(Device::CPU),
            cuda,
            cuda_supported: CUDA_SUPPORTED,
        })
    });
    encode_result(env, result)
}

/// Loads a CT2-converted Whisper directory.
#[rustler::nif(schedule = "DirtyCpu")]
#[allow(clippy::needless_pass_by_value)] // Rustler decodes nif args by value.
fn nif_load_model(env: Env<'_>, path: String, opts: LoadOpts) -> Term<'_> {
    let result = run_with_panic_protection(|| {
        let path_buf = PathBuf::from(&path);
        if !path_buf.is_dir() {
            return Err(
                NativeError::new("invalid_request", "model path is not a directory")
                    .with_detail("path", path.clone()),
            );
        }

        let device = resolve_device(opts.device.as_deref())?;
        let compute_type = opts
            .compute_type
            .as_deref()
            .map_or_else(|| Ok(ComputeType::default()), parse_compute_type)?;

        let device_indices = match opts.device_indices {
            Some(v) if v.is_empty() => {
                return Err(NativeError::new(
                    "invalid_request",
                    "device_indices must be non-empty",
                ));
            }
            Some(v) => v,
            None => vec![0],
        };

        let mut config = Config {
            device,
            compute_type,
            device_indices,
            num_threads_per_replica: opts.num_threads_per_replica.unwrap_or(0) as usize,
            ..Config::default()
        };

        if let Some(v) = opts.max_queued_batches {
            config.max_queued_batches = v;
        }
        if let Some(v) = opts.cpu_core_offset {
            config.cpu_core_offset = v;
        }

        let whisper = Whisper::new(&path_buf, config).map_err(|reason| {
            NativeError::new("load_error", "failed to load Whisper model")
                .with_detail("reason", reason.to_string())
                .with_detail("path", path.clone())
                .with_detail("device", device_label(device))
                .with_detail("compute_type", compute_type_label(compute_type))
        })?;

        let sampling_rate = whisper.sampling_rate();
        let n_samples = whisper.n_samples();
        let multilingual = whisper.is_multilingual();

        Ok(ResourceArc::new(WhisperResource {
            whisper: Mutex::new(whisper),
            sampling_rate,
            n_samples,
            multilingual,
            device: device_label(device),
            compute_type: compute_type_label(compute_type),
        }))
    });

    encode_result(env, result)
}

/// Returns metadata cached at load time.
#[rustler::nif]
#[allow(clippy::needless_pass_by_value)] // Rustler decodes nif args by value.
fn nif_model_info(env: Env<'_>, model: ResourceArc<WhisperResource>) -> Term<'_> {
    let result = run_with_panic_protection(|| {
        Ok(ModelInfo {
            sampling_rate: model.sampling_rate,
            n_samples: model.n_samples,
            multilingual: model.multilingual,
            device: model.device.to_owned(),
            compute_type: model.compute_type.to_owned(),
        })
    });
    encode_result(env, result)
}

fn decode_pcm_f32(bytes: &[u8]) -> Result<Vec<f32>, NativeError> {
    if bytes.len() % 4 != 0 {
        return Err(NativeError::new(
            "invalid_request",
            "samples binary length must be a multiple of 4 (f32)",
        )
        .with_detail("byte_length", bytes.len().to_string()));
    }

    let samples = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    Ok(samples)
}

fn build_whisper_options(opts: &TranscribeOpts) -> WhisperOptions {
    let mut whisper_opts = WhisperOptions::default();
    if let Some(v) = opts.beam_size {
        whisper_opts.beam_size = v as usize;
    }
    if let Some(v) = opts.patience {
        whisper_opts.patience = v;
    }
    if let Some(v) = opts.length_penalty {
        whisper_opts.length_penalty = v;
    }
    if let Some(v) = opts.repetition_penalty {
        whisper_opts.repetition_penalty = v;
    }
    if let Some(v) = opts.no_repeat_ngram_size {
        whisper_opts.no_repeat_ngram_size = v as usize;
    }
    if let Some(v) = opts.sampling_temperature {
        whisper_opts.sampling_temperature = v;
    }
    if let Some(v) = opts.sampling_topk {
        whisper_opts.sampling_topk = v as usize;
    }
    if let Some(v) = opts.suppress_blank {
        whisper_opts.suppress_blank = v;
    }
    if let Some(v) = opts.max_length {
        whisper_opts.max_length = v as usize;
    }
    if let Some(v) = opts.num_hypotheses {
        whisper_opts.num_hypotheses = v as usize;
    }
    if let Some(v) = opts.return_scores {
        whisper_opts.return_scores = v;
    }
    if let Some(v) = opts.return_logits_vocab {
        whisper_opts.return_logits_vocab = v;
    }
    if let Some(v) = opts.return_no_speech_prob {
        whisper_opts.return_no_speech_prob = v;
    }
    if let Some(v) = opts.max_initial_timestamp_index {
        whisper_opts.max_initial_timestamp_index = v as usize;
    }
    if let Some(ref tokens) = opts.suppress_tokens {
        whisper_opts.suppress_tokens.clone_from(tokens);
    }
    whisper_opts
}

/// Runs Whisper inference. `samples_bin` may be longer than the 30 s Whisper
/// window — `ct2rs::Whisper::generate` splits it internally.
#[rustler::nif(schedule = "DirtyCpu")]
#[allow(clippy::needless_pass_by_value)] // Rustler decodes nif args by value.
fn nif_transcribe<'a>(
    env: Env<'a>,
    model: ResourceArc<WhisperResource>,
    samples_bin: Binary,
    opts: TranscribeOpts,
) -> Term<'a> {
    let bytes = samples_bin.as_slice();
    let result = run_with_panic_protection(|| {
        let samples = decode_pcm_f32(bytes)?;
        let whisper_opts = build_whisper_options(&opts);
        let language = opts.language.as_deref();

        let whisper = model
            .whisper
            .lock()
            .map_err(|_| NativeError::new("runtime_error", "model mutex poisoned"))?;

        whisper
            .generate(&samples, language, opts.timestamp, &whisper_opts)
            .map_err(|reason| {
                NativeError::new("inference_error", "Whisper inference failed")
                    .with_detail("reason", reason.to_string())
            })
    });

    encode_result(env, result)
}

fn on_load(env: Env<'_>, _info: Term<'_>) -> bool {
    env.register::<WhisperResource>().is_ok()
}

rustler::init!("Elixir.WhisperCt2.Native", load = on_load);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_pcm_f32_round_trips_samples() {
        let mut bytes = Vec::new();
        for v in [0.0_f32, 1.0, -1.0, 0.5, -0.25] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let decoded = decode_pcm_f32(&bytes).unwrap();
        assert_eq!(decoded, vec![0.0, 1.0, -1.0, 0.5, -0.25]);
    }

    #[test]
    fn decode_pcm_f32_rejects_misaligned_length() {
        let err = decode_pcm_f32(&[1, 2, 3]).unwrap_err();
        assert_eq!(err.r#type, "invalid_request");
        assert_eq!(
            err.details.get("byte_length").map(String::as_str),
            Some("3")
        );
    }

    #[test]
    fn parse_device_accepts_canonical_names() {
        assert!(matches!(parse_device("cpu").unwrap(), Device::CPU));
        assert!(matches!(parse_device("CUDA").unwrap(), Device::CUDA));
        assert!(matches!(parse_device("gpu").unwrap(), Device::CUDA));
        assert!(parse_device("tpu").is_err());
    }

    #[test]
    fn parse_compute_type_accepts_aliases() {
        assert!(matches!(
            parse_compute_type("fp16").unwrap(),
            ComputeType::FLOAT16
        ));
        assert!(matches!(
            parse_compute_type("int8_float16").unwrap(),
            ComputeType::INT8_FLOAT16
        ));
        assert!(parse_compute_type("nibble").is_err());
    }

    #[test]
    fn resolve_device_auto_falls_back_to_cpu_without_cuda() {
        if !CUDA_SUPPORTED || get_device_count(Device::CUDA) == 0 {
            assert!(matches!(resolve_device(None).unwrap(), Device::CPU));
            assert!(matches!(resolve_device(Some("auto")).unwrap(), Device::CPU));
        }
    }

    #[test]
    fn resolve_device_rejects_cuda_when_unavailable() {
        if !CUDA_SUPPORTED {
            let err = resolve_device(Some("cuda")).unwrap_err();
            assert_eq!(err.r#type, "invalid_request");
            assert!(err.message.contains("CUDA"));
        }
    }

    #[test]
    fn run_with_panic_protection_catches_string_panic() {
        let result: Result<(), _> = run_with_panic_protection(|| panic!("boom"));
        let err = result.unwrap_err();
        assert_eq!(err.r#type, "nif_panic");
        assert_eq!(err.message, "boom");
    }

    #[test]
    fn run_with_panic_protection_passes_through_ok_and_err() {
        let ok_result = run_with_panic_protection(|| Ok::<_, NativeError>(42));
        assert_eq!(ok_result.unwrap(), 42);

        let err_result: Result<(), _> =
            run_with_panic_protection(|| Err(NativeError::new("load_error", "x")));
        assert_eq!(err_result.unwrap_err().r#type, "load_error");
    }
}
