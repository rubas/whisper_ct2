//! Whisper feature extractor: reads `preprocessor_config.json` and turns
//! raw `f32` PCM into the `[feature_size, nb_max_frames]` log-mel chunks
//! `CTranslate2` consumes. Numerically matches
//! `faster_whisper.FeatureExtractor` (see `build_chunks`).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    // The STFT and mel-projection loops index multiple arrays in lockstep;
    // an `enumerate` rewrite is uglier than the explicit index loop.
    clippy::needless_range_loop
)]

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use std::f32::consts::PI;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use mel_spec::mel::mel;
use ndarray::Array2;
use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use serde::Deserialize;

use crate::errors::invalid_request;

const PREPROCESSOR_CONFIG_FILE: &str = "preprocessor_config.json";

/// Parsed `preprocessor_config.json` plus the (possibly synthesised) mel
/// filterbank used to produce log-mel features.
#[derive(Debug)]
pub(crate) struct Preprocessor {
    pub(crate) feature_size: usize,
    pub(crate) hop_length: usize,
    pub(crate) n_fft: usize,
    pub(crate) n_samples: usize,
    pub(crate) nb_max_frames: usize,
    pub(crate) sampling_rate: usize,
    pub(crate) mel_filters: Array2<f64>,
}

#[derive(Deserialize)]
struct PreprocessorJson {
    feature_size: usize,
    hop_length: usize,
    n_fft: usize,
    n_samples: usize,
    nb_max_frames: usize,
    sampling_rate: usize,
    mel_filters: Option<Vec<Vec<f64>>>,
}

impl Preprocessor {
    pub(crate) fn load<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let path = model_dir.as_ref().join(PREPROCESSOR_CONFIG_FILE);
        let file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
        let aux: PreprocessorJson = serde_json::from_reader(BufReader::new(file))
            .with_context(|| format!("parsing {}", path.display()))?;

        // A corrupt converted-model directory must fail here, at load,
        // with the offending field named — not as an opaque panic
        // (divide by zero, out-of-bounds filterbank index) at the first
        // transcribe.
        for (field, value) in [
            ("feature_size", aux.feature_size),
            ("hop_length", aux.hop_length),
            ("n_fft", aux.n_fft),
            ("n_samples", aux.n_samples),
            ("nb_max_frames", aux.nb_max_frames),
            ("sampling_rate", aux.sampling_rate),
        ] {
            if value == 0 {
                return Err(anyhow!("{field} must be positive in {}", path.display()));
            }
        }

        let mel_filters = if let Some(rows) = aux.mel_filters {
            let n_rows = rows.len();
            let n_cols = rows.first().map_or(0, Vec::len);
            let expected_rows = aux.feature_size;
            let expected_cols = aux.n_fft / 2 + 1;
            if (n_rows, n_cols) != (expected_rows, expected_cols) {
                return Err(anyhow!(
                    "mel_filters shape [{n_rows}, {n_cols}] does not match \
                     [feature_size, n_fft / 2 + 1] = [{expected_rows}, {expected_cols}] in {}",
                    path.display()
                ));
            }
            Array2::from_shape_vec((n_rows, n_cols), rows.into_iter().flatten().collect())
                .with_context(|| {
                    format!(
                        "mel_filters rows have inconsistent lengths in {}",
                        path.display()
                    )
                })?
        } else {
            mel(
                aux.sampling_rate as f64,
                aux.n_fft,
                aux.feature_size,
                None,
                None,
                false,
                true,
            )
        };

        Ok(Self {
            feature_size: aux.feature_size,
            hop_length: aux.hop_length,
            n_fft: aux.n_fft,
            n_samples: aux.n_samples,
            nb_max_frames: aux.nb_max_frames,
            sampling_rate: aux.sampling_rate,
            mel_filters,
        })
    }

    /// Splits `samples` into 30 s windows and produces one
    /// `[feature_size, nb_max_frames]` log-mel array per window.
    ///
    /// Numerically matches `faster_whisper.FeatureExtractor` (librosa STFT
    /// with `center=True` and reflect padding) so the golden fixtures in
    /// `test/fixtures/mel_golden*/` pass element-wise. Each 30 s chunk
    /// is padded with `n_fft/2` reflected samples on each side, framed at
    /// `hop_length` with a Hann window, FFT'd, squared, mel-projected, and
    /// normalised with Whisper's
    /// `(max(log10(max(x,1e-10)), max_log-8) + 4)/4` curve, where
    /// `max_log` is the maximum over **all** chunks of the audio — the
    /// reference computes `log_spec.max()` once over the whole signal, so
    /// a near-silent window is floored against the loudest window of the
    /// audio rather than its own.
    ///
    /// Known deviation: chunks are STFT'd independently, so the first and
    /// last ~`n_fft / (2 * hop_length)` frames of interior chunks use
    /// reflected instead of real neighbouring samples, while the
    /// reference runs one continuous STFT and slices. The golden tests
    /// trim those edge frames.
    pub(crate) fn build_chunks(&self, samples: &[f32]) -> Result<Vec<Array2<f32>>> {
        if samples.is_empty() {
            return Err(anyhow!("samples buffer is empty"));
        }

        let window = hann_window(self.n_fft);
        let fft = self.fft();
        let mut scratch = vec![Complex32::default(); fft.get_inplace_scratch_len()];
        let mut frame_buf = vec![Complex32::default(); self.n_fft];
        let pad = self.n_fft / 2;
        let n_freq = self.n_fft / 2 + 1;

        let mut out = Vec::new();
        let mut max_log = f32::NEG_INFINITY;
        for chunk in samples.chunks(self.n_samples) {
            let padded = reflect_pad_chunk(chunk, self.n_samples, pad);

            let mut mel_chunk = Array2::<f32>::zeros((self.feature_size, self.nb_max_frames));
            for f in 0..self.nb_max_frames {
                let start = f * self.hop_length;
                let end = start + self.n_fft;
                if end > padded.len() {
                    break;
                }
                for i in 0..self.n_fft {
                    frame_buf[i] = Complex32::new(padded[start + i] * window[i], 0.0);
                }
                fft.process_with_scratch(&mut frame_buf, &mut scratch);

                for m in 0..self.feature_size {
                    let mut sum = 0.0_f64;
                    for (k, bin) in frame_buf.iter().take(n_freq).enumerate() {
                        let power = f64::from(bin.re) * f64::from(bin.re)
                            + f64::from(bin.im) * f64::from(bin.im);
                        sum += self.mel_filters[(m, k)] * power;
                    }
                    mel_chunk[(m, f)] = sum as f32;
                }
            }

            // Finite PCM (enforced at the NIF boundary) still overflows
            // the f32 mel power once amplitudes get far outside Whisper's
            // [-1.0, 1.0] contract. This is the last point corruption is
            // detectable: `log_mel` launders NaN into the mel floor and
            // would let garbage transcribe as silence.
            if mel_chunk.iter().any(|v| !v.is_finite()) {
                return Err(invalid_request(
                    "mel power overflowed f32; PCM amplitude is far outside \
                     the [-1.0, 1.0] contract",
                ));
            }

            max_log = max_log.max(log_mel(&mut mel_chunk));
            out.push(mel_chunk);
        }

        // The clamp floor is `whole-audio max - 8`, so scaling has to
        // wait until every chunk's max is in.
        for mel_chunk in &mut out {
            scale_log_mel(mel_chunk, max_log);
        }

        Ok(out)
    }

    fn fft(&self) -> Arc<dyn Fft<f32>> {
        FftPlanner::new().plan_fft_forward(self.n_fft)
    }

    /// Seconds per encoder output frame. Whisper's encoder downsamples mel
    /// frames by 2 (stride-2 conv), so each encoder frame covers
    /// `2 * hop_length / sampling_rate` seconds (0.02 s at 16 kHz / 160 hop).
    pub(crate) fn seconds_per_encoder_frame(&self) -> f32 {
        (2.0 * self.hop_length as f32) / self.sampling_rate as f32
    }
}

/// Pads one chunk for the centred STFT: `n_fft/2` reflected samples on
/// each side of a chunk zero-extended to `n_samples`, so every output
/// gets the full `nb_max_frames` — faster-whisper zero-pads the tail the
/// same way. Matches `np.pad(..., mode='reflect')` on the zero-padded
/// array: a left reflection that reaches past the chunk's data reads the
/// zero region (relevant only for chunks of `<= pad` samples), and the
/// trailing pad reflects whatever sits at the body's end, zeros included.
fn reflect_pad_chunk(chunk: &[f32], n_samples: usize, pad: usize) -> Vec<f32> {
    let mut padded = vec![0.0_f32; n_samples + 2 * pad];
    for i in 0..pad {
        let src = pad - i;
        padded[i] = if src < chunk.len() { chunk[src] } else { 0.0 };
    }
    padded[pad..pad + chunk.len()].copy_from_slice(chunk);
    let body_end = pad + n_samples;
    for i in 0..pad {
        let src = body_end.saturating_sub(2 + i);
        padded[body_end + i] = padded[src];
    }
    padded
}

/// Periodic Hann window of length `n` (numpy's `np.hanning`-style endpoints).
fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / n as f32).cos()))
        .collect()
}

/// First half of the Whisper log-mel normalisation: `log10(max(x, 1e-10))`
/// in place. Returns the chunk's max log value; `build_chunks` folds the
/// maxes of every chunk into the whole-audio max that [`scale_log_mel`]
/// floors against.
fn log_mel(mel: &mut Array2<f32>) -> f32 {
    let mut max_log = f32::NEG_INFINITY;
    for v in mel.iter_mut() {
        let log = v.max(1e-10).log10();
        *v = log;
        if log > max_log {
            max_log = log;
        }
    }
    max_log
}

/// Second half: `(max(x, max_log - 8) + 4) / 4`, with `max_log` the max
/// over the whole audio (faster-whisper / OpenAI Whisper compute
/// `log_spec.max()` globally, not per 30 s window).
fn scale_log_mel(mel: &mut Array2<f32>, max_log: f32) {
    let floor = max_log - 8.0;
    for v in mel.iter_mut() {
        *v = (v.max(floor) + 4.0) / 4.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Workspace-relative path to a checked-in golden mel fixture.
    /// Generated by `tools/mel-reference/generate.py`. Pinned to the
    /// faster-whisper FeatureExtractor at fixture-generation time.
    fn fixture_dir(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("test")
            .join("fixtures")
            .join(name)
    }

    fn read_f32_le(path: &std::path::Path) -> Vec<f32> {
        let bytes = std::fs::read(path).expect("fixture file");
        assert!(bytes.len().is_multiple_of(4), "fixture not 4-byte aligned");
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn build_chunks_rejects_mel_power_overflow() {
        // f32::MAX-scale samples overflow the FFT into ±inf/NaN mel
        // power. Without this rejection `log_mel` would flush the NaN to
        // the silence floor and corrupted audio would transcribe as
        // {:ok, ...} with garbage text.
        let preprocessor = tiny_preprocessor();
        let samples = vec![f32::MAX; 64];
        let err = preprocessor
            .build_chunks(&samples)
            .expect_err("mel power overflow must be rejected");
        assert_eq!(
            crate::errors::kind_from_chain(&err),
            Some("invalid_request")
        );
        assert!(format!("{err:#}").contains("amplitude"));
    }

    #[test]
    fn seconds_per_encoder_frame_matches_whisper_default() {
        let preprocessor = Preprocessor {
            feature_size: 80,
            hop_length: 160,
            n_fft: 400,
            n_samples: 480_000,
            nb_max_frames: 3_000,
            sampling_rate: 16_000,
            mel_filters: Array2::<f64>::zeros((80, 201)),
        };
        // 2 * 160 / 16_000 = 0.02 s. Hard-coded here to catch a
        // regression in either the formula or the assumed defaults.
        let dt = preprocessor.seconds_per_encoder_frame();
        assert!((dt - 0.02).abs() < 1e-9, "got {dt}");
    }

    /// Element-wise comparison of one mel window against its reference
    /// slice, skipping `trim` frames at each edge (chunk-edge frames use
    /// reflected instead of neighbouring samples — the known deviation
    /// documented on `build_chunks`).
    fn assert_chunk_matches_reference(
        mel: &Array2<f32>,
        reference: &[f32],
        feature_size: usize,
        n_frames: usize,
    ) {
        assert_eq!(mel.shape(), &[feature_size, n_frames]);
        assert_eq!(reference.len(), mel.len());

        let mel_slice = mel.as_slice().expect("contiguous mel chunk");

        // Mel layout is `[feature_size, n_frames]` row-major, so the
        // stride along the frame axis is 1. `mel[bin * n_frames + f]`
        // gives bin `bin` at frame `f`.
        //
        // faster-whisper's librosa STFT uses `center=True` with reflect
        // padding over the continuous signal, so the first and last
        // ~`n_fft / (2 * hop_length)` frames of each window have no
        // per-chunk equivalent and must be excluded. At n_fft=400,
        // hop=160 that's ~1 frame either side; we trim 2 to be safe.
        let trim = 2_usize;
        let mut overall_max = 0.0_f64;
        let mut overall_sumsq = 0.0_f64;
        let mut counted = 0_usize;
        for f in trim..(n_frames - trim) {
            for bin in 0..feature_size {
                let idx = bin * n_frames + f;
                let d = f64::from(mel_slice[idx] - reference[idx]).abs();
                if d > overall_max {
                    overall_max = d;
                }
                overall_sumsq += d * d;
                counted += 1;
            }
        }
        let overall_rms = (overall_sumsq / counted as f64).sqrt();

        // Tolerances: per-element max < 0.05 (mel values normalise to
        // ~[-1, 1] after `norm_mel`, so 5 % is a hard fail), and RMS
        // across all interior frames < 0.005. A wrong filterbank or
        // power-vs-magnitude bug pushes both into the 0.1+ range.
        assert!(
            overall_max < 0.05,
            "mel max-abs delta {overall_max:.4} exceeds 0.05 — \
             likely filterbank or STFT scaling drift"
        );
        assert!(
            overall_rms < 0.005,
            "mel RMS delta {overall_rms:.6} exceeds 0.005 — \
             likely a systemic preprocessor parity issue"
        );
    }

    #[test]
    fn build_chunks_matches_faster_whisper_golden_mel() {
        let dir = fixture_dir("mel_golden");
        let preprocessor = Preprocessor::load(&dir).expect("load preprocessor config");
        let samples = read_f32_le(&dir.join("input.pcm_f32"));
        let reference = read_f32_le(&dir.join("mel.f32"));

        let chunks = preprocessor
            .build_chunks(&samples)
            .expect("build_chunks succeeds");
        assert_eq!(chunks.len(), 1, "fixture is a single 30 s window");
        assert_chunk_matches_reference(
            &chunks[0],
            &reference,
            preprocessor.feature_size,
            preprocessor.nb_max_frames,
        );
    }

    #[test]
    fn build_chunks_matches_faster_whisper_golden_mel_multichunk() {
        // Two-window fixture (3 s windows): a loud first window and a
        // near-silent partial second window (1 s of quiet tone + zero
        // pad). Pins the whole-audio normalisation max — the quiet
        // window must be floored against the loud window's max, not its
        // own — and the tail zero-pad path; the single-window fixture
        // can see neither.
        let dir = fixture_dir("mel_golden_multichunk");
        let preprocessor = Preprocessor::load(&dir).expect("load preprocessor config");
        let samples = read_f32_le(&dir.join("input.pcm_f32"));
        let reference = read_f32_le(&dir.join("mel.f32"));

        let chunks = preprocessor
            .build_chunks(&samples)
            .expect("build_chunks succeeds");
        assert_eq!(chunks.len(), 2, "fixture spans two windows");
        let window_len = preprocessor.feature_size * preprocessor.nb_max_frames;
        assert_eq!(reference.len(), 2 * window_len);
        for (k, chunk) in chunks.iter().enumerate() {
            assert_chunk_matches_reference(
                chunk,
                &reference[k * window_len..(k + 1) * window_len],
                preprocessor.feature_size,
                preprocessor.nb_max_frames,
            );
        }
    }

    fn tiny_preprocessor() -> Preprocessor {
        Preprocessor {
            feature_size: 2,
            hop_length: 4,
            n_fft: 8,
            n_samples: 64,
            nb_max_frames: 16,
            sampling_rate: 16,
            mel_filters: Array2::<f64>::ones((2, 5)),
        }
    }

    #[test]
    fn build_chunks_normalises_against_the_whole_audio_max() {
        let preprocessor = tiny_preprocessor();

        // Chunk 1 carries signal; chunk 2 is pure silence.
        let mut samples = vec![0.0_f32; 128];
        for (i, v) in samples.iter_mut().take(64).enumerate() {
            *v = 0.5 * (i as f32 * 0.7).sin();
        }

        let chunks = preprocessor.build_chunks(&samples).expect("build_chunks");
        assert_eq!(chunks.len(), 2);

        let loud_max = chunks[0].iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Every silent-chunk value sits at the clamp floor, which derives
        // from the WHOLE-audio max: scaled floor = loud max - 8/4 =
        // loud max - 2. Per-chunk normalisation (the old behaviour)
        // floors silence against its own max instead, pinning it at
        // (-10 + 4) / 4 = -1.5 regardless of the rest of the audio.
        let first = chunks[1][(0, 0)];
        assert!(
            chunks[1].iter().all(|v| (v - first).abs() < 1e-6),
            "silent chunk must be uniformly floored, got non-uniform values"
        );
        assert!(
            (first - (loud_max - 2.0)).abs() < 1e-5,
            "silent-chunk floor {first} must be loud-chunk max {loud_max} - 2.0"
        );
    }

    #[test]
    fn reflect_pad_chunk_mirrors_edges_without_repeating_them() {
        // np.pad 'reflect' convention: the edge sample itself is not
        // repeated. Left pad reads chunk[pad], .., chunk[1]; the body is
        // zero-extended to n_samples; the trailing pad mirrors the body
        // end (zeros included).
        let chunk = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let padded = reflect_pad_chunk(&chunk, 8, 4);
        assert_eq!(&padded[..4], &[5.0, 4.0, 3.0, 2.0]);
        assert_eq!(&padded[4..10], &chunk);
        assert_eq!(&padded[10..12], &[0.0, 0.0]);
        assert_eq!(&padded[12..], &[0.0, 6.0, 5.0, 4.0]);
    }

    #[test]
    fn reflect_pad_chunk_reflects_into_zeros_for_tiny_chunks() {
        // A chunk of <= pad samples reflects past its own data into the
        // zero-extended region — np.pad 'reflect' on the zero-padded
        // array reads zeros there. The old clamp duplicated the chunk's
        // last sample instead, injecting spurious energy into the
        // leading frames of sub-12.5 ms audio.
        let padded = reflect_pad_chunk(&[0.5], 8, 4);
        assert_eq!(&padded[..5], &[0.0, 0.0, 0.0, 0.0, 0.5]);

        let padded = reflect_pad_chunk(&[1.0, 2.0, 3.0], 8, 4);
        assert_eq!(&padded[..4], &[0.0, 0.0, 3.0, 2.0]);
    }

    fn write_config(dir: &std::path::Path, json: &str) {
        std::fs::write(dir.join(PREPROCESSOR_CONFIG_FILE), json).expect("write config");
    }

    fn base_config() -> serde_json::Value {
        serde_json::json!({
            "feature_size": 2,
            "hop_length": 4,
            "n_fft": 8,
            "n_samples": 64,
            "nb_max_frames": 16,
            "sampling_rate": 16,
        })
    }

    #[test]
    fn load_rejects_zero_numeric_fields_naming_the_field() {
        // Zero values panic later (slice::chunks(0), divide by zero) and
        // would surface as an opaque :nif_panic at the first transcribe;
        // the load error must name the bad field instead.
        for field in [
            "feature_size",
            "hop_length",
            "n_fft",
            "n_samples",
            "nb_max_frames",
            "sampling_rate",
        ] {
            let dir = tempfile::tempdir().expect("tempdir");
            let mut config = base_config();
            config[field] = serde_json::json!(0);
            write_config(dir.path(), &config.to_string());

            let err = Preprocessor::load(dir.path()).expect_err("zero field must fail at load");
            let msg = format!("{err:#}");
            assert!(msg.contains(field), "error must name {field}, got: {msg}");
        }
    }

    #[test]
    fn load_rejects_mel_filters_shape_mismatch() {
        // An undersized filterbank panics on out-of-bounds indexing in
        // the mel projection at transcribe time; the mismatch must be a
        // load error naming both shapes.
        let dir = tempfile::tempdir().expect("tempdir");
        let mut config = base_config();
        // 2 rows x 4 cols, but n_fft/2 + 1 = 5 columns are required.
        config["mel_filters"] = serde_json::json!([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]);
        write_config(dir.path(), &config.to_string());

        let err = Preprocessor::load(dir.path()).expect_err("shape mismatch must fail at load");
        let msg = format!("{err:#}");
        assert!(msg.contains("mel_filters"), "got: {msg}");
        assert!(
            msg.contains("[2, 4]") && msg.contains("[2, 5]"),
            "got: {msg}"
        );
    }

    #[test]
    fn load_rejects_ragged_mel_filters_rows() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut config = base_config();
        // First row sets 5 columns; the second row is short.
        config["mel_filters"] =
            serde_json::json!([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]);
        write_config(dir.path(), &config.to_string());

        let err = Preprocessor::load(dir.path()).expect_err("ragged rows must fail at load");
        assert!(format!("{err:#}").contains("mel_filters"));
    }

    #[test]
    fn load_accepts_a_consistent_config() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut config = base_config();
        config["mel_filters"] = serde_json::json!(vec![vec![0.1_f64; 5]; 2]);
        write_config(dir.path(), &config.to_string());

        let preprocessor = Preprocessor::load(dir.path()).expect("consistent config loads");
        assert_eq!(preprocessor.mel_filters.shape(), &[2, 5]);
        assert_eq!(preprocessor.n_samples, 64);
    }
}
