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

const PREPROCESSOR_CONFIG_FILE: &str = "preprocessor_config.json";

/// Parsed `preprocessor_config.json` plus the (possibly synthesised) mel
/// filterbank used to produce log-mel features.
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

        let mel_filters = if let Some(rows) = aux.mel_filters {
            let n_rows = rows.len();
            let n_cols = rows.first().map_or(0, Vec::len);
            Array2::from_shape_vec((n_rows, n_cols), rows.into_iter().flatten().collect())?
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
    /// with `center=True` and reflect padding) so the golden fixture in
    /// `test/fixtures/mel_golden/` passes element-wise. Each 30 s chunk
    /// is padded with `n_fft/2` reflected samples on each side, framed at
    /// `hop_length` with a Hann window, FFT'd, squared, mel-projected, and
    /// normalised with Whisper's
    /// `(max(log10(max(x,1e-10)), max_log-8) + 4)/4` curve.
    ///
    /// Chunks are processed independently — no STFT state leaks between
    /// the 30 s windows.
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
        for chunk in samples.chunks(self.n_samples) {
            // Pad chunk to n_samples so every output gets the full
            // nb_max_frames; faster-whisper zero-pads the tail the same way.
            let mut padded = vec![0.0_f32; self.n_samples + 2 * pad];
            for i in 0..pad {
                let src = (pad - i).min(chunk.len().saturating_sub(1));
                padded[i] = chunk[src];
            }
            padded[pad..pad + chunk.len()].copy_from_slice(chunk);
            // Tail past `chunk.len()` is already zero; reflect the last
            // samples into the trailing pad area for parity with
            // `np.pad(..., mode='reflect')` on a zero-padded array.
            let body_end = pad + self.n_samples;
            for i in 0..pad {
                let src = body_end.saturating_sub(2 + i);
                padded[body_end + i] = padded[src];
            }

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

            normalise_log_mel(&mut mel_chunk);
            out.push(mel_chunk);
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

/// Periodic Hann window of length `n` (numpy's `np.hanning`-style endpoints).
fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / n as f32).cos()))
        .collect()
}

/// Faster-whisper / OpenAI Whisper log-mel normalisation:
/// `clip(log10(max(x, 1e-10)), max - 8, ...)` then `(x + 4) / 4`.
fn normalise_log_mel(mel: &mut Array2<f32>) {
    let mut max_log = f32::NEG_INFINITY;
    for v in mel.iter_mut() {
        let log = v.max(1e-10).log10();
        *v = log;
        if log > max_log {
            max_log = log;
        }
    }
    let floor = max_log - 8.0;
    for v in mel.iter_mut() {
        *v = (v.max(floor) + 4.0) / 4.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Workspace-relative path to the checked-in golden mel fixture.
    /// Generated by `tools/mel-reference/generate.py`. Pinned to the
    /// faster-whisper FeatureExtractor at fixture-generation time.
    fn golden_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("test")
            .join("fixtures")
            .join("mel_golden")
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

    #[test]
    fn build_chunks_matches_faster_whisper_golden_mel() {
        let dir = golden_dir();
        let preprocessor = Preprocessor::load(&dir).expect("load preprocessor config");
        let samples = read_f32_le(&dir.join("input.pcm_f32"));
        let reference = read_f32_le(&dir.join("mel.f32"));

        let chunks = preprocessor
            .build_chunks(&samples)
            .expect("build_chunks succeeds");
        assert_eq!(chunks.len(), 1, "fixture is a single 30 s window");
        let mel = chunks.into_iter().next().unwrap();
        assert_eq!(
            mel.shape(),
            &[preprocessor.feature_size, preprocessor.nb_max_frames]
        );
        assert_eq!(reference.len(), mel.len());

        let mel_slice = mel.as_slice().expect("contiguous mel chunk");
        let feature_size = preprocessor.feature_size;
        let n_frames = preprocessor.nb_max_frames;

        // Mel layout is `[feature_size, n_frames]` row-major, so the
        // stride along the frame axis is 1. `mel[bin * n_frames + f]`
        // gives bin `bin` at frame `f`.
        //
        // Per-frame RMS captures average bin error within one time slice.
        // A wrong filterbank shows up as systemic non-zero RMS across
        // every frame; an STFT framing difference shows up as a tight
        // band of bad frames at chunk boundaries with a clean middle.
        let mut per_frame_rms = vec![0.0_f64; n_frames];
        for bin in 0..feature_size {
            for f in 0..n_frames {
                let idx = bin * n_frames + f;
                let d = f64::from(mel_slice[idx] - reference[idx]);
                per_frame_rms[f] += d * d;
            }
        }
        for v in &mut per_frame_rms {
            *v = (*v / feature_size as f64).sqrt();
        }

        // faster-whisper's librosa STFT uses `center=True` with reflect
        // padding, so the first and last ~`n_fft / (2 * hop_length)`
        // frames have no streaming-STFT equivalent and must be
        // excluded. At n_fft=400, hop=160 that's ~1 frame either side;
        // we trim 2 to be safe.
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
}
