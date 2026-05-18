//! Core transcription flow driving `ct2rs::sys::Whisper`. `transcribe_many`
//! stacks every chunk of every audio into one batched `encode`+`generate`
//! call so multi-chunk and multi-audio inputs share a single encoder pass;
//! `word_timestamps` adds one batched `align` on the same encoder output.

// `transcribe_many` is intentionally one long function — every step shares
// per-audio bookkeeping and splitting it costs clarity more than it saves.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::too_many_lines
)]

use anyhow::{anyhow, Context, Result};
use ct2rs::sys::{Device, StorageView, Whisper, WhisperOptions};
use ct2rs::tokenizers::hf;

use crate::align::{align_batch, ChunkAlignInput, DEFAULT_MEDIAN_FILTER_WIDTH};
use crate::errors::{invalid_request, runtime_error};
use crate::preprocessor::Preprocessor;
use crate::tokens::{
    decode_ids, encode_plain, language_token, split_sub_segments, token_id, PromptParts,
    SpecialTokens, SubSegment, NO_TIMESTAMPS, SOT, STARTOFPREV, TRANSCRIBE,
};

/// Soft cap on the flat mel buffer (`total_chunks * n_mels * nb_max_frames`
/// f32 elements) to keep `transcribe_batch` from OOM-killing the BEAM on a
/// pathological caller. 2 GiB ≈ 537 M f32, well past any realistic batch:
/// at the standard tiny config (80 mel × 3000 frames × 4 B) one chunk is
/// 960 kB, so 2 GiB tolerates ~2200 chunks ≈ 18 h of audio in one call.
const MAX_FEATURE_BUFFER_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// Request knobs for one transcribe call (applies to every audio in a batch).
pub(crate) struct TranscribeRequest {
    pub(crate) language: Option<String>,
    pub(crate) with_timestamps: bool,
    pub(crate) initial_prompt: Option<String>,
    pub(crate) prefix: Option<String>,
    pub(crate) word_timestamps: bool,
    pub(crate) options: WhisperOptions,
}

/// One word with absolute time span.
pub(crate) struct WordResult {
    pub(crate) text: String,
    pub(crate) start: f32,
    pub(crate) end: f32,
    pub(crate) probability: f32,
}

/// One `<|t_start|> text <|t_end|>` segment.
pub(crate) struct SegmentResult {
    pub(crate) text: String,
    pub(crate) start: f32,
    pub(crate) end: f32,
    pub(crate) no_speech_prob: f32,
    pub(crate) avg_logprob: f32,
    pub(crate) tokens: Vec<u32>,
    pub(crate) words: Option<Vec<WordResult>>,
}

/// Full transcription of one audio.
pub(crate) struct TranscriptionResult {
    pub(crate) language: String,
    pub(crate) duration_s: f32,
    pub(crate) segments: Vec<SegmentResult>,
}

/// Per-chunk state shared between the parse, align, and materialise
/// passes. Keeping all three fields together prevents the parallel-vec
/// indexing the original implementation had to do by hand.
#[derive(Default)]
struct ChunkState {
    sub_segments: Vec<SubSegment>,
    offset_s: f32,
    num_frames: usize,
}

/// Transcribes a single audio.
pub(crate) fn transcribe_one(
    whisper: &Whisper,
    tokenizer: &hf::Tokenizer,
    preprocessor: &Preprocessor,
    specials: &SpecialTokens,
    samples: &[f32],
    request: &TranscribeRequest,
) -> Result<TranscriptionResult> {
    let results = transcribe_many(
        whisper,
        tokenizer,
        preprocessor,
        specials,
        &[samples],
        request,
    )?;
    results
        .into_iter()
        .next()
        .ok_or_else(|| runtime_error("transcribe_many returned no results"))
}

/// Transcribes multiple audios in a single batched `generate` call.
/// Language detection (when not pinned) runs per-audio on the first chunk.
pub(crate) fn transcribe_many(
    whisper: &Whisper,
    tokenizer: &hf::Tokenizer,
    preprocessor: &Preprocessor,
    specials: &SpecialTokens,
    audios: &[&[f32]],
    request: &TranscribeRequest,
) -> Result<Vec<TranscriptionResult>> {
    if audios.is_empty() {
        return Ok(Vec::new());
    }

    let mut per_audio_chunks: Vec<Vec<ndarray::Array2<f32>>> = Vec::with_capacity(audios.len());
    let mut per_audio_languages: Vec<String> = Vec::with_capacity(audios.len());
    let mut per_audio_duration: Vec<f32> = Vec::with_capacity(audios.len());

    for samples in audios {
        let chunks = preprocessor
            .build_chunks(samples)
            .context("building mel chunks")?;
        if chunks.is_empty() {
            return Err(invalid_request("audio produced no mel chunks"));
        }
        let duration_s = samples.len() as f32 / preprocessor.sampling_rate as f32;
        per_audio_duration.push(duration_s);
        per_audio_chunks.push(chunks);
    }

    let chunk_counts: Vec<usize> = per_audio_chunks.iter().map(Vec::len).collect();
    let chunk_offsets = compute_chunk_offsets(&chunk_counts);

    let multilingual = whisper.is_multilingual();
    if let Some(lang) = &request.language {
        reject_non_english_on_en_checkpoint(multilingual, lang)?;
        let token = language_token(tokenizer, lang)
            .map_err(|_| invalid_request(format!("invalid language code: {lang}")))?;
        per_audio_languages.extend(std::iter::repeat_n(token, audios.len()));
    } else if !multilingual {
        // English-only checkpoints (`*.en`) always transcribe English.
        // Skip `detect_language` and pin the result.
        per_audio_languages.extend(std::iter::repeat_n("<|en|>".to_owned(), audios.len()));
    } else {
        for chunks in &per_audio_chunks {
            let detected = detect_language(whisper, &chunks[0], preprocessor)?;
            per_audio_languages.push(detected);
        }
    }

    let total_chunks: usize = per_audio_chunks.iter().map(Vec::len).sum();
    let n_mels = preprocessor.feature_size;
    let chunk_length = preprocessor.nb_max_frames;

    // Guard against a pathological caller stacking enough audio that the
    // flat mel buffer would dwarf the BEAM's address space. The check
    // also catches usize overflow on the multiplication: if any of the
    // intermediate `checked_mul`s returns `None`, we treat the request
    // as oversized.
    let elements = total_chunks
        .checked_mul(n_mels)
        .and_then(|n| n.checked_mul(chunk_length));
    let bytes = elements.and_then(|n| n.checked_mul(std::mem::size_of::<f32>()));
    match bytes {
        Some(b) if b > MAX_FEATURE_BUFFER_BYTES => {
            return Err(invalid_request(format!(
                "batch feature buffer {b} bytes exceeds {MAX_FEATURE_BUFFER_BYTES} byte cap; \
                 split the input into smaller transcribe_batch calls"
            )));
        }
        None => {
            return Err(invalid_request(format!(
                "batch feature buffer size overflows usize \
                 (total_chunks={total_chunks}, n_mels={n_mels}, chunk_length={chunk_length})"
            )));
        }
        _ => {}
    }

    let mut flat: Vec<f32> = Vec::with_capacity(elements.expect("checked above"));
    for chunks in &per_audio_chunks {
        for chunk in chunks {
            let slice = chunk
                .as_slice()
                .ok_or_else(|| runtime_error("mel chunk not contiguous"))?;
            if slice.iter().any(|v| !v.is_finite()) {
                return Err(runtime_error(
                    "mel features contain NaN or infinity; check for corrupted PCM input",
                ));
            }
            flat.extend_from_slice(slice);
        }
    }

    let features = StorageView::new(
        &[total_chunks, n_mels, chunk_length],
        &mut flat,
        Device::CPU,
    )
    .map_err(|e| anyhow!("StorageView::new failed: {e}"))?;

    // faster-whisper prepends a space before tokenising both initial_prompt
    // and prefix so the first BPE token carries the leading-space marker.
    // Skipping it leaves the model wedged between a "continuation" subword
    // and the SOT block, which empties the output on certain prompts.
    let initial_prompt_tokens = match &request.initial_prompt {
        Some(text) if !text.trim().is_empty() => {
            encode_plain(tokenizer, &format!(" {}", text.trim()))?
        }
        _ => Vec::new(),
    };
    let prefix_tokens = match &request.prefix {
        Some(text) if !text.trim().is_empty() => {
            encode_plain(tokenizer, &format!(" {}", text.trim()))?
        }
        _ => Vec::new(),
    };

    let emit_timestamps = request.with_timestamps || request.word_timestamps;

    let mut prompts: Vec<Vec<String>> = Vec::with_capacity(total_chunks);
    for (audio_idx, chunks) in per_audio_chunks.iter().enumerate() {
        let lang_token = &per_audio_languages[audio_idx];
        for _ in 0..chunks.len() {
            let parts = PromptParts {
                sot: SOT,
                startofprev: STARTOFPREV,
                language_token: lang_token,
                transcribe: TRANSCRIBE,
                no_timestamps: NO_TIMESTAMPS,
                initial_prompt: &initial_prompt_tokens,
                prefix: &prefix_tokens,
                with_timestamps: emit_timestamps,
                multilingual,
            };
            prompts.push(parts.build());
        }
    }
    let prompt_refs: Vec<Vec<&str>> = prompts
        .iter()
        .map(|p| p.iter().map(String::as_str).collect())
        .collect();

    // Encode once; reuse for generate and (optionally) align.
    let encoder_output = whisper
        .encode(&features, false)
        .map_err(|e| anyhow!("Whisper::encode failed: {e}"))?;

    let mut opts = request.options.clone();
    opts.return_no_speech_prob = true;
    opts.return_scores = true;

    let generated = whisper
        .generate(&encoder_output, &prompt_refs, &opts)
        .map_err(|e| anyhow!("Whisper::generate failed: {e}"))?;
    if generated.len() != total_chunks {
        return Err(anyhow!(
            "expected {} generation results, got {}",
            total_chunks,
            generated.len()
        ));
    }

    let chunk_duration_s = preprocessor.n_samples as f32 / preprocessor.sampling_rate as f32;

    let mut chunk_state: Vec<ChunkState> = Vec::with_capacity(total_chunks);
    for (audio_idx, chunks) in per_audio_chunks.iter().enumerate() {
        for within_audio_idx in 0..chunks.len() {
            let chunk_offset_s = within_audio_idx as f32 * chunk_duration_s;
            let global_idx = chunk_offsets[audio_idx] + within_audio_idx;

            let token_ids = generated[global_idx]
                .sequences_ids
                .first()
                .ok_or_else(|| anyhow!("generation result missing first hypothesis"))?;
            let token_ids_u32: Vec<u32> = token_ids
                .iter()
                .map(|id| u32::try_from(*id).map_err(|_| anyhow!("token id {id} exceeds u32")))
                .collect::<Result<_>>()?;

            chunk_state.push(ChunkState {
                sub_segments: split_sub_segments(
                    &token_ids_u32,
                    specials.timestamp_begin,
                    chunk_duration_s,
                ),
                offset_s: chunk_offset_s,
                num_frames: encoder_frames_for_chunk(
                    audios[audio_idx].len(),
                    within_audio_idx,
                    preprocessor,
                ),
            });
        }
    }

    let words_per_chunk: Vec<Vec<Vec<crate::align::Word>>> = if request.word_timestamps {
        // sys::Whisper::align takes one start_sequence for the whole batch,
        // so every audio in the batch must share the same SOT block we
        // used at generate time. For multilingual auto-detect this means
        // every audio's detected language must match.
        let align_start_sequence =
            build_align_start_sequence(tokenizer, specials, multilingual, &per_audio_languages)?;

        let align_inputs: Vec<ChunkAlignInput<'_>> = chunk_state
            .iter()
            .map(|c| ChunkAlignInput {
                sub_segments: &c.sub_segments,
                chunk_offset_s: c.offset_s,
                num_frames: c.num_frames,
            })
            .collect();
        align_batch(
            whisper,
            tokenizer,
            &encoder_output,
            &align_inputs,
            &align_start_sequence,
            preprocessor.seconds_per_encoder_frame(),
            DEFAULT_MEDIAN_FILTER_WIDTH,
        )?
    } else {
        Vec::new()
    };

    // Materialise per-audio results.
    let mut output: Vec<TranscriptionResult> = Vec::with_capacity(audios.len());
    for audio_idx in 0..audios.len() {
        let mut segments: Vec<SegmentResult> = Vec::new();
        let global_range = chunk_offsets[audio_idx]..chunk_offsets[audio_idx + 1];

        for global_idx in global_range {
            let chunk = std::mem::take(&mut chunk_state[global_idx]);
            let chunk_offset_s = chunk.offset_s;
            let subs = chunk.sub_segments;
            let result = &generated[global_idx];
            // return_scores is forced on in the request, so a missing score
            // is a real bug — not something to paper over with 0.0.
            let avg_logprob = *result.scores.first().ok_or_else(|| {
                anyhow!("ct2 generation result is missing scores despite return_scores=true")
            })?;

            for (sub_idx, sub) in subs.into_iter().enumerate() {
                let text = decode_ids(tokenizer, &sub.text_token_ids)?
                    .trim()
                    .to_owned();
                if text.is_empty() {
                    continue;
                }
                let words = if request.word_timestamps {
                    let aligned_chunk = words_per_chunk.get(global_idx).ok_or_else(|| {
                        anyhow!(
                            "word_timestamps: alignment result missing for chunk {global_idx} \
                             (expected {total_chunks} chunks, got {})",
                            words_per_chunk.len()
                        )
                    })?;
                    let ws = aligned_chunk.get(sub_idx).ok_or_else(|| {
                        anyhow!(
                            "word_timestamps: alignment result missing for sub-segment \
                             {sub_idx} of chunk {global_idx} ({} sub-segments aligned)",
                            aligned_chunk.len()
                        )
                    })?;
                    Some(
                        ws.iter()
                            .map(|w| WordResult {
                                text: w.text.clone(),
                                start: w.start,
                                end: w.end,
                                probability: w.probability,
                            })
                            .collect::<Vec<_>>(),
                    )
                } else {
                    None
                };

                segments.push(SegmentResult {
                    text,
                    start: chunk_offset_s + sub.start_in_chunk,
                    end: chunk_offset_s + sub.end_in_chunk,
                    no_speech_prob: result.no_speech_prob,
                    avg_logprob,
                    tokens: sub.text_token_ids,
                    words,
                });
            }
        }

        let lang_token = &per_audio_languages[audio_idx];
        let language = lang_token
            .trim_start_matches("<|")
            .trim_end_matches("|>")
            .to_owned();

        output.push(TranscriptionResult {
            language,
            duration_s: per_audio_duration[audio_idx],
            segments,
        });
    }

    Ok(output)
}

fn detect_language(
    whisper: &Whisper,
    chunk: &ndarray::Array2<f32>,
    preprocessor: &Preprocessor,
) -> Result<String> {
    let mut buf = chunk
        .as_slice()
        .ok_or_else(|| anyhow!("mel chunk not contiguous"))?
        .to_vec();
    let features = StorageView::new(
        &[1, preprocessor.feature_size, preprocessor.nb_max_frames],
        &mut buf,
        Device::CPU,
    )
    .map_err(|e| anyhow!("StorageView::new for detect_language: {e}"))?;
    let result = whisper
        .detect_language(&features)
        .map_err(|e| anyhow!("Whisper::detect_language failed: {e}"))?;
    let detected = result
        .into_iter()
        .next()
        .and_then(|v| v.into_iter().next())
        .ok_or_else(|| anyhow!("detect_language returned no candidates"))?;
    Ok(detected.language)
}

/// Builds the SOT block used by `Whisper::align`, mirroring the prompt
/// shape `generate` was given so word boundaries land where they were
/// scored. `*.en` checkpoints get `[sot, no_timestamps]`; multilingual
/// gets `[sot, lang, transcribe, no_timestamps]`.
///
/// Errors out when the batch mixes detected languages: `sys::Whisper::align`
/// only accepts one start_sequence for the whole batch, so the caller has
/// to split the work or pin `:language`.
fn build_align_start_sequence(
    tokenizer: &hf::Tokenizer,
    specials: &SpecialTokens,
    multilingual: bool,
    per_audio_languages: &[String],
) -> Result<Vec<usize>> {
    if !multilingual {
        return Ok(vec![specials.sot as usize, specials.no_timestamps as usize]);
    }
    let first = uniform_align_language(per_audio_languages)?;
    let lang_id = token_id(tokenizer, first)?;
    Ok(vec![
        specials.sot as usize,
        lang_id as usize,
        specials.transcribe as usize,
        specials.no_timestamps as usize,
    ])
}

/// Guards against the silent-mismatch case where a caller pins
/// `:language` to a non-English code on an English-only checkpoint
/// (`*.en`). Decoding ignores the pinned token (the SOT block on `*.en`
/// is just `[<|startoftranscript|>]`) and runs English, but the
/// returned `language` would still echo the pinned code — misrouting
/// any downstream language-based logic.
fn reject_non_english_on_en_checkpoint(multilingual: bool, lang: &str) -> Result<()> {
    if !multilingual && lang != "en" {
        return Err(invalid_request(format!(
            "language {lang:?} requested on an English-only checkpoint; \
             only \"en\" is valid (or omit :language). Use a multilingual \
             checkpoint to transcribe other languages."
        )));
    }
    Ok(())
}

/// Pure check used by `build_align_start_sequence`: returns the common
/// language token of the batch, or an `invalid_request` error if the
/// languages disagree. Extracted so the mixed-language guard can be
/// unit-tested without a loaded tokenizer.
fn uniform_align_language(per_audio_languages: &[String]) -> Result<&str> {
    let first = per_audio_languages
        .first()
        .ok_or_else(|| runtime_error("align: no audios in batch"))?;
    if per_audio_languages.iter().any(|l| l != first) {
        return Err(invalid_request(format!(
            "word_timestamps requires every audio in a batch to share the same \
             resolved language; got {per_audio_languages:?}. Pin :language or \
             split the batch."
        )));
    }
    Ok(first.as_str())
}

/// Builds the chunk-offset prefix-sum used to translate
/// `(audio_idx, within_audio_idx)` into a flat batch index. Result has
/// length `chunk_counts.len() + 1`, with `offsets[i]..offsets[i+1]` covering
/// audio `i`. Pulled out of `transcribe_many` so the index arithmetic can
/// be exercised in isolation.
fn compute_chunk_offsets(chunk_counts: &[usize]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(chunk_counts.len() + 1);
    offsets.push(0);
    for n in chunk_counts {
        let prev = *offsets.last().expect("seeded above");
        offsets.push(prev + n);
    }
    offsets
}

/// Number of valid frames in chunk `chunk_idx` of a `samples_len`-sample
/// audio, in the units `sys::Whisper::align` expects.
///
/// Despite the API doc saying "encoder frames", faster-whisper passes the
/// mel-frame count (`samples / hop_length`, ~100 Hz) and that is what
/// produces correct DTW output — passing the encoder-frame count
/// (`mel / 2`, ~50 Hz) compresses every word into the first half of the
/// clip. We follow faster-whisper.
fn encoder_frames_for_chunk(
    samples_len: usize,
    chunk_idx: usize,
    preprocessor: &Preprocessor,
) -> usize {
    let start = chunk_idx * preprocessor.n_samples;
    let remaining = samples_len.saturating_sub(start);
    let chunk_samples = remaining.min(preprocessor.n_samples);
    (chunk_samples / preprocessor.hop_length)
        .min(preprocessor.nb_max_frames)
        .max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::kind_from_chain;
    use ndarray::Array2;

    fn tiny_preprocessor() -> Preprocessor {
        Preprocessor {
            feature_size: 80,
            hop_length: 160,
            n_fft: 400,
            n_samples: 480_000,
            nb_max_frames: 3_000,
            sampling_rate: 16_000,
            mel_filters: Array2::<f64>::zeros((80, 201)),
        }
    }

    #[test]
    fn compute_chunk_offsets_is_a_prefix_sum() {
        assert_eq!(compute_chunk_offsets(&[]), vec![0]);
        assert_eq!(compute_chunk_offsets(&[3]), vec![0, 3]);
        // Three audios, each 2 / 5 / 1 chunks — offsets must let us
        // recover any (audio_idx, within_audio_idx) -> global_idx with
        // `offsets[audio_idx] + within_audio_idx`.
        let offsets = compute_chunk_offsets(&[2, 5, 1]);
        assert_eq!(offsets, vec![0, 2, 7, 8]);

        let ranges: Vec<_> = (0..3).map(|i| offsets[i]..offsets[i + 1]).collect();
        assert_eq!(ranges[0].clone().collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(ranges[1].clone().collect::<Vec<_>>(), vec![2, 3, 4, 5, 6]);
        assert_eq!(ranges[2].clone().collect::<Vec<_>>(), vec![7]);
    }

    #[test]
    fn encoder_frames_for_chunk_clamps_to_nb_max_frames() {
        let preprocessor = tiny_preprocessor();

        // Full 30 s chunk: samples / hop = 480_000 / 160 = 3000 frames,
        // matches `nb_max_frames` exactly. Clamp is a no-op.
        assert_eq!(encoder_frames_for_chunk(480_000, 0, &preprocessor), 3000);

        // Partial first chunk: half-second of audio = 8000 samples →
        // 8000 / 160 = 50 frames.
        assert_eq!(encoder_frames_for_chunk(8_000, 0, &preprocessor), 50);

        // Second chunk of 35 s audio: only 5 s remain → 80_000 / 160 = 500.
        assert_eq!(encoder_frames_for_chunk(560_000, 1, &preprocessor), 500);

        // Tail past the end must clamp to a non-zero minimum so `align`
        // never sees `num_frames = 0` (which the DTW path would divide by).
        assert_eq!(encoder_frames_for_chunk(480_000, 5, &preprocessor), 1);
    }

    #[test]
    fn uniform_align_language_passes_through_matching_batch() {
        let langs = vec!["<|en|>".to_owned(), "<|en|>".to_owned()];
        assert_eq!(uniform_align_language(&langs).unwrap(), "<|en|>");
    }

    #[test]
    fn uniform_align_language_rejects_mixed_languages_as_invalid_request() {
        // Mixed-language batch + word_timestamps is a caller bug, not an
        // inference failure. The error category must reflect that so
        // Elixir surfaces `:invalid_request`.
        let langs = vec!["<|en|>".to_owned(), "<|de|>".to_owned()];
        let err = uniform_align_language(&langs).unwrap_err();
        assert_eq!(kind_from_chain(&err), Some("invalid_request"));
        let msg = format!("{err:#}");
        assert!(msg.contains("word_timestamps"), "got: {msg}");
        assert!(
            msg.contains("<|en|>") && msg.contains("<|de|>"),
            "got: {msg}"
        );
    }

    #[test]
    fn reject_non_english_on_en_checkpoint_allows_en() {
        assert!(reject_non_english_on_en_checkpoint(false, "en").is_ok());
    }

    #[test]
    fn reject_non_english_on_en_checkpoint_allows_anything_on_multilingual() {
        // Multilingual checkpoints decode the pinned language for real, so
        // the guard must not fire there.
        assert!(reject_non_english_on_en_checkpoint(true, "de").is_ok());
        assert!(reject_non_english_on_en_checkpoint(true, "fr").is_ok());
    }

    #[test]
    fn reject_non_english_on_en_checkpoint_rejects_non_en_as_invalid_request() {
        // Pinning :language to anything but "en" on an `.en` checkpoint
        // would otherwise be silently ignored by the prompt and then echoed
        // back in TranscriptionResult.language, misrouting downstream
        // language-based logic.
        let err = reject_non_english_on_en_checkpoint(false, "de").unwrap_err();
        assert_eq!(kind_from_chain(&err), Some("invalid_request"));
        let msg = format!("{err:#}");
        assert!(msg.contains("English-only"), "got: {msg}");
        assert!(msg.contains("\"de\""), "got: {msg}");
    }

    #[test]
    fn uniform_align_language_empty_batch_is_runtime_error() {
        // Reaching this guard with an empty per_audio_languages indicates
        // a NIF-internal bug (transcribe_many should already have early-
        // returned), so the category is runtime rather than invalid.
        let langs: Vec<String> = Vec::new();
        let err = uniform_align_language(&langs).unwrap_err();
        assert_eq!(kind_from_chain(&err), Some("runtime_error"));
    }
}
