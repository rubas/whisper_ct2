//! Word-level alignment.
//!
//! After `sys::Whisper::generate` we have per-chunk token sequences. To get
//! word boundaries we feed the encoder output and the **text** tokens of
//! every chunk into `sys::Whisper::align` in one batched call, then group
//! each chunk's per-text-token frame indices into Whisper words the way
//! faster-whisper does: tokens are first merged into the smallest runs
//! that decode to complete UTF-8 through the tokenizer's real decoder
//! (`split_tokens_on_unicode`), then split into words on leading spaces
//! and isolated punctuation (`split_tokens_on_spaces`) — or kept one
//! codepoint-run per word for languages written without spaces.
//!
//! The raw DTW boundaries get the same post-processing faster-whisper
//! applies before returning per-word timings:
//!
//! 1. **Median-duration + sentence-end clamp.** Words that absorbed
//!    silence end up multi-second long; we cap them at `2 * median(word
//!    duration)` when they border a sentence-end mark (`.。!！?？`).
//! 2. **`merge_punctuations`.** Standalone punctuation tokens
//!    (`,`, `.`, `"`, `(`, ...) get folded back into the adjacent real
//!    word, mirroring faster-whisper's `prepend_punctuations` /
//!    `append_punctuations` defaults. The merged-into word keeps its
//!    original (already-clamped) end, which is the punctuation token's
//!    frame — so the silence after `"so,"` doesn't get attributed to
//!    `"so,"`.
//! 3. **Subsegment snap.** Each generated `<|t_start|> text <|t_end|>`
//!    block carries its own timestamps; when the DTW first/last word
//!    drifts past those, we snap it back. Same rule faster-whisper uses
//!    to avoid first-word overshoot at chunk boundaries.

// Frame indices and token ids stay within u32; the pedantic cast lints
// don't catch any real bug here.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

use anyhow::{Result, anyhow};
use ct2rs::sys::{StorageView, Whisper};
use ct2rs::tokenizers::hf;
use tokenizers::Tokenizer as InnerTokenizer;

use crate::tokens::SubSegment;

/// One word with absolute time span.
#[derive(Debug)]
pub(crate) struct Word {
    pub(crate) text: String,
    pub(crate) start: f32,
    pub(crate) end: f32,
    pub(crate) probability: f32,
}

pub(crate) const DEFAULT_MEDIAN_FILTER_WIDTH: i64 = 7;

/// Punctuation lists mirror faster-whisper's defaults
/// (`transcribe.py:286-287`). Single-char tokens whose stripped form is in
/// one of these strings are folded into the adjacent real word by
/// `merge_punctuations`.
const PREPEND_PUNCT: &str = "\"'\u{201C}\u{00BF}([{-";
const APPEND_PUNCT: &str = "\"'.\u{3002},\u{FF0C}!\u{FF01}?\u{FF1F}:\u{FF1A}\u{201D})]}\u{3001}";
/// Sentence-final marks used to decide which side of an over-long word
/// gets clamped.
const SENTENCE_END_MARKS: &[&str] = &[".", "\u{3002}", "!", "\u{FF01}", "?", "\u{FF1F}"];
/// Upper bound on the per-chunk median word duration (seconds).
/// faster-whisper caps the median at 0.7 s so a chunk full of long words
/// can't lift the clamp threshold high enough to be useless.
const MEDIAN_DURATION_CAP_S: f32 = 0.7;

/// Languages written without spaces between words. faster-whisper
/// (`tokenizer.py::split_to_word_tokens`) word-splits these purely on
/// decoded codepoint runs; space/punctuation splitting would glue a whole
/// segment into one word.
const UNICODE_SPLIT_LANGUAGES: &[&str] = &["zh", "ja", "th", "lo", "my", "yue"];

fn splits_words_on_unicode_only(language_token: &str) -> bool {
    let code = language_token
        .trim_start_matches("<|")
        .trim_end_matches("|>");
    UNICODE_SPLIT_LANGUAGES.contains(&code)
}

/// Per-chunk input to `align_batch`.
pub(crate) struct ChunkAlignInput<'a> {
    pub(crate) sub_segments: &'a [SubSegment],
    pub(crate) chunk_offset_s: f32,
    pub(crate) num_frames: usize,
}

/// Batch-level alignment parameters shared by every chunk.
pub(crate) struct AlignParams<'a> {
    /// SOT block mirroring the prompt used at `generate` time: `[sot]`
    /// for `*.en` checkpoints, `[sot, lang, transcribe]` for multilingual
    /// — without `<|notimestamps|>`, which CTranslate2's `align` appends
    /// internally. `sys::Whisper::align` takes a single start sequence
    /// for the whole batch, so all chunks must have been generated
    /// against the same one.
    pub(crate) start_sequence: &'a [usize],
    /// The batch's resolved language token (`"<|en|>"`, `"<|de|>"`, ...);
    /// decides whether words split on spaces or on codepoint runs.
    pub(crate) language_token: &'a str,
    pub(crate) seconds_per_frame: f32,
    pub(crate) median_filter_width: i64,
}

/// Runs one batched `align` call across every chunk in `encoder_output` and
/// returns post-processed per-sub-segment word lists.
pub(crate) fn align_batch(
    whisper: &Whisper,
    tokenizer: &hf::Tokenizer,
    encoder_output: &StorageView<'_>,
    chunks: &[ChunkAlignInput<'_>],
    params: &AlignParams<'_>,
) -> Result<Vec<Vec<Vec<Word>>>> {
    if chunks.is_empty() {
        return Ok(Vec::new());
    }

    let mut text_tokens_per_chunk: Vec<Vec<usize>> = Vec::with_capacity(chunks.len());
    let mut spans_per_chunk: Vec<Vec<std::ops::Range<usize>>> = Vec::with_capacity(chunks.len());
    let mut num_frames: Vec<usize> = Vec::with_capacity(chunks.len());

    for chunk in chunks {
        let mut flat: Vec<usize> = Vec::new();
        let mut spans: Vec<std::ops::Range<usize>> = Vec::new();
        for seg in chunk.sub_segments {
            let start = flat.len();
            flat.extend(seg.text_token_ids.iter().map(|id| *id as usize));
            spans.push(start..flat.len());
        }
        text_tokens_per_chunk.push(flat);
        spans_per_chunk.push(spans);
        num_frames.push(chunk.num_frames);
    }

    if text_tokens_per_chunk.iter().all(Vec::is_empty) {
        return Ok(chunks
            .iter()
            .map(|c| (0..c.sub_segments.len()).map(|_| Vec::new()).collect())
            .collect());
    }

    let results = whisper
        .align(
            encoder_output,
            params.start_sequence,
            &text_tokens_per_chunk,
            &num_frames,
            params.median_filter_width,
        )
        .map_err(|e| anyhow!("Whisper::align failed: {e}"))?;
    if results.len() != chunks.len() {
        return Err(anyhow!(
            "expected {} alignment results, got {}",
            chunks.len(),
            results.len()
        ));
    }

    let inner: &InnerTokenizer = tokenizer;
    let unicode_only = splits_words_on_unicode_only(params.language_token);
    let seconds_per_frame = params.seconds_per_frame;
    let mut out: Vec<Vec<Vec<Word>>> = Vec::with_capacity(chunks.len());

    for (chunk_idx, alignment) in results.into_iter().enumerate() {
        let flat = &text_tokens_per_chunk[chunk_idx];
        let spans = &spans_per_chunk[chunk_idx];
        let chunk = &chunks[chunk_idx];

        if flat.is_empty() {
            out.push((0..chunk.sub_segments.len()).map(|_| Vec::new()).collect());
            continue;
        }

        let (token_frames, eot_frame) = token_first_frames(&alignment.alignments, flat.len());
        let token_probs = &alignment.text_token_probs;

        // 1. Pre-words: decode token runs and split into words, matching
        //    faster-whisper's `split_to_word_tokens`. Each pre-word
        //    carries the subsegment it came from.
        let mut pre_words = group_pre_words(
            inner,
            flat,
            token_probs,
            &token_frames,
            spans,
            eot_frame,
            unicode_only,
        )?;

        // 2. Median-driven sentence-end clamp on the still-pre-merge list.
        let max_duration_frames = compute_max_duration_frames(&pre_words, seconds_per_frame);
        clamp_sentence_boundaries(&mut pre_words, max_duration_frames);

        // 3. Fold prepend/append punctuation back into the adjacent word.
        merge_punctuations(&mut pre_words);

        // 4. Partition surviving words back to sub-segments, then snap each
        //    sub-segment's first/last word to the `<|t_..|>` timestamps.
        let mut per_sub: Vec<Vec<Word>> = (0..spans.len()).map(|_| Vec::new()).collect();
        for pw in pre_words.iter().filter(|w| !w.text.is_empty()) {
            let words_of_sub = &mut per_sub[pw.subsegment_idx];
            words_of_sub.push(materialise_word(
                pw,
                chunk.chunk_offset_s,
                seconds_per_frame,
            ));
        }
        for (sub_idx, sub) in chunk.sub_segments.iter().enumerate() {
            snap_subsegment_boundaries(
                &mut per_sub[sub_idx],
                chunk.chunk_offset_s + sub.start_in_chunk,
                chunk.chunk_offset_s + sub.end_in_chunk,
                MEDIAN_DURATION_CAP_S.min(max_duration_frames as f32 * seconds_per_frame / 2.0),
            );
        }

        out.push(per_sub);
    }
    Ok(out)
}

/// One word in its pre-merge form. Frames are encoder frames within the
/// chunk; conversion to absolute seconds happens in `materialise_word`.
struct PreWord {
    text: String,
    tokens: Vec<u32>,
    probs: Vec<f32>,
    start_frame: usize,
    /// First frame of the next pre-word (or `last_token_frame + 1` for the
    /// chunk's last pre-word). Becomes the word's `end` after conversion.
    end_frame: usize,
    subsegment_idx: usize,
}

impl PreWord {
    fn duration_frames(&self) -> usize {
        self.end_frame.saturating_sub(self.start_frame)
    }
}

/// For each text-token index in `[0, n_tokens)`, returns the first encoder
/// frame the DTW path assigns to it. Tokens missing from the path inherit
/// the previous token's frame.
///
/// The second value is the first frame of the EOT row: `CTranslate2`
/// aligns `text_tokens + [eot]`, so the path's entry into row `n_tokens`
/// is the model's own boundary after the last word — what faster-whisper
/// takes as the final `end_time`. Indices past `n_tokens` are padding
/// noise and stay ignored.
fn token_first_frames(
    alignments: &[ct2rs::sys::WhisperTokenAlignment],
    n_tokens: usize,
) -> (Vec<usize>, Option<usize>) {
    let mut out = vec![usize::MAX; n_tokens];
    let mut eot_frame: Option<usize> = None;
    for align in alignments {
        let Ok(token_x) = usize::try_from(align.token_x) else {
            continue;
        };
        let frame_x = align.frame_x.max(0) as usize;
        if token_x < n_tokens {
            if out[token_x] == usize::MAX {
                out[token_x] = frame_x;
            }
        } else if token_x == n_tokens {
            eot_frame = Some(eot_frame.map_or(frame_x, |f| f.min(frame_x)));
        }
    }
    let mut last = 0usize;
    for slot in &mut out {
        if *slot == usize::MAX {
            *slot = last;
        } else {
            last = *slot;
        }
    }
    (out, eot_frame)
}

const REPLACEMENT_CHAR: char = '\u{FFFD}';

/// The smallest run of tokens that decodes to complete UTF-8, plus the
/// indices of those tokens in the chunk's flat token list. One element of
/// faster-whisper's `split_tokens_on_unicode` output.
struct SubWord {
    text: String,
    flat_indices: Vec<usize>,
}

/// Splits one sub-segment's tokens into the smallest runs that decode to
/// complete UTF-8 through the tokenizer's real decoder (byte-level BPE).
/// Mirrors faster-whisper's `split_tokens_on_unicode`: accumulate tokens
/// until the decoded text carries no dangling `U+FFFD` — a replacement
/// char counts as dangling unless the full decoding genuinely contains
/// one at that position. Vocab surfaces (`id_to_token`) must never be
/// used as word text: they are in the GPT-2 byte alphabet, where every
/// non-ASCII character is mojibake (`"schön"` surfaces as `"schÃ¶n"`).
fn split_tokens_on_unicode(
    inner: &InnerTokenizer,
    token_ids: &[usize],
    span: std::ops::Range<usize>,
) -> Result<Vec<SubWord>> {
    let ids_u32 = span
        .clone()
        .map(|idx| {
            let id = token_ids[idx];
            u32::try_from(id).map_err(|_| anyhow!("token id {id} exceeds u32"))
        })
        .collect::<Result<Vec<u32>>>()?;
    let decoded_full: Vec<char> = decode_text(inner, &ids_u32)?.chars().collect();

    let mut out: Vec<SubWord> = Vec::new();
    let mut current_ids: Vec<u32> = Vec::new();
    let mut current_indices: Vec<usize> = Vec::new();
    let mut unicode_offset = 0usize;

    for (pos, &id) in ids_u32.iter().enumerate() {
        current_ids.push(id);
        current_indices.push(span.start + pos);
        let decoded = decode_text(inner, &current_ids)?;
        let dangling = decoded
            .chars()
            .position(|c| c == REPLACEMENT_CHAR)
            .is_some_and(|at| decoded_full.get(unicode_offset + at) != Some(&REPLACEMENT_CHAR));
        if dangling {
            continue;
        }
        unicode_offset += decoded.chars().count();
        out.push(SubWord {
            text: decoded,
            flat_indices: std::mem::take(&mut current_indices),
        });
        current_ids.clear();
    }
    // Bytes that never complete a codepoint (generation truncated
    // mid-character) still surface, replacement char and all, instead of
    // dropping their tokens' timing slots.
    if !current_indices.is_empty() {
        out.push(SubWord {
            text: decode_text(inner, &current_ids)?,
            flat_indices: current_indices,
        });
    }
    Ok(out)
}

fn decode_text(inner: &InnerTokenizer, ids: &[u32]) -> Result<String> {
    inner
        .decode(ids, true)
        .map_err(|e| anyhow!("failed to decode tokens: {e}"))
}

/// Groups a chunk's tokens into pre-words. Token runs come from
/// [`split_tokens_on_unicode`]; with `unicode_only` every run is its own
/// word, otherwise a new word starts on:
///
/// - a run whose decoded text has a leading space;
/// - an isolated punctuation run (single non-space ASCII character).
///
/// The second rule mirrors faster-whisper's `split_tokens_on_spaces`: it
/// lets the punctuation token surface as its own (very short) word that
/// then gets folded into the adjacent real word by `merge_punctuations`,
/// keeping the real word's end time at the punctuation token's frame
/// instead of the next word's.
fn group_pre_words(
    inner: &InnerTokenizer,
    token_ids: &[usize],
    token_probs: &[f32],
    token_frames: &[usize],
    spans: &[std::ops::Range<usize>],
    eot_frame: Option<usize>,
    unicode_only: bool,
) -> Result<Vec<PreWord>> {
    let mut words: Vec<PreWord> = Vec::new();

    for (sub_idx, span) in spans.iter().enumerate() {
        let mut current: Option<PreWord> = None;
        for sub_word in split_tokens_on_unicode(inner, token_ids, span.clone())? {
            let starts_word =
                unicode_only || sub_word.text.starts_with(' ') || is_isolated_punct(&sub_word.text);

            match current.as_mut() {
                Some(w) if !starts_word => {
                    w.text.push_str(&sub_word.text);
                    for &idx in &sub_word.flat_indices {
                        w.tokens.push(token_ids[idx] as u32);
                        w.probs.push(token_probs[idx]);
                    }
                }
                _ => {
                    if let Some(prev) = current.take() {
                        words.push(prev);
                    }
                    let frame = token_frames[sub_word.flat_indices[0]];
                    current = Some(PreWord {
                        tokens: sub_word
                            .flat_indices
                            .iter()
                            .map(|&idx| token_ids[idx] as u32)
                            .collect(),
                        probs: sub_word
                            .flat_indices
                            .iter()
                            .map(|&idx| token_probs[idx])
                            .collect(),
                        text: sub_word.text,
                        start_frame: frame,
                        end_frame: frame, // filled in below
                        subsegment_idx: sub_idx,
                    });
                }
            }
        }
        if let Some(prev) = current {
            words.push(prev);
        }
    }

    // Set each pre-word's end_frame to the next pre-word's start. The
    // chunk's last pre-word ends where the DTW path first enters the EOT
    // row — the model's own boundary after the final word. Without that
    // pair (defensive: the DTW backtrace always reaches the EOT row) keep
    // the old one-frame extension.
    for i in 0..words.len() {
        let end = if i + 1 < words.len() {
            words[i + 1].start_frame
        } else {
            eot_frame.unwrap_or(words[i].start_frame + 1)
        };
        words[i].end_frame = end.max(words[i].start_frame);
    }

    Ok(words)
}

fn is_isolated_punct(s: &str) -> bool {
    let trimmed = s.trim();
    let mut chars = trimmed.chars();
    let Some(c) = chars.next() else {
        return false;
    };
    chars.next().is_none() && c.is_ascii_punctuation()
}

/// Caps any pre-word whose `[start, end]` exceeds `max_duration_frames`
/// and that borders (`words[i-1]` or `words[i]`) a sentence-end mark.
/// Mirrors `faster_whisper.transcribe._WhisperModel._add_word_timestamps`
/// lines 1607-1616.
fn clamp_sentence_boundaries(words: &mut [PreWord], max_duration_frames: usize) {
    if max_duration_frames == 0 {
        return;
    }
    let is_sentence_end = |s: &str| SENTENCE_END_MARKS.iter().any(|m| s.trim() == *m);
    for i in 1..words.len() {
        if words[i].duration_frames() > max_duration_frames {
            if is_sentence_end(&words[i].text) {
                words[i].end_frame = words[i].start_frame + max_duration_frames;
            } else if is_sentence_end(&words[i - 1].text) {
                words[i].start_frame = words[i].end_frame.saturating_sub(max_duration_frames);
            }
        }
    }
}

/// `2 * median(nonzero word durations)` in frames, capped at 0.7 s worth
/// of frames. Returns 0 when no word has a non-zero duration, which
/// disables the downstream clamp.
fn compute_max_duration_frames(words: &[PreWord], seconds_per_frame: f32) -> usize {
    let mut durations: Vec<usize> = words
        .iter()
        .map(PreWord::duration_frames)
        .filter(|d| *d > 0)
        .collect();
    if durations.is_empty() {
        return 0;
    }
    durations.sort_unstable();
    let median = durations[durations.len() / 2];
    let cap_frames = (MEDIAN_DURATION_CAP_S / seconds_per_frame).round() as usize;
    let capped = median.min(cap_frames);
    capped.saturating_mul(2)
}

/// Folds standalone prepend/append punctuation tokens back into the
/// adjacent real word. Adapted from faster-whisper's
/// `merge_punctuations` (`transcribe.py:1910-1941`). The merged-into
/// word's `start`/`end` are intentionally left alone — the original
/// frame boundaries become the timing of the combined word, which is
/// what gives `"so,"` an end at the comma's frame.
fn merge_punctuations(words: &mut Vec<PreWord>) {
    if words.len() < 2 {
        return;
    }

    // Prepend pass (right to left).
    let mut i = words.len() as isize - 2;
    let mut j: usize = words.len() - 1;
    while i >= 0 {
        let iu = i as usize;
        let candidate = is_single_char_member(&words[iu].text, PREPEND_PUNCT)
            && words[iu].text.starts_with(' ');
        if candidate {
            // Move prev's tokens/probs into following (j), inherit start.
            let prev_text = std::mem::take(&mut words[iu].text);
            let prev_tokens = std::mem::take(&mut words[iu].tokens);
            let prev_probs = std::mem::take(&mut words[iu].probs);
            let prev_start = words[iu].start_frame;

            let following = &mut words[j];
            following.text = prev_text + &following.text;
            following.tokens = [prev_tokens, std::mem::take(&mut following.tokens)].concat();
            following.probs = [prev_probs, std::mem::take(&mut following.probs)].concat();
            following.start_frame = prev_start;
        } else {
            j = iu;
        }
        i -= 1;
    }

    // Append pass (left to right).
    let mut i = 0usize;
    let mut j = 1usize;
    while j < words.len() {
        let mergeable = !words[i].text.is_empty()
            && !words[i].text.ends_with(' ')
            && is_single_char_member(&words[j].text, APPEND_PUNCT)
            && !words[j].text.starts_with(' ');
        if mergeable {
            let foll_text = std::mem::take(&mut words[j].text);
            let foll_tokens = std::mem::take(&mut words[j].tokens);
            let foll_probs = std::mem::take(&mut words[j].probs);

            let previous = &mut words[i];
            previous.text.push_str(&foll_text);
            previous.tokens.extend(foll_tokens);
            previous.probs.extend(foll_probs);
            // previous.end_frame is left alone: it already equals the
            // punctuation word's start_frame (== the punct token's frame).
        } else {
            i = j;
        }
        j += 1;
    }

    words.retain(|w| !w.text.is_empty());
}

fn is_single_char_member(s: &str, set: &str) -> bool {
    let trimmed = s.trim();
    let mut chars = trimmed.chars();
    let Some(c) = chars.next() else {
        return false;
    };
    if chars.next().is_some() {
        return false;
    }
    set.contains(c)
}

/// Aligns a sub-segment's first/last word with the `<|t_..|>` timestamps
/// the model emitted, when the raw DTW boundaries clearly overshoot.
/// Mirrors `_WhisperModel._add_word_timestamps` lines 1670-1692.
fn snap_subsegment_boundaries(
    words: &mut [Word],
    subseg_start_s: f32,
    subseg_end_s: f32,
    median_duration_s: f32,
) {
    if words.is_empty() {
        return;
    }
    let last_idx = words.len() - 1;

    // Snap word[0].start back when DTW placed it more than 0.5 s before
    // the segment's own timestamp.
    let first = &mut words[0];
    if subseg_start_s < first.end && subseg_start_s - 0.5 > first.start {
        first.start = 0.0_f32.max((first.end - median_duration_s).min(subseg_start_s));
    }
    // Snap word[-1].end forward when DTW placed it more than 0.5 s after
    // the segment's own timestamp.
    let last = &mut words[last_idx];
    if subseg_end_s > last.start && subseg_end_s + 0.5 < last.end {
        last.end = (last.start + median_duration_s).max(subseg_end_s);
    }
}

fn materialise_word(pre: &PreWord, chunk_offset_s: f32, seconds_per_frame: f32) -> Word {
    let start = chunk_offset_s + pre.start_frame as f32 * seconds_per_frame;
    let end = chunk_offset_s + pre.end_frame as f32 * seconds_per_frame;
    let probability = if pre.probs.is_empty() {
        0.0
    } else {
        pre.probs.iter().sum::<f32>() / pre.probs.len() as f32
    };
    Word {
        text: pre.text.trim_start().to_owned(),
        start,
        end: end.max(start),
        probability,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pre_word(start: usize, end: usize, text: &str) -> PreWord {
        PreWord {
            text: text.to_owned(),
            tokens: vec![1],
            probs: vec![1.0],
            start_frame: start,
            end_frame: end,
            subsegment_idx: 0,
        }
    }

    #[test]
    fn token_first_frames_fills_missing_with_previous() {
        // Two alignments touching tokens 0 and 2; token 1 has no DTW path
        // assignment and must inherit token 0's frame, not stay at usize::MAX
        // (which would translate to a giant `end` time downstream).
        let alignments = vec![
            ct2rs::sys::WhisperTokenAlignment {
                token_x: 0,
                frame_x: 5,
            },
            ct2rs::sys::WhisperTokenAlignment {
                token_x: 2,
                frame_x: 30,
            },
        ];
        let (frames, eot_frame) = token_first_frames(&alignments, 4);
        assert_eq!(frames, vec![5, 5, 30, 30]);
        assert_eq!(eot_frame, None);
    }

    #[test]
    fn token_first_frames_ignores_out_of_range_token_x() {
        // A token_x past `n_tokens + 1` is library noise and must not
        // panic (the DTW backend can emit padding indices). Only exactly
        // `n_tokens` is the EOT row.
        let alignments = vec![ct2rs::sys::WhisperTokenAlignment {
            token_x: 100,
            frame_x: 7,
        }];
        let (frames, eot_frame) = token_first_frames(&alignments, 3);
        assert_eq!(frames, vec![0, 0, 0]);
        assert_eq!(eot_frame, None);
    }

    #[test]
    fn token_first_frames_captures_eot_row_entry() {
        // CTranslate2 aligns `text_tokens + [eot]`; the DTW path's first
        // entry into row `n_tokens` is the true end of the chunk's last
        // word and must be surfaced, not filtered as noise.
        let alignments = vec![
            ct2rs::sys::WhisperTokenAlignment {
                token_x: 0,
                frame_x: 5,
            },
            ct2rs::sys::WhisperTokenAlignment {
                token_x: 1,
                frame_x: 9,
            },
            ct2rs::sys::WhisperTokenAlignment {
                token_x: 2,
                frame_x: 30,
            },
            ct2rs::sys::WhisperTokenAlignment {
                token_x: 2,
                frame_x: 45,
            },
        ];
        let (frames, eot_frame) = token_first_frames(&alignments, 2);
        assert_eq!(frames, vec![5, 9]);
        assert_eq!(eot_frame, Some(30));
    }

    #[test]
    fn compute_max_duration_frames_returns_2x_median() {
        // Durations 10, 20, 30 frames; median = 20; 2x median = 40, well
        // under the 0.7 s cap at seconds_per_frame=0.02 (35 frames).
        // So the cap dominates: min(20, 35) * 2 = 40 .. wait — cap is
        // `min(median, cap_frames)`, then `* 2`. cap_frames at 0.02 s/frame
        // = 35. min(20, 35) = 20 → 40.
        let words = vec![
            make_pre_word(0, 10, "a"),
            make_pre_word(0, 20, "b"),
            make_pre_word(0, 30, "c"),
        ];
        assert_eq!(compute_max_duration_frames(&words, 0.02), 40);
    }

    #[test]
    fn compute_max_duration_frames_caps_at_median_duration_cap() {
        // Median 100 frames > 35-frame cap → uses cap, doubled to 70.
        let words = vec![
            make_pre_word(0, 80, "a"),
            make_pre_word(0, 100, "b"),
            make_pre_word(0, 120, "c"),
        ];
        assert_eq!(compute_max_duration_frames(&words, 0.02), 70);
    }

    #[test]
    fn compute_max_duration_frames_returns_zero_with_no_nonzero_durations() {
        // All-zero-duration input must yield 0 — that is the sentinel that
        // disables the sentence-end clamp downstream.
        let words = vec![make_pre_word(5, 5, "a"), make_pre_word(7, 7, "b")];
        assert_eq!(compute_max_duration_frames(&words, 0.02), 0);
    }

    #[test]
    fn clamp_sentence_boundaries_caps_word_after_sentence_end() {
        // `group_pre_words` splits standalone punctuation into its own
        // pre-word, so a sentence end here is literally `"."`. `is_sentence_end`
        // checks `text.trim() == mark` (not "ends with"). When the previous
        // word is the mark and the next word is too long, trim the next
        // word's start so silence after the period doesn't get absorbed.
        let mut words = vec![make_pre_word(0, 10, "."), make_pre_word(10, 200, "bar")];
        clamp_sentence_boundaries(&mut words, 40);
        assert_eq!(words[1].start_frame, 160);
        assert_eq!(words[1].end_frame, 200);
    }

    #[test]
    fn clamp_sentence_boundaries_caps_sentence_end_word_itself() {
        // `"foo" | "."` where `"."` is too long — `"."` is a sentence end,
        // so trim its end, not its start (its start is the previous word's
        // boundary).
        let mut words = vec![make_pre_word(0, 50, "foo"), make_pre_word(50, 250, ".")];
        clamp_sentence_boundaries(&mut words, 40);
        assert_eq!(words[1].start_frame, 50);
        assert_eq!(words[1].end_frame, 90);
    }

    #[test]
    fn clamp_sentence_boundaries_is_a_noop_with_zero_threshold() {
        // Threshold 0 means "no median was computable" → disable clamping
        // entirely (otherwise every word would be flagged as too long).
        let mut words = vec![make_pre_word(0, 10, "foo."), make_pre_word(10, 200, "bar")];
        clamp_sentence_boundaries(&mut words, 0);
        assert_eq!(words[1].start_frame, 10);
        assert_eq!(words[1].end_frame, 200);
    }

    #[test]
    fn is_isolated_punct_matches_single_ascii_punct() {
        assert!(is_isolated_punct(","));
        assert!(is_isolated_punct(" ."));
        assert!(!is_isolated_punct("hello"));
        assert!(!is_isolated_punct(",,"));
        assert!(!is_isolated_punct(""));
    }

    #[test]
    fn splits_words_on_unicode_only_for_spaceless_languages() {
        assert!(splits_words_on_unicode_only("<|zh|>"));
        assert!(splits_words_on_unicode_only("<|ja|>"));
        assert!(!splits_words_on_unicode_only("<|en|>"));
        assert!(!splits_words_on_unicode_only("<|de|>"));
    }

    /// Minimal byte-level BPE tokenizer matching the shape of a real
    /// Whisper `tokenizer.json`: vocab surfaces are in the GPT-2 byte
    /// alphabet (`Ġ` = space, `Ã¶` = the two UTF-8 bytes of `ö`), and the
    /// ByteLevel decoder maps them back to bytes before UTF-8 decoding.
    fn byte_level_tokenizer(entries: &[(&str, u32)]) -> InnerTokenizer {
        use tokenizers::DecoderWrapper;
        use tokenizers::decoders::byte_level::ByteLevel;
        use tokenizers::models::bpe::{BPE, Vocab};

        let vocab: Vocab = entries
            .iter()
            .map(|(surface, id)| ((*surface).to_owned(), *id))
            .collect();
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, Vec::new())
            .build()
            .expect("valid test vocab");
        let mut tokenizer = InnerTokenizer::new(bpe);
        tokenizer.with_decoder(Some(DecoderWrapper::ByteLevel(ByteLevel::default())));
        tokenizer
    }

    /// `"Das ist schön, oder?"` as byte-alphabet vocab surfaces. `ö` is
    /// UTF-8 `C3 B6`, which surfaces as `Ã¶`; ids 6/7 split those two
    /// bytes across two tokens the way CJK and rare words tokenise.
    fn german_vocab() -> Vec<(&'static str, u32)> {
        vec![
            ("Das", 0),
            ("\u{120}ist", 1),
            ("\u{120}sch\u{c3}\u{b6}n", 2),
            (",", 3),
            ("\u{120}oder", 4),
            ("?", 5),
            ("\u{c3}", 6),
            ("\u{b6}", 7),
            ("\u{120}f", 8),
            ("\u{ef}\u{bf}\u{bd}", 9),
        ]
    }

    fn group(
        tokenizer: &InnerTokenizer,
        token_ids: &[usize],
        eot_frame: Option<usize>,
        unicode_only: bool,
    ) -> Vec<PreWord> {
        let probs = vec![1.0_f32; token_ids.len()];
        let frames: Vec<usize> = (0..token_ids.len()).map(|i| i * 10).collect();
        let span = 0..token_ids.len();
        group_pre_words(
            tokenizer,
            token_ids,
            &probs,
            &frames,
            std::slice::from_ref(&span),
            eot_frame,
            unicode_only,
        )
        .expect("grouping succeeds")
    }

    #[test]
    fn group_pre_words_decodes_byte_level_bpe_to_real_text() {
        // Regression for the mojibake bug: word text must come from the
        // tokenizer's decoder, not raw `id_to_token` surfaces — the
        // surface of `" schön"` is `"ĠschÃ¶n"`, and only `Ġ` used to be
        // mapped back, so every umlaut word shipped as `"schÃ¶n"`.
        let tokenizer = byte_level_tokenizer(&german_vocab());
        let words = group(&tokenizer, &[0, 1, 2, 3, 4, 5], Some(60), false);
        let texts: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(texts, vec!["Das", " ist", " schön", ",", " oder", "?"]);
    }

    #[test]
    fn group_pre_words_joins_codepoints_split_across_tokens() {
        // `ö` split into its two UTF-8 bytes across tokens: the partial
        // decode is U+FFFD ("dangling") and must accumulate until the
        // codepoint completes instead of flushing mojibake.
        let tokenizer = byte_level_tokenizer(&german_vocab());
        let words = group(&tokenizer, &[8, 6, 7], Some(40), false);
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text, " fö");
        assert_eq!(words[0].tokens, vec![8, 6, 7]);
        assert_eq!(words[0].probs.len(), 3);
    }

    #[test]
    fn group_pre_words_keeps_genuine_replacement_chars() {
        // A vocab token decoding to a real U+FFFD must flush immediately
        // (the replacement char matches the full decoding at the same
        // position) — otherwise it would absorb the rest of the segment
        // into one giant word.
        let tokenizer = byte_level_tokenizer(&german_vocab());
        let words = group(&tokenizer, &[9, 1], Some(30), false);
        let texts: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(texts, vec!["\u{FFFD}", " ist"]);
    }

    #[test]
    fn group_pre_words_flushes_trailing_incomplete_bytes() {
        // Generation truncated mid-codepoint: the dangling byte still
        // surfaces (as U+FFFD) so its token keeps a timing slot.
        let tokenizer = byte_level_tokenizer(&german_vocab());
        let words = group(&tokenizer, &[0, 6], Some(30), false);
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text, "Das\u{FFFD}");
    }

    #[test]
    fn group_pre_words_unicode_only_makes_every_run_a_word() {
        // Spaceless-language mode (zh/ja/...): every decoded codepoint
        // run is its own word; space/punct splitting would glue the whole
        // segment together.
        let tokenizer = byte_level_tokenizer(&german_vocab());
        let words = group(&tokenizer, &[0, 0], Some(30), true);
        let texts: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(texts, vec!["Das", "Das"]);
    }

    #[test]
    fn group_pre_words_last_word_ends_at_eot_frame() {
        // The chunk-final word's end is the DTW path's entry into the EOT
        // row, not a fabricated one-frame duration.
        let tokenizer = byte_level_tokenizer(&german_vocab());
        let words = group(&tokenizer, &[0, 1], Some(42), false);
        assert_eq!(words.len(), 2);
        assert_eq!(words[1].start_frame, 10);
        assert_eq!(words[1].end_frame, 42);
    }

    #[test]
    fn group_pre_words_without_eot_falls_back_to_one_frame() {
        let tokenizer = byte_level_tokenizer(&german_vocab());
        let words = group(&tokenizer, &[0, 1], None, false);
        assert_eq!(words[1].start_frame, 10);
        assert_eq!(words[1].end_frame, 11);
    }
}
