#![allow(clippy::cast_precision_loss)]

use anyhow::{Result, anyhow};
use ct2rs::tokenizers::hf;
use tokenizers::Tokenizer as InnerTokenizer;

/// IDs of every Whisper special token we need to interpret the generated
/// sequence. Resolved once at model load by querying the HF tokenizer.
pub(crate) struct SpecialTokens {
    /// `<|startoftranscript|>`
    pub(crate) sot: u32,
    /// `<|transcribe|>`
    pub(crate) transcribe: u32,
    /// `<|notimestamps|>`
    pub(crate) no_timestamps: u32,
    /// `<|0.00|>` — base of the 1501 timestamp-token range. Any token ID
    /// `>= timestamp_begin` is a timestamp; its value is
    /// `(id - timestamp_begin) * 0.02` seconds.
    pub(crate) timestamp_begin: u32,
}

impl SpecialTokens {
    pub(crate) fn resolve(inner: &InnerTokenizer) -> Result<Self> {
        // Whisper's timestamp tokens (`<|0.00|>` ..= `<|30.00|>`) are NOT
        // in the tokenizer vocab; they live in the model output space
        // immediately after `<|notimestamps|>`, matching faster-whisper's
        // convention: `timestamp_begin = no_timestamps_id + 1`. We
        // additionally probe `<|startofprev|>` because `initial_prompt`
        // expects it to exist even though we never store the ID here.
        let no_timestamps = lookup(inner, NO_TIMESTAMPS)?;
        Ok(Self {
            sot: lookup(inner, SOT)?,
            transcribe: lookup(inner, TRANSCRIBE)?,
            no_timestamps,
            timestamp_begin: no_timestamps + 1,
        })
    }
}

// Whisper special-token literals used in prompt construction. Centralised
// so the prompt builder and `SpecialTokens::resolve` cannot drift.
pub(crate) const SOT: &str = "<|startoftranscript|>";
pub(crate) const STARTOFPREV: &str = "<|startofprev|>";
pub(crate) const TRANSCRIBE: &str = "<|transcribe|>";
pub(crate) const NO_TIMESTAMPS: &str = "<|notimestamps|>";

fn lookup(inner: &InnerTokenizer, token: &str) -> Result<u32> {
    inner
        .token_to_id(token)
        .ok_or_else(|| anyhow!("special token {token} missing from tokenizer vocab"))
}

pub(crate) fn language_token(inner: &InnerTokenizer, code: &str) -> Result<String> {
    let token = format!("<|{code}|>");
    if inner.token_to_id(&token).is_none() {
        return Err(anyhow!("language token {token} not in vocab"));
    }
    Ok(token)
}

pub(crate) fn token_id(inner: &InnerTokenizer, token: &str) -> Result<u32> {
    inner
        .token_to_id(token)
        .ok_or_else(|| anyhow!("token {token} missing from tokenizer vocab"))
}

pub(crate) fn encode_plain(tokenizer: &hf::Tokenizer, text: &str) -> Result<Vec<String>> {
    let inner: &InnerTokenizer = tokenizer;
    let encoding = inner
        .encode(text, false)
        .map_err(|e| anyhow!("failed to tokenize prompt text: {e}"))?;
    Ok(encoding.get_tokens().to_vec())
}

/// Builds the per-chunk prompt vector that `sys::Whisper::generate` consumes.
///
/// Layout:
///
/// ```text
/// [<|startofprev|> <initial_prompt_tokens>]? <|startoftranscript|> <|lang|>
///   <|transcribe|> [<|notimestamps|>]? <prefix_tokens>*
/// ```
///
/// `with_timestamps` controls whether `<|notimestamps|>` is appended; when
/// `false`, the model emits `<|t_..|>` tokens we parse back out.
pub(crate) struct PromptParts<'a> {
    pub(crate) sot: &'a str,
    pub(crate) startofprev: &'a str,
    pub(crate) language_token: &'a str,
    pub(crate) transcribe: &'a str,
    pub(crate) no_timestamps: &'a str,
    pub(crate) initial_prompt: &'a [String],
    pub(crate) prefix: &'a [String],
    pub(crate) with_timestamps: bool,
    /// `false` for English-only checkpoints (`*.en`): the SOT block is
    /// just `<|startoftranscript|>`. Multilingual checkpoints append the
    /// language and `<|transcribe|>` tokens, matching faster-whisper.
    pub(crate) multilingual: bool,
}

impl PromptParts<'_> {
    pub(crate) fn build(&self) -> Vec<String> {
        let mut out: Vec<String> =
            Vec::with_capacity(self.initial_prompt.len() + self.prefix.len() + 5);
        if !self.initial_prompt.is_empty() {
            out.push(self.startofprev.to_owned());
            out.extend(self.initial_prompt.iter().cloned());
        }
        out.push(self.sot.to_owned());
        if self.multilingual {
            out.push(self.language_token.to_owned());
            out.push(self.transcribe.to_owned());
        }
        if !self.with_timestamps {
            out.push(self.no_timestamps.to_owned());
        }
        out.extend(self.prefix.iter().cloned());
        out
    }
}

/// One `<|start_ts|> text... <|end_ts|>` sub-segment carved out of a chunk's
/// generated token IDs. Offsets are relative to the chunk; the caller adds
/// the chunk's start time to produce absolute audio time.
#[derive(Debug)]
pub(crate) struct SubSegment {
    pub(crate) text_token_ids: Vec<u32>,
    pub(crate) start_in_chunk: f32,
    pub(crate) end_in_chunk: f32,
}

/// Parses a generated chunk's token IDs into `<|t_start|> text <|t_end|>`
/// sub-segments. Token IDs `>= timestamp_begin` are treated as timestamps;
/// anything before the first timestamp pair is discarded as preamble.
///
/// `chunk_duration_s` is the wall-clock length of the Whisper window
/// (30 s for every published checkpoint). It is used as the fallback
/// `end_in_chunk` in two situations:
///
/// 1. **Unclosed pair**: the model emitted `<|t_start|> text [EOT]` with
///    no closing timestamp. Some fine-tunes (notably notebotIE Swiss-German)
///    only reliably emit the opening timestamp.
/// 2. **No timestamps at all**: the prompt asked for `<|notimestamps|>`,
///    or the fine-tune ignored the timestamp instruction and emitted
///    plain text. The whole token stream becomes one sub-segment
///    spanning `[0, chunk_duration_s)`.
///
/// Faster-whisper handles both cases the same way; dropping the text
/// silently is how multi-second turns turned into empty transcripts.
pub(crate) fn split_sub_segments(
    token_ids: &[u32],
    timestamp_begin: u32,
    chunk_duration_s: f32,
) -> Vec<SubSegment> {
    let mut out = Vec::new();
    let mut i = 0;
    let mut saw_first_timestamp = false;

    while i < token_ids.len() {
        let preamble_start = i;

        while i < token_ids.len() && token_ids[i] < timestamp_begin {
            i += 1;
        }

        if i >= token_ids.len() {
            // No timestamps in this entire chunk. Flush every token as
            // one sub-segment covering the whole chunk window — without
            // this, `<|notimestamps|>` mode (or any fine-tune that just
            // refuses to emit timestamps) would lose all of its output.
            if !saw_first_timestamp && preamble_start < token_ids.len() {
                out.push(SubSegment {
                    text_token_ids: token_ids[preamble_start..].to_vec(),
                    start_in_chunk: 0.0,
                    end_in_chunk: chunk_duration_s,
                });
            }
            break;
        }

        saw_first_timestamp = true;
        let start_id = token_ids[i];
        let text_start = i + 1;
        i += 1;

        while i < token_ids.len() && token_ids[i] < timestamp_begin {
            i += 1;
        }

        if i >= token_ids.len() {
            // Unclosed pair: model emitted `<|t_start|> text [EOT]` with
            // no closing timestamp. Flush the pending text with the
            // chunk window's end as the fallback boundary instead of
            // silently dropping it.
            if text_start < token_ids.len() {
                out.push(SubSegment {
                    text_token_ids: token_ids[text_start..].to_vec(),
                    start_in_chunk: timestamp_seconds(start_id, timestamp_begin),
                    end_in_chunk: chunk_duration_s,
                });
            }
            break;
        }

        let end_id = token_ids[i];
        let text_end = i;
        i += 1;

        if text_start >= text_end {
            continue;
        }

        out.push(SubSegment {
            text_token_ids: token_ids[text_start..text_end].to_vec(),
            start_in_chunk: timestamp_seconds(start_id, timestamp_begin),
            end_in_chunk: timestamp_seconds(end_id, timestamp_begin),
        });
    }
    out
}

#[inline]
fn timestamp_seconds(token_id: u32, timestamp_begin: u32) -> f32 {
    (token_id - timestamp_begin) as f32 * 0.02
}

/// Decodes a flat list of text-only token IDs to a single string.
pub(crate) fn decode_ids(tokenizer: &hf::Tokenizer, ids: &[u32]) -> Result<String> {
    let inner: &InnerTokenizer = tokenizer;
    inner
        .decode(ids, true)
        .map_err(|e| anyhow!("failed to decode tokens: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const BEGIN: u32 = 50_000;
    const CHUNK_S: f32 = 30.0;

    fn ts(offset: u32) -> u32 {
        BEGIN + offset
    }

    #[test]
    fn split_sub_segments_returns_empty_for_no_tokens() {
        assert!(split_sub_segments(&[], BEGIN, CHUNK_S).is_empty());
    }

    #[test]
    fn split_sub_segments_discards_preamble_before_first_timestamp() {
        let out = split_sub_segments(&[10, 20, ts(0), 100, 101, ts(100)], BEGIN, CHUNK_S);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text_token_ids, vec![100, 101]);
        assert!((out[0].start_in_chunk - 0.0).abs() < 1e-6);
        assert!((out[0].end_in_chunk - 2.0).abs() < 1e-6);
    }

    #[test]
    fn split_sub_segments_handles_back_to_back_pairs() {
        let out = split_sub_segments(&[ts(0), 100, ts(50), ts(50), 200, ts(150)], BEGIN, CHUNK_S);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].text_token_ids, vec![100]);
        assert!((out[0].end_in_chunk - 1.0).abs() < 1e-6);
        assert_eq!(out[1].text_token_ids, vec![200]);
        assert!((out[1].start_in_chunk - 1.0).abs() < 1e-6);
        assert!((out[1].end_in_chunk - 3.0).abs() < 1e-6);
    }

    #[test]
    fn split_sub_segments_skips_pairs_with_empty_text() {
        let out = split_sub_segments(&[ts(0), ts(50)], BEGIN, CHUNK_S);
        assert!(out.is_empty());
    }

    #[test]
    fn split_sub_segments_flushes_all_text_when_no_timestamps_emitted() {
        // `<|notimestamps|>` mode or fine-tunes that just refuse to emit
        // timestamps: the whole token stream becomes one segment spanning
        // [0, chunk_duration_s).
        let out = split_sub_segments(&[100, 101, 102, 103], BEGIN, CHUNK_S);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text_token_ids, vec![100, 101, 102, 103]);
        assert!((out[0].start_in_chunk - 0.0).abs() < 1e-6);
        assert!((out[0].end_in_chunk - CHUNK_S).abs() < 1e-6);
    }

    #[test]
    fn split_sub_segments_flushes_text_after_unclosed_start_timestamp() {
        // Some fine-tunes emit `<|t_start|> text [EOT]` without a closing
        // timestamp. The text must be flushed with `chunk_duration_s` as
        // the fallback end, not dropped.
        let out = split_sub_segments(&[ts(0), 100, 101, 102], BEGIN, CHUNK_S);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text_token_ids, vec![100, 101, 102]);
        assert!((out[0].start_in_chunk - 0.0).abs() < 1e-6);
        assert!((out[0].end_in_chunk - CHUNK_S).abs() < 1e-6);
    }

    #[test]
    fn split_sub_segments_flushes_trailing_text_after_closed_pair() {
        // Mixed case: one balanced pair followed by an unclosed
        // `<|t_start|> text` tail. Both must appear in the output.
        let out = split_sub_segments(&[ts(0), 100, ts(50), ts(60), 200, 201], BEGIN, CHUNK_S);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].text_token_ids, vec![100]);
        assert_eq!(out[1].text_token_ids, vec![200, 201]);
        assert!((out[1].start_in_chunk - 1.2).abs() < 1e-6);
        assert!((out[1].end_in_chunk - CHUNK_S).abs() < 1e-6);
    }

    #[test]
    fn split_sub_segments_drops_lone_dangling_start_timestamp() {
        // `<|t_start|>` immediately followed by EOT (no text) still
        // produces nothing — there is nothing to flush.
        let out = split_sub_segments(&[ts(0)], BEGIN, CHUNK_S);
        assert!(out.is_empty());
    }

    #[test]
    fn timestamp_seconds_uses_two_centisecond_step() {
        assert!((timestamp_seconds(BEGIN, BEGIN) - 0.0).abs() < 1e-6);
        assert!((timestamp_seconds(BEGIN + 1, BEGIN) - 0.02).abs() < 1e-6);
        assert!((timestamp_seconds(BEGIN + 1500, BEGIN) - 30.0).abs() < 1e-4);
    }

    // PromptParts uses the literal token strings from this module, not
    // numeric ids, so we can build prompts without a loaded tokenizer.
    fn parts<'a>(
        initial_prompt: &'a [String],
        prefix: &'a [String],
        with_timestamps: bool,
        multilingual: bool,
        lang: &'a str,
    ) -> PromptParts<'a> {
        PromptParts {
            sot: SOT,
            startofprev: STARTOFPREV,
            language_token: lang,
            transcribe: TRANSCRIBE,
            no_timestamps: NO_TIMESTAMPS,
            initial_prompt,
            prefix,
            with_timestamps,
            multilingual,
        }
    }

    #[test]
    fn prompt_english_only_no_timestamps() {
        // .en checkpoints: SOT block is just `<|startoftranscript|>`,
        // then `<|notimestamps|>`. No lang or `<|transcribe|>`.
        let p = parts(&[], &[], false, false, "<|en|>");
        assert_eq!(p.build(), vec![SOT.to_owned(), NO_TIMESTAMPS.to_owned()]);
    }

    #[test]
    fn prompt_multilingual_with_timestamps() {
        // Multilingual + with_timestamps: SOT, lang, transcribe, no
        // `<|notimestamps|>` because the model must emit timestamps.
        let p = parts(&[], &[], true, true, "<|de|>");
        assert_eq!(
            p.build(),
            vec![SOT.to_owned(), "<|de|>".to_owned(), TRANSCRIBE.to_owned()]
        );
    }

    #[test]
    fn prompt_with_initial_prompt_prepends_startofprev() {
        let initial = vec!["hello".to_owned(), "world".to_owned()];
        let p = parts(&initial, &[], false, true, "<|en|>");
        assert_eq!(
            p.build(),
            vec![
                STARTOFPREV.to_owned(),
                "hello".to_owned(),
                "world".to_owned(),
                SOT.to_owned(),
                "<|en|>".to_owned(),
                TRANSCRIBE.to_owned(),
                NO_TIMESTAMPS.to_owned(),
            ]
        );
    }

    #[test]
    fn prompt_with_prefix_appended_after_sot_block() {
        let prefix = vec!["The".to_owned(), "topic".to_owned()];
        let p = parts(&[], &prefix, false, true, "<|en|>");
        assert_eq!(
            p.build(),
            vec![
                SOT.to_owned(),
                "<|en|>".to_owned(),
                TRANSCRIBE.to_owned(),
                NO_TIMESTAMPS.to_owned(),
                "The".to_owned(),
                "topic".to_owned(),
            ]
        );
    }
}
