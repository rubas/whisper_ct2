//! Categorized errors for the NIF boundary.
//!
//! `anyhow` is convenient for threading errors through the transcribe
//! pipeline, but it has no notion of "kind" — every error would otherwise
//! surface to Elixir as `:inference_error`, hiding the difference between
//! bad user input (`invalid_request`), internal NIF state issues
//! (`runtime_error`), and genuine model failures (`inference_error`).
//!
//! Use [`invalid_request`] or [`runtime_error`] when constructing an
//! `anyhow::Error` whose category is known at call-site. The NIF
//! boundary downcasts the resulting `anyhow::Error` chain to recover the
//! category; unrecognised errors fall back to `inference_error`.

use std::fmt;

/// Error wrapper that pins a category at construction. The NIF boundary
/// (`impl From<anyhow::Error> for NativeError` in `lib.rs`) walks the
/// `anyhow` chain and uses this category when present.
#[derive(Debug)]
pub(crate) struct Categorized {
    /// One of the `WhisperCt2.Error` reason strings: `"invalid_request"`,
    /// `"runtime_error"`, `"inference_error"`, `"load_error"`.
    pub(crate) kind: &'static str,
    pub(crate) message: String,
}

impl fmt::Display for Categorized {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Categorized {}

/// User input was rejected at a NIF-internal boundary (bad language code,
/// no mel chunks produced, mixed-language batch with `word_timestamps`,
/// over-large input buffer). Surfaces as `:invalid_request` in Elixir.
pub(crate) fn invalid_request(message: impl Into<String>) -> anyhow::Error {
    anyhow::Error::new(Categorized {
        kind: "invalid_request",
        message: message.into(),
    })
}

/// Internal NIF runtime fault (NaN propagation, unexpected ct2rs state).
/// Surfaces as `:runtime_error` in Elixir — distinct from
/// `:inference_error`, which is reserved for CTranslate2-side failures.
pub(crate) fn runtime_error(message: impl Into<String>) -> anyhow::Error {
    anyhow::Error::new(Categorized {
        kind: "runtime_error",
        message: message.into(),
    })
}

/// Walks the `anyhow` chain for the first attached [`Categorized`] kind.
pub(crate) fn kind_from_chain(err: &anyhow::Error) -> Option<&'static str> {
    err.chain()
        .find_map(|cause| cause.downcast_ref::<Categorized>())
        .map(|c| c.kind)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_from_chain_returns_categorized_kind() {
        let err = invalid_request("bad lang");
        assert_eq!(kind_from_chain(&err), Some("invalid_request"));

        let err = runtime_error("oops");
        assert_eq!(kind_from_chain(&err), Some("runtime_error"));
    }

    #[test]
    fn kind_from_chain_walks_context() {
        let err = invalid_request("bad lang").context("while building chunks");
        assert_eq!(kind_from_chain(&err), Some("invalid_request"));
    }

    #[test]
    fn kind_from_chain_returns_none_for_uncategorised() {
        let err = anyhow::anyhow!("plain anyhow error");
        assert_eq!(kind_from_chain(&err), None);
    }
}
