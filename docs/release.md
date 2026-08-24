# Release

`.github/workflows/release.yml` builds the precompiled NIF artefacts and
creates the GitHub release. Publishing to Hex stays manual.

## Artefacts

| Artefact id                     | CPU backend | GPU            |
| ------------------------------- | ----------- | -------------- |
| `x86_64-unknown-linux-gnu`      | oneDNN      | `cuda-dynamic` |
| `x86_64-unknown-linux-gnu--mkl` | Intel MKL   | `cuda-dynamic` |
| `aarch64-unknown-linux-gnu`     | oneDNN      | `cuda-dynamic` |
| `aarch64-apple-darwin`          | Accelerate  | none           |

Consumers fetch the artefact matching their target triple through
`rustler_precompiled`. `WHISPER_CT2_VARIANT=mkl` picks the MKL build on
x86_64 Linux, and `WHISPER_CT2_BUILD=1` forces a source build instead.

## Publish a version

1. Bump `@version` in `mix.exs`, add the `CHANGELOG.md` entry, and push to
   `main`.
2. The workflow compares `@version` against the previous commit. On a bump it
   builds every artefact, pushes the `v<version>` tag, and uploads the
   tarballs plus `SHA256SUMS` to a new GitHub release. Pushing a tag by hand
   triggers nothing.
3. Once the assets are up, regenerate the checksum file locally:

   ```bash
   mix rustler_precompiled.download WhisperCt2.Native --all --ignore-unavailable --print
   ```

   Commit `checksum-Elixir.WhisperCt2.Native.exs` and push it to `main`.
4. Run `mix hex.publish` from a clean tree. `mix.exs` lists `checksum-*.exs`
   under `:files`, so the checksum travels inside the Hex tarball.

To rebuild and re-release an existing tag without a version bump, dispatch the
workflow manually. Select that tag under "Use workflow from" and pass the same
tag as the `tag` input. The build checks out the commit behind the dispatch
ref. The `tag` input only names the release, so a dispatch from `main` uploads
binaries built from `main` under the old tag.
