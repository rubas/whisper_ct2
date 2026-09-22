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
2. On every push to `main`, the workflow releases when the `v<version>` tag
   for `@version` does not exist yet. It builds every artefact, pushes the
   tag, and uploads the tarballs plus `SHA256SUMS` to a new GitHub release. It
   pushes the tag only after every artefact builds, so a failed build loses
   nothing: the next push to `main` releases the version. When the run fails
   after the tag push, dispatch the workflow with that tag. Do not push the
   tag by hand: the workflow then treats the version as released and builds
   nothing.
3. Once the assets are up, regenerate the checksum file locally:

   ```bash
   mix rustler_precompiled.download WhisperCt2.Native --all --ignore-unavailable --print
   ```

   Commit `checksum-Elixir.WhisperCt2.Native.exs` and push it to `main`.
4. Run `mix hex.publish` from a clean tree. `mix.exs` lists `checksum-*.exs`
   under `:files`, so the checksum travels inside the Hex tarball.

To rebuild and re-release an existing tag without a version bump, dispatch the
workflow manually and pass the tag as the `tag` input. The build checks out
that tag, whatever ref you select under "Use workflow from". The run fails
when the tag does not exist or when its `@version` differs from the tag. The
rebuilt tarballs replace the assets of the release, and their bytes differ, so
regenerate the checksum file afterwards. Do not rebuild a tag that is already
on Hex: its package carries the old checksums.
