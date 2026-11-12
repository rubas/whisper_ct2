# Release

`.github/workflows/release.yml` builds the precompiled NIF artefacts,
creates the GitHub release, and publishes the package to Hex when the
checksum PR merges.

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
   mix rustler_precompiled.download WhisperCt2.Native --all --print
   ```

   The task fails when an asset of a target is missing. Commit
   `checksum-Elixir.WhisperCt2.Native.exs`, and open a pull request.
4. The merge of the checksum PR publishes. A push to `main` that changes the
   checksum file, for a version that has its tag, starts the
   `Publish to Hex` job. It checks out the tag, which the NIF archives were
   built from, and takes the checksum file of the merge. It compares that
   file with the release's `SHA256SUMS` and stops when they differ. Then it
   runs `mix hex.publish --yes` with `WHISPER_CT2_BUILD=0`, so the docs build
   downloads the precompiled NIF and checks it against the checksum file.
   `mix.exs` lists `checksum-*.exs` under `:files`, so the checksum travels
   inside the Hex tarball.

To rebuild and re-release an existing tag without a version bump, dispatch the
workflow manually and pass the tag as the `tag` input. The build checks out
that tag, whatever ref you select under "Use workflow from". The run fails
when the tag does not exist or when its `@version` differs from the tag. The
rebuilt tarballs replace the assets of the release, and their bytes differ, so
regenerate the checksum file afterwards. Do not rebuild a tag that is already
on Hex: its package carries the old checksums.

## Hex

The job stores no Hex key. It runs in the GitHub environment `hex`, which
only `main` can use, and Hex exchanges the job's OIDC token for a token that
can only publish `whisper_ct2`. For that, the package has a workload
identity on hex.pm, in its tab "Workload identities": repository owner
`rubas`, repository `whisper_ct2`, workflow `release.yml`, environment
`hex`. Only a full owner of the package can add or change it.

The package has staged publishing on. Hex holds each release from the job,
and the job prints `Package staged for approval at <url>`. A package
owner approves it on hex.pm with their 2FA, within 7 days. The job does not
wait for that approval: a GitHub job stops after 6 hours.

When the job fails, fix the cause and re-run the run of the checksum merge.
It publishes the same tag with the same checksum file. A re-run keeps the
workflow file of that run, so when the job itself needs a fix, a package
owner runs `WHISPER_CT2_BUILD=0 mix hex.publish` from the tag with the
checksum file of the merge. The job fails when the log says
`Publishing failed`, because `mix hex.publish` exits 0 when only the docs
upload fails.

The tag `vX.Y.Z` points at the version bump, so its checksum file is still
the one of the release before. A git dependency on the tag needs
`WHISPER_CT2_BUILD=1`.
