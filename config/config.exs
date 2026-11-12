import Config

# This repo builds the NIF from source, unless WHISPER_CT2_BUILD=0: the Hex
# publish job of `release.yml` then downloads the precompiled NIF and checks
# it against the checksum file. Consumers of the Hex package never load this
# file and get the precompiled NIF for their target triple.
config :rustler_precompiled, :force_build, whisper_ct2: System.get_env("WHISPER_CT2_BUILD") != "0"
