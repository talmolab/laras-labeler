# HiDRA patches

## reviewed-frames-only.patch

Adds `finetune.py prepare --reviewed <csv>` to HiDRA. It is what the labeler's HiDRA "train on"
menu needs for two of its three choices:

| train on | what HiDRA learns from | needs this patch |
|---|---|---|
| every frame | every frame of each staged clip; anything not marked yes counts as no | no |
| inside social window | the same, but frames outside each clip's social window are left out | yes |
| reviewed frames only | only the frames you answered yes or no; every other frame counts as neither | yes |

Without the patch, choosing one of the last two stops the fine-tune with a message saying so.

Written by Andy Gibson (Kennedy lab) against HiDRA 0.1.0 (a93b3d0) and ported to HiDRA v0.2.0. It
applies cleanly to f0c1a85 and 2bfb926. The one conflict with v0.2.0 was in `finetune.py prepare`,
where v0.2.0 added per-file scored lists for `.slp` annotations; the port keeps both.

Apply it to the HiDRA checkout the labeler uses (`HIDRA_HOME`):

    git -C "$HIDRA_HOME" apply /path/to/laras-labeler/scripts/hidra_patches/reviewed-frames-only.patch

Check it took: `finetune.py prepare --help` lists `--reviewed`. The patch also adds
`tests/test_reviewed_frames.py` to HiDRA.

This belongs in HiDRA itself; once HiDRA ships `prepare --reviewed`, this file can go.
