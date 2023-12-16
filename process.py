import os
import torch
import resampy
import argparse
import torchaudio

from deepafx_st.utils import DSPMode
from deepafx_st.system import System

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        help="Path to audio file to process.",
        type=str,
    )
    parser.add_argument(
        "-r",
        "--reference",
        help="Path to reference audio file.",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--ckpt",
        help="Path to pre-trained checkpoint.",
        type=str,
    )

    args = parser.parse_args()

    use_dsp = False
    system = System.load_from_checkpoint(
        args.ckpt, dsp_mode=DSPMode.NONE, batch_size=1
    ).eval()

    # load audio data
    x, x_sr = torchaudio.load(args.input)
    r, r_sr = torchaudio.load(args.reference)
    print(f"{r}, {r_sr}")

    # resample if needed
    if x_sr != 24000:
        x_24000 = torch.tensor(resampy.resample(x.view(-1).numpy(), x_sr, 24000))
        x_24000 = x_24000.view(1, -1)
    else:
        x_24000 = x

    if r_sr != 24000:
        r_24000 = torch.tensor(resampy.resample(r.view(-1).numpy(), r_sr, 24000))
        r_24000 = r_24000.view(1, -1)
    else:
        r_24000 = r

    # peak normalize to -12 dBFS
    x_24000 = x_24000[0:1, : int(24000 * len(x[1]) / x_sr)]
    x_24000 = x_24000[0:1, : ]
    x_24000 /= x_24000.abs().max()
    x_24000 *= 10 ** (-12 / 20.0)
    x_24000 = x_24000.view(1, 1, -1)

    # peak normalize to -12 dBFS
    # print(int(24000 * len(r[1]) / r_sr))
    r_24000 = r_24000[0:1, : ]
    r_24000 /= r_24000.abs().max()
    r_24000 *= 10 ** (-12 / 20.0)
    r_24000 = r_24000.view(1, 1, -1)

    with torch.no_grad():
        y_hat, p, e = system(x_24000, r_24000)

    y_hat = y_hat.view(1, -1)
    y_hat /= y_hat.abs().max()
    x_24000 /= x_24000.abs().max()

    # save to disk
    dirname = os.path.dirname(args.input)
    filename = os.path.basename(args.input).replace(".wav", "")
    reference = os.path.basename(args.reference).replace(".wav", "")
    out_filepath = os.path.join(dirname, f"{filename}_out_ref={reference}.wav")
    print(f"Saved output to {out_filepath}")
    torchaudio.save(out_filepath, y_hat.cpu().view(1, -1), 24000)

    system.shutdown()

