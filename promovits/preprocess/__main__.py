import argparse

import promovits


###############################################################################
# Entry point
###############################################################################


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_extension", default="cleaned")
    parser.add_argument("--text_index", default=1, type=int)
    parser.add_argument("--filelists", nargs="+", default=["filelists/ljs_audio_text_val_filelist.txt", "filelists/ljs_audio_text_test_filelist.txt"])
    return parser.parse_args()


if __name__ == '__main__':
  promovits.preprocess.dataset(**vars(parse_args()))
