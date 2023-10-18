#!/usr/bin/env python
import argparse
from Region_segmentation.totalsegmentator.config import setup_totalseg


def main():
    """
    Manually setup totalsegmentator config file

    Usage:
    totalseg_setup_manually -id totalseg_12345678
    """
    parser = argparse.ArgumentParser(description="Combine masks.",
                                     epilog="Written by Jakob Wasserthal. If you use this tool please cite https://pubs.rsna.org/doi/10.1148/ryai.230024")

    parser.add_argument("-id", "--totalseg_id", type=str, help="totalseg_id. Must start with totalseg_.",
                        required=True)

    args = parser.parse_args()

    if not args.totalseg_id.startswith("totalseg_"):
        raise ValueError("totalseg_id must start with totalseg_")
    if len(args.totalseg_id) != 17:
        raise ValueError("totalseg_id must have exactly 17 characters.")

    setup_totalseg(args.totalseg_id)


if __name__ == "__main__":
    main()
