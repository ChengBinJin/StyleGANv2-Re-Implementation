# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html
# Cheng-Bin Jin re-implementation.

import argparse

_examples = '''examples:

  # Generate ffhq uncurated images (matches paper Figure 12)
  python %(prog)s generate-images --network=gdrive:networks/stylegans-ffhq-config-f.pkl --seeds=6600-6625 --truncation-psi=0.5
  
  # Generate ffhq curated images (matches paper Figure 11)
  python (%prog)s generate-images --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --seeds=66,230,389,1518 --truncation-psi=1.0
  
  # Generate uncurated car images (matches paper Figure 12)
  python %(prog)s generate-images --network=gdrive:networks/stylegan2-car-config-f.pk --seeds=6000-6025 --truncation-psi=0.5
  
  # Generate style mixing example (matches style mixing video clip)
  python %(prog)s style-mixing-example --network=gdrive:networks/stylegans-ffhq-config-f.pkl --row--seeds=85,100,75,458,15000 --col-seeds=55,821,1789,293 --truncation-psi=1.0
'''

#-----------------------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='''StyleGAN2 generator. Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(help='Sub-commans', dest='command')

#-----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#-----------------------------------------------------------------------------------------------------------------------