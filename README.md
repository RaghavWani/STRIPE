# _STRIPE_
### _SPOTLIGHT Time-domain RFI Processing Engine_

STRIPE is a high-performance module for mitigating "_stripe-like_" Radio Frequency Interference (RFI) in SPOTLIGHT beam-formed data.

It implements:

* Skewness-Kurtosis Filter (SKF)
* Time-domain patch flagging
* Band equalization
* Baseline correction

Gaussian noise replacement of flagged samples \
Designed for efficient block-based processing of large time-frequency datasets. \
Adapted from PulsarX's [filtool](https://github.com/ypmen/PulsarX/blob/main/src/pulsar/filtool.cpp)

## Installation
```bash
git clone https://github.com/RaghavWani/STRIPE.git
cd STRIPE

# compile using script
./compile.sh

# or manually 
g++ -O3 -march=native -std=c++17 ./rfi_mitigation.cpp -o stripe
```

## Requirements
* GCC ≥ 7 (with C++17 support)
* Linux environment (tested on x86_64)
* Standard C++ libraries only (no external dependencies!)

## Usage
`./stripe <input.raw> <block_size> <IQR_threshold>`

#### Arguments
| Argument | Description |
| -------- | ----------- |
| `input.raw` |	8-bit time-freq binary data file |
| `block_size`	| Number of time samples processed per block |
| `IQR_threshold`	| Threshold multiplier for SKF-based bad channel detection |

Output file is automatically created as `<input_filename>_stripe.raw` in the same directory as the input file.

## Data Assumptions (SPOTLIGHT Beam Mode)
* Frequency channels: 4096
* Time resolution: 1.31072 ms
* Bits per sample: 8
* Mean, Std dev of output data: 64, 3
* Baseline width: 0 

