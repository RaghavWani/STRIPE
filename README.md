# _STRIPE_
## _SPOTLIGHT Time-domain RFI Processing Engine_

STRIPE is a high-performance RFI mitigation module for beamformed data from [SPOTLIGHT](https://spotlight.ncra.tifr.res.in/), a commensal radio transient survey backend at the upgraded Giant Metrewave Radio Telescope (uGMRT). It is integrated within the [`Ares`](https://github.com/nsmspotlight/Ares) module of SPOTLIGHT data processing chain. 

STRIPE processes data in blocks and applies three stages of time-domain RFI mitigation:

* Patch Filter: Identifies impulsive broadband interference using zero-variance samples and flags the affected samples.
* Skewness-Kurtosis Filter (SKF): Detects anomalous frequency channels using statistical measures of the time series.
* Baseline Filter: Estimates and subtracts slowly-varying baseline fluctuations using a running-median estimate.

Flagged samples and channels are replaced with Gaussian noise. 

STRIPE is designed for efficient block-based processing of large time-frequency datasets and is adapted from the `filtool` utility in [PulsarX](https://github.com/ypmen/PulsarX).

Within SPOTLIGHT, STRIPE processes a block of 800 samples at **3.4x real-time speed** across 640 simultaneous beams.

#### Example
The figure below illustrates the effect of the STRIPE on representative beamformed data recorded with SPOTLIGHT backend.

![Beamformed data before (left) and after (right) RFI mitigation using STRIPE](/assets/dynspect_comparison_IA_B3.png)

_Frequency-time beamformed data before (left) and after (right) RFI mitigation using STRIPE. The comparison illustrates the removal of impulsive and channelised RFI, as well as the correction of time-varying baselines. The dispersed single pulse is more prominent relative to the background noise after mitigation._

## Installation
```bash
git clone https://github.com/RaghavWani/STRIPE.git
cd STRIPE

# compile as
gcc -O2 -std=c11 -fopenmp -o stripe stripe.c -lm
```

## Requirements
* GCC with C11 and OpenMP support
* Linux environment (tested on x86_64)
* Standard C library and math library; no additional external dependencies!

## Usage
`./stripe <input.raw> <block_size> <IQR_threshold> <width> [num_threads]`

#### Arguments
| Argument | Description |
| -------- | ----------- |
| `input.raw` |	Input file containing 8-bit binary beamformed data |
| `block_size`	| Number of time samples processed per block |
| `IQR_threshold`	| Threshold multiplier used for SKF-based bad channel detection |
| `width`	| Running-median window size, in seconds, for baseline correction |
| `num_threads` | Optional number of OpenMP threads |

SPOTLIGHT uses `IQR_threshold` of 4 and `width` of 0.5 seconds.

Output file is automatically created in the same directory as the input file, with the suffix `_stripe.raw` appended to the input filename.

For detailed instructions on running STRIPE, see the [Standard Operating Procedure](SOP.md).

## Data Assumptions
The following parameters correspond to the SPOTLIGHT beamformed data format used for STRIPE:

* Frequency channels: 4096
* Time resolution: 1.31072 ms
* Bits per sample: 8
* Mean, Std dev of output data: 64, 3 

## Citation
If you use STRIPE in your research, please cite:

R. Wani et al. (2026), _Real-Time RFI Mitigation in SPOTLIGHT: A Two-Stage Approach for Transient Searches_, [arXiv:2608.19102](https://arxiv.org/abs/2608.19102)