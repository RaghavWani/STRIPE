#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include <random>
#include <limits>
#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <iterator>
#include <filesystem>
#include <chrono>
// Last edited 18th Feb 2026; Raghav Wani

namespace fs = std::filesystem;

// Raw binary data file reader
std::vector<uint8_t> read_binary_data(const std::string& filename, const size_t num_freq) {  
    std::ifstream file(filename, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open file: " + filename);

    file.seekg(0, std::ios::end);
    std::streampos file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (file_size % num_freq != 0)
        throw std::runtime_error("File size not multiple of num_freq");

    size_t num_time = static_cast<size_t>(file_size) / num_freq;

    std::vector<uint8_t> data(num_time * num_freq);
    file.read(reinterpret_cast<char*>(data.data()),  data.size() * sizeof(uint8_t));

    if (!file)
        throw std::runtime_error("Error reading file");
    file.close();
    return data;
}

// Datatype conversion function1 
void uint8_to_float(const std::vector<uint8_t>& in, std::vector<float>& out, size_t N, float zero_off = 64.0f) {
    out.resize(N);
    for (size_t k = 0; k < N; ++k) 
        out[k] = static_cast<float>(in[k]) - zero_off;
}

// Datatype conversion function2
void float_to_uint8(std::vector<float>& in, std::vector<uint8_t>& out, size_t N, float outmean, float outstd) {
    double tmpmean = 0.;
    double tmpstd = 0.;

    for (size_t k = 0; k < N; ++k) {
        double v = in[k];
        tmpmean += v;
        tmpstd  += v*v;
    }

    auto tmp_1 = N;
    tmpmean /= tmp_1;
    tmpstd /= tmp_1;
    tmpstd -= tmpmean * tmpmean;
    tmpstd = std::sqrt(tmpstd);

    float scl = outstd/tmpstd;
	float offs = outmean-scl*tmpmean;

    for (size_t k = 0; k < N; ++k) {
        float tmp = scl * in[k] + offs;
        tmp = std::round(tmp);
        tmp = std::clamp(tmp, 0.0f, 255.0f);
        out[k] = static_cast<uint8_t>(tmp);
    }
}

// Raw binary data file writer
void write_binary_data(const std::string& filename, const std::vector<uint8_t>& data, size_t nsamples, size_t nchans) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    size_t num_time = data.size();
    if ( data.size() % nsamples == 0 || data.size() % nchans != 0 ) {
        throw std::runtime_error("Data dimensions do not match expected size (nsamples x 4096): " + filename);
    }

    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(uint8_t));
    file.close();
}

// SKF based RFI mitigation function
void skf_filter(std::vector<float>& data, float thresig, size_t nsamples, size_t nchans, bool fill_rand = true) {
    if (data.empty()) {
        throw std::runtime_error("Error: Empty data provided. ");
    }
    
    // Step 1: Compute per-channel statistics (mean1, mean2, mean3, mean4 for moments, and correlation)
    std::vector<double> chmean1(nchans, 0.), chmean2(nchans, 0.), chmean3(nchans, 0.), 
    chmean4(nchans, 0.), chcorr(nchans, 0.), last_data(nchans, 0.);
    
    for (int j = 0; j < nchans; ++j) {
        double m1=0,m2=0,m3=0,m4=0,corr=0;
        double last = data[j];
    
        for (int i=1; i<nsamples; ++i) {
            double tmp = data[i*nchans + j];
            double tmp2 = tmp*tmp;

            m1 += tmp;
            m2 += tmp2;
            m3 += tmp2*tmp;
            m4 += tmp2*tmp2;
            corr += tmp * last;
            last = tmp;
        }

        chmean1[j] = m1;
        chmean2[j] = m2;
        chmean3[j] = m3;
        chmean4[j] = m4;
        chcorr[j]  = corr;
    }

    // Step 2: Compute derived statistics (mean, std, skewness, kurtosis, correlation) per channel
    std::vector<float> chmean(nchans, 0.), chstd(nchans, 0.), chskewness(nchans, 0.), chkurtosis(nchans, 0.);
    
    for (int j = 0; j < nchans; ++j) {
        chmean1[j] /= nsamples;
        chmean2[j] /= nsamples;
        chmean3[j] /= nsamples;
        chmean4[j] /= nsamples;
        chcorr[j] /= (nsamples - 1);
        
        double tmp = chmean1[j] * chmean1[j];
        chmean[j] = chmean1[j];
        chstd[j] = chmean2[j] - tmp;
        
        if (chstd[j] > 0.) {
           
            chskewness[j] = chmean3[j] - 3. * chmean2[j] * chmean1[j] + 2. * tmp * chmean1[j];
            chkurtosis[j] = chmean4[j] - 4. * chmean3[j] * chmean1[j] + 6. * chmean2[j] * tmp - 3. * tmp * tmp;
            chkurtosis[j] /= chstd[j] * chstd[j];
            chkurtosis[j] -= 3.;  // pearson kurtosis
            chskewness[j] /= chstd[j] * std::sqrt(chstd[j]);
            chcorr[j] -= tmp;
            chcorr[j] /= chstd[j];

        } else {
            chstd[j] = 1.;
            chkurtosis[j] = std::numeric_limits<float>::max();
            chskewness[j] = std::numeric_limits<float>::max();
            chcorr[j] = std::numeric_limits<float>::max();
        }
        chstd[j] = std::sqrt(chstd[j]);
    }
    
    // Step 3: Compute IQR for skewness, kurtosis, and correlation
    // kurtosis
    std::vector<float> kurtosis_sort = chkurtosis;
    std::nth_element(kurtosis_sort.begin(), kurtosis_sort.begin()+kurtosis_sort.size()/4, kurtosis_sort.end(), std::less<float>());
	float kurtosis_q1 = kurtosis_sort[kurtosis_sort.size()/4];
	std::nth_element(kurtosis_sort.begin(), kurtosis_sort.begin()+kurtosis_sort.size()/4, kurtosis_sort.end(), std::greater<float>());
	float kurtosis_q3 =kurtosis_sort[kurtosis_sort.size()/4];
	float kurtosis_R = kurtosis_q3-kurtosis_q1;

    // skewness
    std::vector<float> skewness_sort = chskewness;
    std::nth_element(skewness_sort.begin(), skewness_sort.begin()+skewness_sort.size()/4, skewness_sort.end(), std::less<float>());
	float skewness_q1 = skewness_sort[skewness_sort.size()/4];
	std::nth_element(skewness_sort.begin(), skewness_sort.begin()+skewness_sort.size()/4, skewness_sort.end(), std::greater<float>());
	float skewness_q3 =skewness_sort[skewness_sort.size()/4];
	float skewness_R = skewness_q3-skewness_q1;

    // Auto Correlation with Lag-1
    std::vector<double> corr_sort(chcorr.begin(), chcorr.end());
    std::nth_element(corr_sort.begin(), corr_sort.begin()+corr_sort.size()/4, corr_sort.end(), std::less<float>());
	double corr_q1 = corr_sort[corr_sort.size()/4];
	std::nth_element(corr_sort.begin(), corr_sort.begin()+corr_sort.size()/4, corr_sort.end(), std::greater<float>());
	double corr_q3 = corr_sort[corr_sort.size()/4];
	double corr_R = corr_q3-corr_q1;

    // Step 4: Flag channels based on IQR thresholds
    std::vector<unsigned char> tmpmask(nchans, 0);
    std::vector<uint8_t> weights(nchans, 0);
    std::fill(weights.begin(), weights.end(), 0.); // Initialize all weights to 0 (flagged)
    long int kill_count = 0;

    if (thresig >= 0) {
        for (int j = 0; j < nchans; ++j) {
            if (chkurtosis[j]>=kurtosis_q1-thresig*kurtosis_R && \
				chkurtosis[j]<=kurtosis_q3+thresig*kurtosis_R && \
				chskewness[j]>=skewness_q1-thresig*skewness_R && \
				chskewness[j]<=skewness_q3+thresig*skewness_R && \
				chcorr[j]>=corr_q1-thresig*corr_R && \
				chcorr[j]<=corr_q3+thresig*corr_R) {
                weights[j] = 1.;
            } else {
                tmpmask[j] = 1;
                ++kill_count;
            }
        }
    } else { for (int j = 0; j < nchans; ++j) {
            weights[j] = 1.;
        }
    }
    float killrate = kill_count * 1. / nchans;
    std::cout << "SKF: Kill rate = " << killrate * 100 << "%" << std::endl;

    // Step5: Normalization
    for (int i = 0; i < nsamples; ++i) {
        float* row = &data[i*nchans];
        for (int j = 0; j < nchans; ++j) {
            row[j] = weights[j] * (row[j] - chmean[j]) / chstd[j];
        }
    }
    
    // Step 6: Bad channel Replacement
    std::random_device r;
    std::mt19937 generator(r());
    std::normal_distribution<float> distribution(0., 1.);

    for (long int i=0; i<nsamples; ++i)
		{
			for (long int j=0; j<nchans; ++j)
			{
				if (weights[j] == 0.)
					data[i*nchans + j] = distribution(generator);
			}
		}
} 

// Patch filter
void patch_filter(std::vector<float>& data, size_t nsamples, size_t nchans, std::string filltype){
    std::vector<uint8_t> mask(nsamples, 0); 

    // Step 1: detect zero-variance rows
    auto step1_start = std::chrono::high_resolution_clock::now();
    long int kill_count = 0;
    double killrate = 0.;

    for (int i = 0; i < nsamples; ++i)
    {
        double sum = 0;
        double sq_sum = 0;
        
        for (int j = 0; j < nchans; ++j){
            sum += data[i*nchans + j];
            sq_sum += data[i*nchans + j] * data[i*nchans + j];}
        double mean = sum / nchans;
        double var = sq_sum / nchans - mean * mean;

        if (var == 0.0){
            mask[i] = 1; // mask[i] = true;
            if (i!=0) mask[i-1] = 1;
            if (i!= nsamples -1) mask[i+1] = 1;
            ++kill_count;
        }
    }
    killrate = kill_count * 1. / nsamples;
    std::cout << "Patch Filter: Kill rate = " << killrate << std::endl;
    
    // Step 2: Mean-Variance Calculation of non-flagged time sample
    std::vector<double> chvar_patch(nchans, 0.);
	std::vector<double> chmean_patch(nchans, 0.);
    long int count = 0;
    for (int i = 0; i < nsamples; ++i) //swap row for chans
    {
        if (mask[i])
            std::cout << "This time samples is flagged:" << i << std::endl;
            continue;
        
        for (int j = 0; j < nchans; ++j)
        {
            chmean_patch[j] += data[i*nchans + j];
			chvar_patch[j] += data[i*nchans + j] * data[i*nchans + j];
        }
        ++count;
    }

    for (long int j = 0; j < nchans; ++j) {
        chmean_patch[j] /= count;
        chvar_patch[j] = chvar_patch[j] / count - chmean_patch[j] * chmean_patch[j];
    }
    
    // Step 3: Fill patched time samples
    if (filltype == "mean")
    {
        for (long int i=0; i<nsamples; ++i)
		{
			if (!mask[i]) continue;
			for (long int j=0; j<nchans; ++j) {	data[i*nchans + j] = chmean_patch[j];}
		}
    }
    else if (filltype == "rand")
    {
        std::vector<std::random_device> r(nchans);
		std::vector<std::mt19937> generators;
		std::vector<std::normal_distribution<float>> distributions;
        
        for (long int j=0; j<nchans; ++j)
		{
			generators.emplace_back(std::mt19937(r[j]()));
			distributions.emplace_back(std::normal_distribution<float>(chmean_patch[j], std::sqrt(chvar_patch[j])));
		}

        for (long int i=0; i<nsamples; ++i)
		{
			if (!mask[i]) continue;
			for (long int j=0; j<nchans; ++j) { data[i*nchans + j] = distributions[j](generators[j]);}
        }
    }
}

// Band Equalization
void equalization(std::vector<float>& data, size_t nsamples, size_t nchans, std::vector<float>& chmean, std::vector<float>& chstd) {  
    chmean.assign(nchans, 0.0f);
    chstd.assign(nchans, 0.0f);

    // Step 1: Calculate Mean and Stddev
    for (int i = 0; i < nsamples; ++i)
    {
        for (int j = 0; j < nchans; ++j){
            chmean[j] += data[i*nchans + j];
            chstd[j] += data[i*nchans + j] * data[i*nchans + j];        
        }
    }
    for (int j = 0; j < nchans; ++j)
    {
        chmean[j] /= nsamples;
        chstd[j] /=  nsamples;
        chstd[j] -= chmean[j] * chmean[j];
        chstd[j] = std::sqrt(chstd[j]);
        if (chstd[j] == 0.)
            chstd[j] = 1.;
    }

    // Step2: Perform Band Equalization
    for (int i = 0; i < nsamples; ++i) {
        for (int j = 0; j < nchans; ++j) {
            data[i*nchans + j] = (data[i*nchans + j] - chmean[j]) / chstd[j];
        }
    }
}

// Baseline
template <typename T>
void sliding_median(const T* data, T* out, long size, int w)
{
    // low -pass filter; gives smooth gain drift curve
    w = std::min<long>(w, size); 

    std::multiset<T, std::greater<T>> low;   // max heap
    std::multiset<T, std::less<T>> high;     // min heap

    auto rebalance = [&]() {
        if (low.size() > high.size() + 1) {
            high.insert(*low.begin());
            low.erase(low.begin());
        } else if (high.size() > low.size() + 1) {
            low.insert(*high.begin());
            high.erase(high.begin());
        }
    };

    auto get_median = [&]() -> T {
        if (low.size() > high.size()) return *low.begin();
        else if (high.size() > low.size()) return *high.begin();
        return (*low.begin() + *high.begin()) / 2;
    };

    int a = -w/2 - 1;
    int b = (w - 1) / 2;

    // seed
    low.insert(data[0]);
    T median = data[0];

    for (int i = 1; i < b; ++i) {
        if (data[i] >= median) high.insert(data[i]);
        else low.insert(data[i]);
        rebalance();
        median = get_median();
    }

    // Step 2: Median of first half-window
    for (int i = 0; i < w/2 + 1 && b < size; ++i) {
        if (data[b] >= median) high.insert(data[b]);
        else low.insert(data[b]);
        rebalance();
        median = get_median();
        out[i] = median;
        ++a; ++b;
    }
    
    // Step 3: Median of full sliding window
    for (int i = w/2 + 1; i < size - (w-1)/2; ++i) {
        if (data[b] >= median) high.insert(data[b]);
        else low.insert(data[b]);

        auto it = low.find(data[a]);
        if (it != low.end()) low.erase(it);
        else high.erase(high.find(data[a]));

        rebalance();
        median = get_median();
        out[i] = median;

        ++a; ++b;
    }

    // Step 4: Median of tail window
    for (int i = size - (w-1)/2; i < size; ++i) {
        auto it = low.find(data[a]);
        if (it != low.end()) low.erase(it);
        else high.erase(high.find(data[a]));

        rebalance();
        median = get_median();
        out[i] = median;

        ++a;
    }
}

void baseline_filter(std::vector<float>& data, size_t nsamples, size_t nchans, float width, float tsamp) {
    std::vector<double> s(nsamples, 0.0);
    int window_size = width / tsamp;

    // Step 1: channel mean per time -- timeseries (power vs time) -- captures things common across frequencies
    for (long i = 0; i < nsamples; ++i) {
        double sum = 0.0;
        for (long j = 0; j < nchans; ++j)
            sum += data[i*nchans + j];
        s[i] = sum / nchans;
    }

    // Step 2: Median filter s
    std::vector<double> s_med(nsamples);
    sliding_median(s.data(), s_med.data(), nsamples, window_size);
    s = std::move(s_med);

    // Step 3: Regression terms estimation
    std::vector<double> xe(nchans, 0.0), xs(nchans, 0.0);
    double se = 0.0, ss = 0.0;
    for (long i = 0; i < nsamples; ++i) {
        
        for (long j = 0; j < nchans; ++j) {
            xe[j] += data[i*nchans + j];
            xs[j] += data[i*nchans + j] * s[i];
        }
        se += s[i];
        ss += s[i]*s[i];
    }

    // 4) coefficients
    double denom = se*se - ss*nsamples;
    std::vector<double> alpha(nchans, 0.0), beta(nchans, 0.0);
    if (denom != 0.0) {
        for (long j = 0; j < nchans; ++j) {
            alpha[j] = (xe[j]*se - xs[j]*nsamples) / denom;
            beta[j]  = (xs[j]*se - xe[j]*ss) / denom;
        }
    }

    // 5) subtract baseline
    for (long i = 0; i < nsamples; ++i) {
        for (long j = 0; j < nchans; ++j) {
            data[i*nchans + j] -= alpha[j] * s[i] + beta[j];
        }
    }
}

// Main function
int main(int argc, char* argv[]) {
    auto step0_start = std::chrono::high_resolution_clock::now();

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <binary_file_path> <block_size> <threshold>" << std::endl;
        return 1;
    }
    std::string in_file = argv[1];

    fs::path in_path(in_file);
    fs::path directory = in_path.parent_path();
    std::string stem = in_path.stem().string();
    std::string output_filename = stem + "_stripe.raw";
    fs::path out_path = directory / output_filename;
    std::string out_file = out_path.string();

    int block_size = std::stoi(argv[2]);
    float thresig = std::stof(argv[3]);
    std::string filltype = "rand"; // fixed replacement for masked time samples
    bool fill_rand = true; // fixed random value replacement for SKF flagged channels
    int nchans = 4096; // fixed number of channels 
    float outmean = 64.0;
    float outstd = 3.0;
    float width = 0.0;
    float tsamp = 1.31072e-3f;

    try
    {
        // Read the binary data
        auto raw_data = read_binary_data(in_file, nchans); 

        size_t nsamples = block_size;
        size_t block_len = block_size * nchans;
        size_t n_full = raw_data.size() / block_len;
        size_t remainder = raw_data.size() % block_len;
        size_t N = nsamples * nchans;

        // process data block by block
        for (int blk=0; blk< n_full; ++blk) {
            std::cout << "-----------Processing block " << blk+1 << " / " << n_full << std::endl;
            auto blk_start = std::chrono::high_resolution_clock::now();

            // Extract block
            std::vector<uint8_t> block_data(
                raw_data.begin() + blk * block_len,
                raw_data.begin() + (blk + 1) * block_len
            );
            std::vector<float> float_data;
            std::vector<float> chmean, chstd;
            uint8_to_float(block_data, float_data, N, 0.0f); // Convert uint8_t data to float
            patch_filter(float_data, nsamples, nchans, filltype);  // Time masking          
            skf_filter(float_data, thresig, nsamples, nchans, fill_rand); // SKF RFI mitigation
            equalization(float_data, nsamples, nchans, chmean, chstd); // Band Equalization
            baseline_filter(float_data, nsamples, nchans, width, tsamp); // Baseline Removal
            float_to_uint8(float_data, block_data, N, outmean, outstd); // Convert float data to uint8
       
            // Copy back the processed block
            std::copy(
                block_data.begin(),
                block_data.end(),
                raw_data.begin() + blk * block_len
            );
             
            auto blk_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::micro> blk_time = blk_end - blk_start;

            float rtf = (block_size * tsamp*1000000) / blk_time.count();
            std::cout << "Real-Time Factor for block " << blk + 1 << ": " << rtf << std::endl;
        }

        if (remainder > 0) {
            auto blk_start = std::chrono::high_resolution_clock::now();

            std::cout << "-----------Processing Last block " << std::endl;
            std::vector<uint8_t> block_data(
                raw_data.begin() + n_full * block_size,
                raw_data.end()
            );
            std::vector<float> float_data;
            std::vector<float> chmean, chstd;
            uint8_to_float(block_data, float_data, N, 0.0f);
            patch_filter(float_data, nsamples, nchans, filltype);
            skf_filter(float_data, thresig, nsamples, nchans, fill_rand);
            equalization(float_data, nsamples, nchans, chmean, chstd);
            baseline_filter(float_data, nsamples, nchans, width, tsamp);
            float_to_uint8(float_data, block_data, N, outmean, outstd);
            std::copy(
                block_data.begin(),
                block_data.end(),
                raw_data.begin() + n_full * block_size
            );
            auto blk_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::micro> blk_time = blk_end - blk_start;
            std::cout << blk_time.count() << "microsec" << std::endl;
            float rtf_last = (remainder * tsamp * 1000000) / (nchans * blk_time.count());
            std::cout << "Real-Time Factor for the last block: " << rtf_last << std::endl;
        }

        // Write the mitigated data back to a binary file
        write_binary_data(out_file, raw_data, nsamples, nchans);

        auto step0_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> step0_time = step0_end - step0_start;
        std::cout << "\nTime taken for RFI Mitigation " << step0_time.count() << " milliseconds" << std::endl;
        
        float total_rtf = (raw_data.size() * tsamp*1000) / (4096*step0_time.count()); 
        std::cout << "Total Real-Time Factor: " << total_rtf << std::endl;
    } 
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}