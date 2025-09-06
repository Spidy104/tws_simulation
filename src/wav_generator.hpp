#pragma once
#include <vector>
#include <string>
#include <memory> 
#include <sndfile.h> // <-- correct include

// Data structure for readWav return value for better readability
struct WavData {
    std::vector<float> samples;
    int channels{};
    int sample_rate{};
};

class WavGenerator {
private:
    int sample_rate;
    double duration_s;

    // RAII wrapper for SNDFILE using a custom deleter with unique_ptr
    // SNDFILE is an opaque struct typedef provided by <sndfile.h>
    using SndfileHandle = std::unique_ptr<SNDFILE, decltype(&sf_close)>;
    
    // Private helper to avoid code duplication in generate methods
    void writeWavFile(const std::string& filename, const std::vector<float>& buffer) const;

public:
    explicit WavGenerator(int sample_rate = 44100, double duration_s = 5.0) noexcept;

    // Simple stereo: left = freq_left, right = freq_right
    void generateStereoWav(const std::string& filename,
                           double freq_left,
                           double freq_right,
                           double amplitude = 0.8) const;

    // ANC test: left = signal + noise (primary), right = noise (reference)
    void generateANCWav(const std::string& filename,
                        double signal_freq = 440.0,
                        double noise_freq  = 200.0,
                        double signal_amp  = 0.8,
                        double noise_amp   = 0.3) const;

    // Read WAV: returns a struct for clarity
    [[nodiscard]]
    WavData readWav(const std::string& filename) const;
};
