#pragma once
#include <vector>
#include <random>
#include <span>
#include <memory>
#include <cstddef>

class AudioProcessor {
public:
    // Constructor with validation
    AudioProcessor(int filter_len = 32, float mu = 0.1f, unsigned int seed = 42);
    
    // Copy constructor and assignment operator
    AudioProcessor(const AudioProcessor& other);
    AudioProcessor& operator=(const AudioProcessor& other);
    
    // Move constructor and assignment operator
    AudioProcessor(AudioProcessor&& other) noexcept;
    AudioProcessor& operator=(AudioProcessor&& other) noexcept;
    
    // Destructor
    ~AudioProcessor() = default;

    // Static helpers with better error handling
    static std::vector<float> deinterleaveChannel(std::span<const float> interleaved_buffer,
                                                  int num_channels,
                                                  int channel_to_extract);
    
    static std::vector<float> getLeftChannel(std::span<const float> stereo_buffer);
    static std::vector<float> getRightChannel(std::span<const float> stereo_buffer);

    // Network simulation effects
    std::vector<float> applyPacketLoss(std::span<const float> buffer,
                                       float loss_prob,
                                       int sample_rate = 44100,
                                       int packet_ms = 20);
    
    std::vector<float> applyJitter(std::span<const float> buffer,
                                   float jitter_ms,
                                   int sample_rate = 44100,
                                   int packet_ms = 20);

    // Adaptive Noise Cancellation
    std::vector<float> applyANC(std::span<const float> primary,
                                std::span<const float> reference);

    // Parameter getters and setters with validation
    void setMu(float mu);
    float getMu() const noexcept { return mu_; }
    
    void setLeakage(float leak);
    float getLeakage() const noexcept { return leak_; }
    
    void setFilterLength(int filter_len);
    int getFilterLength() const noexcept { return filter_len_; }
    
    // Filter state management
    void resetFilter();
    std::vector<float> getFilterWeights() const;
    void setFilterWeights(const std::vector<float>& weights);
    
    // Performance and diagnostic methods
    double getConvergenceMetric() const noexcept { return convergence_metric_; }
    size_t getProcessedSamples() const noexcept { return processed_samples_; }
    
    // Thread safety
    void enableThreadSafety(bool enable = true);

private:
    // Core parameters
    std::vector<float> filter_weights_;
    int filter_len_;
    float mu_;
    float leak_;
    
    // Circular buffer for reference samples (more memory efficient)
    std::vector<double> reference_buffer_;
    size_t buffer_index_;
    
    // Performance tracking
    double convergence_metric_;
    size_t processed_samples_;
    
    // Random number generation
    mutable std::mt19937 rng_;
    
    // Thread safety
    mutable std::unique_ptr<std::mutex> mutex_;
    
    // Internal validation helpers
    void validateParameters() const;
    void ensureBufferSize();
    
    // Constants
    static constexpr double DEFAULT_EPSILON = 1e-8;
    static constexpr float MAX_MU = 2.0f;
    static constexpr float MIN_MU = 1e-6f;
    static constexpr float MAX_LEAKAGE = 1.0f;
    static constexpr int MAX_FILTER_LENGTH = 8192;
    static constexpr int MIN_FILTER_LENGTH = 1;

public:
    // For pybind11 access - kept public but marked as implementation detail
    std::vector<float>& getFilterWeightsRef() { return filter_weights_; }
    std::mt19937& getRngRef() { return rng_; }
};
