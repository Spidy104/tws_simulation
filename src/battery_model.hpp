#pragma once

class BatteryModel {
private:
    // --- State is now stored in absolute units (mAh) ---
    float capacity_mAh;
    float current_charge_mAh; // Renamed from current_level for clarity

    // --- Drain rates are now const as they are set only once ---
    const float drain_rate_base_mAh_per_s;
    const float drain_rate_anc_mAh_per_s;

public:
    // --- Constructor is updated for more flexibility ---
    explicit BatteryModel(float capacity = 50.0f,
                          float initial_level_percent = 100.0f,
                          float base_drain = 0.005f,
                          float anc_drain = 0.002f) noexcept;

    // Update battery state over duration_s seconds; anc_active adds extra drain
    void update(float duration_s, bool anc_active);

    // --- Getters are updated to reflect the new state representation ---
    // Returns the battery level as a percentage
    [[nodiscard]] float getLevelPercent() const noexcept;

    // Returns the current charge in mAh
    [[nodiscard]] float getCurrentCharge() const noexcept;

    [[nodiscard]] float getCapacity() const noexcept;
};