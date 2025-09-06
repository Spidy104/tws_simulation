// src/battery_model.cpp
#include "battery_model.hpp"
#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>      // <<< add this: enables std::vector/std::string conversions
#include <pybind11/stl_bind.h> // optional; not required for simple types
#include <pybind11/complex.h>  // only if you use complex types

namespace py = pybind11;

// --- existing implementation (unchanged) ---

BatteryModel::BatteryModel(float capacity,
                           float initial_level_percent,
                           float base_drain,
                           float anc_drain) noexcept
    : capacity_mAh(capacity),
      // clamp the initial percent to [0,100] and compute initial charge in mAh
      current_charge_mAh(std::clamp(initial_level_percent, 0.0f, 100.0f) * 0.01f * capacity),
      drain_rate_base_mAh_per_s(base_drain),
      drain_rate_anc_mAh_per_s(anc_drain)
{
    if (capacity_mAh <= 0.0f) {
        capacity_mAh = 1.0f;
        current_charge_mAh = std::clamp(initial_level_percent, 0.0f, 100.0f) * 0.01f * capacity_mAh;
    }
}

void BatteryModel::update(float duration_s, bool anc_active) {
    if (duration_s <= 0.0f) {
        return;
    }

    const float base_drain = duration_s * drain_rate_base_mAh_per_s;
    const float anc_drain_total = anc_active ? (duration_s * drain_rate_anc_mAh_per_s) : 0.0f;
    const float total_drain = base_drain + anc_drain_total;

    current_charge_mAh = std::clamp(current_charge_mAh - total_drain, 0.0f, capacity_mAh);
}

float BatteryModel::getLevelPercent() const noexcept {
    if (capacity_mAh <= 0.0f) return 0.0f;
    return (current_charge_mAh / capacity_mAh) * 100.0f;
}

float BatteryModel::getCurrentCharge() const noexcept {
    return current_charge_mAh;
}

float BatteryModel::getCapacity() const noexcept {
    return capacity_mAh;
}

// -------------------- pybind11 bindings --------------------

PYBIND11_MODULE(battery_model, m) {
    m.doc() = "BatteryModel pybind11 bindings";

    // We'll expose a Python constructor BatteryModel(level_percent, capacity)
    // but internally the C++ constructor is BatteryModel(capacity, initial_level_percent, base_drain, anc_drain).
    // Provide sensible default drain rates so Python callers don't need to specify them.
    constexpr float DEFAULT_BASE_DRAIN = 0.01f; // mAh per second (toy default)
    constexpr float DEFAULT_ANC_DRAIN  = 0.01f; // additional mAh/s when ANC is enabled

    py::class_<BatteryModel>(m, "BatteryModel")
        .def(py::init([](float level_percent, float capacity) {
                 // map Python args to C++ constructor ordering
                 return BatteryModel(capacity, level_percent, DEFAULT_BASE_DRAIN, DEFAULT_ANC_DRAIN);
             }),
             py::arg("level") = 50.0f,
             py::arg("capacity") = 100.0f,
             "Create BatteryModel(level_percent, capacity). Internally uses sensible default drain rates.")

        .def("update", &BatteryModel::update,
             py::arg("duration_s"),
             py::arg("anc_enabled"),
             "Update battery consumption over duration (seconds); anc_enabled increases drain if true.")

        // Expose get_level() for Python but call the C++ getLevelPercent()
        .def("get_level", &BatteryModel::getLevelPercent,
             "Return current battery level (%)")

        // Additional helpers
        .def("get_current_charge", &BatteryModel::getCurrentCharge,
             "Return current charge in mAh")

        .def("get_capacity", &BatteryModel::getCapacity,
             "Return battery capacity in mAh");
}
