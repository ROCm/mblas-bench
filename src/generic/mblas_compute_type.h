#pragma once
#include <map>
#include <string>
#include "mblas_data.h"

// Forward declaration of mblas_data_type
class mblas_data_type;

class mblas_compute_type {
  public:
    // Public class "enum" values
    static const mblas_compute_type MBLAS_COMPUTE_16F;
    static const mblas_compute_type MBLAS_COMPUTE_16F_PEDANTIC;
    static const mblas_compute_type MBLAS_COMPUTE_32F;
    static const mblas_compute_type MBLAS_COMPUTE_32F_PEDANTIC;
    static const mblas_compute_type MBLAS_COMPUTE_32F_FAST_16F;
    static const mblas_compute_type MBLAS_COMPUTE_32F_FAST_16BF;
    static const mblas_compute_type MBLAS_COMPUTE_32F_FAST_TF32;
    static const mblas_compute_type MBLAS_COMPUTE_32F_EMULATED_16BFX9;
    static const mblas_compute_type MBLAS_COMPUTE_64F;
    static const mblas_compute_type MBLAS_COMPUTE_64F_PEDANTIC;
    static const mblas_compute_type MBLAS_COMPUTE_32I;
    static const mblas_compute_type MBLAS_COMPUTE_32I_PEDANTIC;
    static const mblas_compute_type MBLAS_COMPUTE_NULL;

    // Internal enum
  private:
    // Internal enum value
    mblas_compute_type_enum value;
    static const std::map<std::string, mblas_compute_type_enum> computeDType;
  protected:
    void set(const mblas_compute_type mdt) {
      value = mdt;
    }
    std::string to_string(std::string) const;
  public:
    explicit constexpr mblas_compute_type(int y = 0) : mblas_compute_type{static_cast<mblas_compute_type_enum>(y)} {}
    constexpr mblas_compute_type(mblas_compute_type_enum y) : value(y) {}
    constexpr operator mblas_compute_type_enum() const { return value; }
    //mblas_compute_type();
    mblas_compute_type(std::string instr);
    bool operator==(const mblas_compute_type& other) const;
    bool operator!=(const mblas_compute_type& other) const;
    bool operator<(const mblas_compute_type& other) const;
    bool operator>(const mblas_compute_type& other) const;
    bool operator<=(const mblas_compute_type& other) const;
    bool operator>=(const mblas_compute_type& other) const;

    virtual std::string to_string() const { return to_string("MBLAS"); }
  
    virtual void set_compute(std::string computestr, mblas_data_type& precision);
};

constexpr const mblas_compute_type mblas_compute_type::MBLAS_COMPUTE_16F{mblas_compute_type_enum::MBLAS_COMPUTE_16F};
constexpr const mblas_compute_type mblas_compute_type::MBLAS_COMPUTE_16F_PEDANTIC{mblas_compute_type_enum::MBLAS_COMPUTE_16F_PEDANTIC};
constexpr const mblas_compute_type mblas_compute_type::MBLAS_COMPUTE_32F{mblas_compute_type_enum::MBLAS_COMPUTE_32F};
constexpr const mblas_compute_type mblas_compute_type::MBLAS_COMPUTE_32F_PEDANTIC{mblas_compute_type_enum::MBLAS_COMPUTE_32F_PEDANTIC};
constexpr const mblas_compute_type mblas_compute_type::MBLAS_COMPUTE_32F_FAST_16F{mblas_compute_type_enum::MBLAS_COMPUTE_32F_FAST_16F};
constexpr const mblas_compute_type mblas_compute_type::MBLAS_COMPUTE_32F_FAST_16BF{mblas_compute_type_enum::MBLAS_COMPUTE_32F_FAST_16BF};
constexpr const mblas_compute_type mblas_compute_type::MBLAS_COMPUTE_32F_FAST_TF32{mblas_compute_type_enum::MBLAS_COMPUTE_32F_FAST_TF32};
constexpr const mblas_compute_type mblas_compute_type::MBLAS_COMPUTE_32F_EMULATED_16BFX9{mblas_compute_type_enum::MBLAS_COMPUTE_32F_EMULATED_16BFX9};
constexpr const mblas_compute_type mblas_compute_type::MBLAS_COMPUTE_64F{mblas_compute_type_enum::MBLAS_COMPUTE_64F};
constexpr const mblas_compute_type mblas_compute_type::MBLAS_COMPUTE_64F_PEDANTIC{mblas_compute_type_enum::MBLAS_COMPUTE_64F_PEDANTIC};
constexpr const mblas_compute_type mblas_compute_type::MBLAS_COMPUTE_32I{mblas_compute_type_enum::MBLAS_COMPUTE_32I};
constexpr const mblas_compute_type mblas_compute_type::MBLAS_COMPUTE_32I_PEDANTIC{mblas_compute_type_enum::MBLAS_COMPUTE_32I_PEDANTIC};
constexpr const mblas_compute_type mblas_compute_type::MBLAS_COMPUTE_NULL{mblas_compute_type_enum::MBLAS_COMPUTE_NULL};