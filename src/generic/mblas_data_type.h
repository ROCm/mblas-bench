#pragma once
#include <map>
#include <string>
#include "mblas_data.h"

// Forward declaration of mblas_compute_type
class mblas_compute_type;

class mblas_data_type {
  public:
    // Public class "enum" values
    static const mblas_data_type MBLAS_R_64F,     MBLAS_C_64F;
    static const mblas_data_type MBLAS_R_32F,     MBLAS_C_32F;
    static const mblas_data_type MBLAS_R_16F,     MBLAS_C_16F;
    static const mblas_data_type MBLAS_R_16BF,    MBLAS_C_16BF;
    static const mblas_data_type MBLAS_R_8F_E4M3, MBLAS_R_8F_E5M2;
    static const mblas_data_type MBLAS_R_8F_UE4M3, MBLAS_R_8F_UE8M0;
    static const mblas_data_type MBLAS_R_6F_E2M3, MBLAS_R_6F_E3M2;
    static const mblas_data_type MBLAS_R_4F_E2M1;
    static const mblas_data_type MBLAS_R_64I,     MBLAS_C_64I;
    static const mblas_data_type MBLAS_R_64U,     MBLAS_C_64U;
    static const mblas_data_type MBLAS_R_32I,     MBLAS_C_32I;
    static const mblas_data_type MBLAS_R_32U,     MBLAS_C_32U;
    static const mblas_data_type MBLAS_R_16I,     MBLAS_C_16I;
    static const mblas_data_type MBLAS_R_16U,     MBLAS_C_16U;
    static const mblas_data_type MBLAS_R_8I,      MBLAS_C_8I;
    static const mblas_data_type MBLAS_R_8U,      MBLAS_C_8U;
    static const mblas_data_type MBLAS_R_4I,      MBLAS_C_4I;
    static const mblas_data_type MBLAS_R_4U,      MBLAS_C_4U;
    static const mblas_data_type MBLAS_NULL,      MBLAS_ANY;

    // Internal enum
  private:
    // Internal enum value
    mblas_data_type_enum value;
    static const std::map<std::string, mblas_data_type_enum> precDType;
  protected:
    void set(const mblas_data_type mdt) {
      value = mdt;
    }
    std::string to_string(std::string) const;
  public:
    explicit constexpr mblas_data_type(int y = 0) : mblas_data_type{static_cast<mblas_data_type_enum>(y)} {}
    constexpr mblas_data_type(mblas_data_type_enum y) : value(y) {}
    constexpr operator mblas_data_type_enum() const { return value; }
    //mblas_data_type();
    mblas_data_type(std::string instr);
    bool operator==(const mblas_data_type& other) const;
    bool operator!=(const mblas_data_type& other) const;
    bool operator<(const mblas_data_type& other) const;
    bool operator>(const mblas_data_type& other) const;
    bool operator<=(const mblas_data_type& other) const;
    bool operator>=(const mblas_data_type& other) const;
    virtual std::string to_string() const { return to_string("MBLAS"); }
    bool is_real() const;
    bool is_fp8() const;
    bool is_fp4() const;
    int get_packing_count() const;
    void set_scalar(std::string scalarstr, mblas_data_type precision, mblas_compute_type& compute);
};

constexpr const mblas_data_type mblas_data_type::MBLAS_R_64F{mblas_data_type_enum::MBLAS_R_64F},         mblas_data_type::MBLAS_C_64F{mblas_data_type_enum::MBLAS_C_64F};
constexpr const mblas_data_type mblas_data_type::MBLAS_R_32F{mblas_data_type_enum::MBLAS_R_32F},         mblas_data_type::MBLAS_C_32F{mblas_data_type_enum::MBLAS_C_32F};
constexpr const mblas_data_type mblas_data_type::MBLAS_R_16F{mblas_data_type_enum::MBLAS_R_16F},         mblas_data_type::MBLAS_C_16F{mblas_data_type_enum::MBLAS_C_16F};
constexpr const mblas_data_type mblas_data_type::MBLAS_R_16BF{mblas_data_type_enum::MBLAS_R_16BF},       mblas_data_type::MBLAS_C_16BF{mblas_data_type_enum::MBLAS_C_16BF};
constexpr const mblas_data_type mblas_data_type::MBLAS_R_8F_E4M3{mblas_data_type_enum::MBLAS_R_8F_E4M3}, mblas_data_type::MBLAS_R_8F_E5M2{mblas_data_type_enum::MBLAS_R_8F_E5M2};
constexpr const mblas_data_type mblas_data_type::MBLAS_R_8F_UE4M3{mblas_data_type_enum::MBLAS_R_8F_UE4M3}, mblas_data_type::MBLAS_R_8F_UE8M0{mblas_data_type_enum::MBLAS_R_8F_UE8M0};
constexpr const mblas_data_type mblas_data_type::MBLAS_R_6F_E2M3{mblas_data_type_enum::MBLAS_R_6F_E2M3}, mblas_data_type::MBLAS_R_6F_E3M2{mblas_data_type_enum::MBLAS_R_6F_E3M2};
constexpr const mblas_data_type mblas_data_type::MBLAS_R_4F_E2M1{mblas_data_type_enum::MBLAS_R_4F_E2M1};
constexpr const mblas_data_type mblas_data_type::MBLAS_R_64I{mblas_data_type_enum::MBLAS_R_64I},         mblas_data_type::MBLAS_C_64I{mblas_data_type_enum::MBLAS_C_64I};
constexpr const mblas_data_type mblas_data_type::MBLAS_R_64U{mblas_data_type_enum::MBLAS_R_64U},         mblas_data_type::MBLAS_C_64U{mblas_data_type_enum::MBLAS_C_64U};
constexpr const mblas_data_type mblas_data_type::MBLAS_R_32I{mblas_data_type_enum::MBLAS_R_32I},         mblas_data_type::MBLAS_C_32I{mblas_data_type_enum::MBLAS_C_32I};
constexpr const mblas_data_type mblas_data_type::MBLAS_R_32U{mblas_data_type_enum::MBLAS_R_32U},         mblas_data_type::MBLAS_C_32U{mblas_data_type_enum::MBLAS_C_32U};
constexpr const mblas_data_type mblas_data_type::MBLAS_R_16I{mblas_data_type_enum::MBLAS_R_16I},         mblas_data_type::MBLAS_C_16I{mblas_data_type_enum::MBLAS_C_16I};
constexpr const mblas_data_type mblas_data_type::MBLAS_R_16U{mblas_data_type_enum::MBLAS_R_16U},         mblas_data_type::MBLAS_C_16U{mblas_data_type_enum::MBLAS_C_16U};
constexpr const mblas_data_type mblas_data_type::MBLAS_R_8I{mblas_data_type_enum::MBLAS_R_8I},           mblas_data_type::MBLAS_C_8I{mblas_data_type_enum::MBLAS_C_8I};
constexpr const mblas_data_type mblas_data_type::MBLAS_R_8U{mblas_data_type_enum::MBLAS_R_8U},           mblas_data_type::MBLAS_C_8U{mblas_data_type_enum::MBLAS_C_8U};
constexpr const mblas_data_type mblas_data_type::MBLAS_R_4I{mblas_data_type_enum::MBLAS_R_4I},           mblas_data_type::MBLAS_C_4I{mblas_data_type_enum::MBLAS_C_4I};
constexpr const mblas_data_type mblas_data_type::MBLAS_R_4U{mblas_data_type_enum::MBLAS_R_4U},           mblas_data_type::MBLAS_C_4U{mblas_data_type_enum::MBLAS_C_4U};
constexpr const mblas_data_type mblas_data_type::MBLAS_NULL{mblas_data_type_enum::MBLAS_NULL},           mblas_data_type::MBLAS_ANY{mblas_data_type_enum::MBLAS_ANY};


