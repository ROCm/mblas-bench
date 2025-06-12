#pragma once
#include <map>
#include <string>
#include "mblasData.h"

// Forward declaration of mblasComputeType
class mblasComputeType;

class mblasDataType {
  public:
    // Public class "enum" values
    static const mblasDataType MBLAS_R_64F,     MBLAS_C_64F;
    static const mblasDataType MBLAS_R_32F,     MBLAS_C_32F;
    static const mblasDataType MBLAS_R_16F,     MBLAS_C_16F;
    static const mblasDataType MBLAS_R_16BF,    MBLAS_C_16BF;
    static const mblasDataType MBLAS_R_8F_E4M3, MBLAS_R_8F_E5M2;
    static const mblasDataType MBLAS_R_8F_UE4M3, MBLAS_R_8F_UE8M0;
    static const mblasDataType MBLAS_R_6F_E2M3, MBLAS_R_6F_E3M2;
    static const mblasDataType MBLAS_R_4F_E2M1;
    static const mblasDataType MBLAS_R_64I,     MBLAS_C_64I;
    static const mblasDataType MBLAS_R_64U,     MBLAS_C_64U;
    static const mblasDataType MBLAS_R_32I,     MBLAS_C_32I;
    static const mblasDataType MBLAS_R_32U,     MBLAS_C_32U;
    static const mblasDataType MBLAS_R_16I,     MBLAS_C_16I;
    static const mblasDataType MBLAS_R_16U,     MBLAS_C_16U;
    static const mblasDataType MBLAS_R_8I,      MBLAS_C_8I;
    static const mblasDataType MBLAS_R_8U,      MBLAS_C_8U;
    static const mblasDataType MBLAS_R_4I,      MBLAS_C_4I;
    static const mblasDataType MBLAS_R_4U,      MBLAS_C_4U;
    static const mblasDataType MBLAS_NULL,      MBLAS_ANY;

    // Internal enum
  private:
    // Internal enum value
    mblasDataTypeEnum value;
    static const std::map<std::string, mblasDataTypeEnum> precDType;
  protected:
    void set(const mblasDataType mdt) {
      value = mdt;
    }
    std::string to_string(std::string) const;
  public:
    explicit constexpr mblasDataType(int y = 0) : mblasDataType{static_cast<mblasDataTypeEnum>(y)} {}
    constexpr mblasDataType(mblasDataTypeEnum y) : value(y) {}
    constexpr operator mblasDataTypeEnum() const { return value; }
    //mblasDataType();
    mblasDataType(std::string instr);
    bool operator==(const mblasDataType& other) const;
    bool operator!=(const mblasDataType& other) const;
    bool operator<(const mblasDataType& other) const;
    bool operator>(const mblasDataType& other) const;
    bool operator<=(const mblasDataType& other) const;
    bool operator>=(const mblasDataType& other) const;
    virtual std::string to_string() const { return to_string("MBLAS"); }
    bool isReal() const;
    bool isFp8() const;
    bool isFp4() const;
    void set_scalar(std::string scalarstr, mblasDataType precision, mblasComputeType& compute);
};

constexpr const mblasDataType mblasDataType::MBLAS_R_64F{mblasDataTypeEnum::MBLAS_R_64F},         mblasDataType::MBLAS_C_64F{mblasDataTypeEnum::MBLAS_C_64F};
constexpr const mblasDataType mblasDataType::MBLAS_R_32F{mblasDataTypeEnum::MBLAS_R_32F},         mblasDataType::MBLAS_C_32F{mblasDataTypeEnum::MBLAS_C_32F};
constexpr const mblasDataType mblasDataType::MBLAS_R_16F{mblasDataTypeEnum::MBLAS_R_16F},         mblasDataType::MBLAS_C_16F{mblasDataTypeEnum::MBLAS_C_16F};
constexpr const mblasDataType mblasDataType::MBLAS_R_16BF{mblasDataTypeEnum::MBLAS_R_16BF},       mblasDataType::MBLAS_C_16BF{mblasDataTypeEnum::MBLAS_C_16BF};
constexpr const mblasDataType mblasDataType::MBLAS_R_8F_E4M3{mblasDataTypeEnum::MBLAS_R_8F_E4M3}, mblasDataType::MBLAS_R_8F_E5M2{mblasDataTypeEnum::MBLAS_R_8F_E5M2};
constexpr const mblasDataType mblasDataType::MBLAS_R_8F_UE4M3{mblasDataTypeEnum::MBLAS_R_8F_UE4M3}, mblasDataType::MBLAS_R_8F_UE8M0{mblasDataTypeEnum::MBLAS_R_8F_UE8M0};
constexpr const mblasDataType mblasDataType::MBLAS_R_6F_E2M3{mblasDataTypeEnum::MBLAS_R_6F_E2M3}, mblasDataType::MBLAS_R_6F_E3M2{mblasDataTypeEnum::MBLAS_R_6F_E3M2};
constexpr const mblasDataType mblasDataType::MBLAS_R_4F_E2M1{mblasDataTypeEnum::MBLAS_R_4F_E2M1};
constexpr const mblasDataType mblasDataType::MBLAS_R_64I{mblasDataTypeEnum::MBLAS_R_64I},         mblasDataType::MBLAS_C_64I{mblasDataTypeEnum::MBLAS_C_64I};
constexpr const mblasDataType mblasDataType::MBLAS_R_64U{mblasDataTypeEnum::MBLAS_R_64U},         mblasDataType::MBLAS_C_64U{mblasDataTypeEnum::MBLAS_C_64U};
constexpr const mblasDataType mblasDataType::MBLAS_R_32I{mblasDataTypeEnum::MBLAS_R_32I},         mblasDataType::MBLAS_C_32I{mblasDataTypeEnum::MBLAS_C_32I};
constexpr const mblasDataType mblasDataType::MBLAS_R_32U{mblasDataTypeEnum::MBLAS_R_32U},         mblasDataType::MBLAS_C_32U{mblasDataTypeEnum::MBLAS_C_32U};
constexpr const mblasDataType mblasDataType::MBLAS_R_16I{mblasDataTypeEnum::MBLAS_R_16I},         mblasDataType::MBLAS_C_16I{mblasDataTypeEnum::MBLAS_C_16I};
constexpr const mblasDataType mblasDataType::MBLAS_R_16U{mblasDataTypeEnum::MBLAS_R_16U},         mblasDataType::MBLAS_C_16U{mblasDataTypeEnum::MBLAS_C_16U};
constexpr const mblasDataType mblasDataType::MBLAS_R_8I{mblasDataTypeEnum::MBLAS_R_8I},           mblasDataType::MBLAS_C_8I{mblasDataTypeEnum::MBLAS_C_8I};
constexpr const mblasDataType mblasDataType::MBLAS_R_8U{mblasDataTypeEnum::MBLAS_R_8U},           mblasDataType::MBLAS_C_8U{mblasDataTypeEnum::MBLAS_C_8U};
constexpr const mblasDataType mblasDataType::MBLAS_R_4I{mblasDataTypeEnum::MBLAS_R_4I},           mblasDataType::MBLAS_C_4I{mblasDataTypeEnum::MBLAS_C_4I};
constexpr const mblasDataType mblasDataType::MBLAS_R_4U{mblasDataTypeEnum::MBLAS_R_4U},           mblasDataType::MBLAS_C_4U{mblasDataTypeEnum::MBLAS_C_4U};
constexpr const mblasDataType mblasDataType::MBLAS_NULL{mblasDataTypeEnum::MBLAS_NULL},           mblasDataType::MBLAS_ANY{mblasDataTypeEnum::MBLAS_ANY};


