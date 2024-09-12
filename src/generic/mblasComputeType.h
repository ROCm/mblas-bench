#pragma once
#include <map>
#include <string>
#include "mblasData.h"

// Forward declaration of mblasDataType
class mblasDataType;

class mblasComputeType {
  public:
    // Public class "enum" values
    static const mblasComputeType MBLAS_COMPUTE_16F;
    static const mblasComputeType MBLAS_COMPUTE_16F_PEDANTIC;
    static const mblasComputeType MBLAS_COMPUTE_32F;
    static const mblasComputeType MBLAS_COMPUTE_32F_PEDANTIC;
    static const mblasComputeType MBLAS_COMPUTE_32F_FAST_16F;
    static const mblasComputeType MBLAS_COMPUTE_32F_FAST_16BF;
    static const mblasComputeType MBLAS_COMPUTE_32F_FAST_TF32;
    static const mblasComputeType MBLAS_COMPUTE_64F;
    static const mblasComputeType MBLAS_COMPUTE_64F_PEDANTIC;
    static const mblasComputeType MBLAS_COMPUTE_32I;
    static const mblasComputeType MBLAS_COMPUTE_32I_PEDANTIC;
    static const mblasComputeType MBLAS_COMPUTE_NULL;

    // Internal enum
  private:
    // Internal enum value
    mblasComputeTypeEnum value;
    static const std::map<std::string, mblasComputeTypeEnum> computeDType;
  protected:
    void set(const mblasComputeType mdt) {
      value = mdt;
    }
    std::string toString(std::string) const;
  public:
    explicit constexpr mblasComputeType(int y = 0) : mblasComputeType{static_cast<mblasComputeTypeEnum>(y)} {}
    constexpr mblasComputeType(mblasComputeTypeEnum y) : value(y) {}
    constexpr operator mblasComputeTypeEnum() const { return value; }
    //mblasComputeType();
    mblasComputeType(std::string instr);
    bool operator==(const mblasComputeType& other) const;
    bool operator!=(const mblasComputeType& other) const;
    bool operator<(const mblasComputeType& other) const;
    bool operator>(const mblasComputeType& other) const;
    bool operator<=(const mblasComputeType& other) const;
    bool operator>=(const mblasComputeType& other) const;

    virtual std::string toString() const { return toString("MBLAS"); }
  
    void setCompute(std::string computestr, mblasDataType& precision);
};

constexpr const mblasComputeType mblasComputeType::MBLAS_COMPUTE_16F{mblasComputeTypeEnum::MBLAS_COMPUTE_16F};
constexpr const mblasComputeType mblasComputeType::MBLAS_COMPUTE_16F_PEDANTIC{mblasComputeTypeEnum::MBLAS_COMPUTE_16F_PEDANTIC};
constexpr const mblasComputeType mblasComputeType::MBLAS_COMPUTE_32F{mblasComputeTypeEnum::MBLAS_COMPUTE_32F};
constexpr const mblasComputeType mblasComputeType::MBLAS_COMPUTE_32F_PEDANTIC{mblasComputeTypeEnum::MBLAS_COMPUTE_32F_PEDANTIC};
constexpr const mblasComputeType mblasComputeType::MBLAS_COMPUTE_32F_FAST_16F{mblasComputeTypeEnum::MBLAS_COMPUTE_32F_FAST_16F};
constexpr const mblasComputeType mblasComputeType::MBLAS_COMPUTE_32F_FAST_16BF{mblasComputeTypeEnum::MBLAS_COMPUTE_32F_FAST_16BF};
constexpr const mblasComputeType mblasComputeType::MBLAS_COMPUTE_32F_FAST_TF32{mblasComputeTypeEnum::MBLAS_COMPUTE_32F_FAST_TF32};
constexpr const mblasComputeType mblasComputeType::MBLAS_COMPUTE_64F{mblasComputeTypeEnum::MBLAS_COMPUTE_64F};
constexpr const mblasComputeType mblasComputeType::MBLAS_COMPUTE_64F_PEDANTIC{mblasComputeTypeEnum::MBLAS_COMPUTE_64F_PEDANTIC};
constexpr const mblasComputeType mblasComputeType::MBLAS_COMPUTE_32I{mblasComputeTypeEnum::MBLAS_COMPUTE_32I};
constexpr const mblasComputeType mblasComputeType::MBLAS_COMPUTE_32I_PEDANTIC{mblasComputeTypeEnum::MBLAS_COMPUTE_32I_PEDANTIC};
constexpr const mblasComputeType mblasComputeType::MBLAS_COMPUTE_NULL{mblasComputeTypeEnum::MBLAS_COMPUTE_NULL};