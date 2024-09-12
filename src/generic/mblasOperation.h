#pragma once
#include <map>
#include <string>
#include "mblasData.h"

class mblasOperation {
  public:
    // Public class "enum" values
    static const mblasOperation MBLAS_OP_N;
    static const mblasOperation MBLAS_OP_T;
    static const mblasOperation MBLAS_OP_C;
    static const mblasOperation MBLAS_OP_CONJG;
    static const mblasOperation MBLAS_OP_NULL;

  private:
    // Internal enum value
    mblasOperationEnum value;
    static const std::map<std::string, mblasOperationEnum> opDType;
    static const std::map<mblasOperationEnum, std::string> opSShort;
  protected:
    void set(const mblasOperation mdt) {
      value = mdt;
    }
    std::string toString(std::string) const;
  public:
    explicit constexpr mblasOperation(int y = 0) : mblasOperation{static_cast<mblasOperationEnum>(y)} {}
    constexpr mblasOperation(mblasOperationEnum y) : value(y) {}
    constexpr operator mblasOperationEnum() const { return value; }
    //mblasOperation();
    mblasOperation(std::string instr);
    bool operator==(const mblasOperation& other) const;
    bool operator!=(const mblasOperation& other) const;
    bool operator<(const mblasOperation& other) const;
    bool operator>(const mblasOperation& other) const;
    bool operator<=(const mblasOperation& other) const;
    bool operator>=(const mblasOperation& other) const;
    virtual std::string toString() const { return toString("MBLAS"); }
    std::string toStringShort();
};

constexpr const mblasOperation mblasOperation::MBLAS_OP_N{mblasOperationEnum::MBLAS_OP_N};
constexpr const mblasOperation mblasOperation::MBLAS_OP_T{mblasOperationEnum::MBLAS_OP_T};
constexpr const mblasOperation mblasOperation::MBLAS_OP_C{mblasOperationEnum::MBLAS_OP_C};
constexpr const mblasOperation mblasOperation::MBLAS_OP_CONJG{mblasOperationEnum::MBLAS_OP_CONJG};
constexpr const mblasOperation mblasOperation::MBLAS_OP_NULL{mblasOperationEnum::MBLAS_OP_NULL};


