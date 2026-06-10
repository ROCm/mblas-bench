#pragma once
#include <map>
#include <string>
#include "mblas_data.h"

class mblas_operation {
  public:
    // Public class "enum" values
    static const mblas_operation MBLAS_OP_N;
    static const mblas_operation MBLAS_OP_T;
    static const mblas_operation MBLAS_OP_C;
    static const mblas_operation MBLAS_OP_CONJG;
    static const mblas_operation MBLAS_OP_NULL;

  private:
    // Internal enum value
    mblas_operation_enum value;
    static const std::map<std::string, mblas_operation_enum> opDType;
    static const std::map<mblas_operation_enum, std::string> opSShort;
  protected:
    void set(const mblas_operation mdt) {
      value = mdt;
    }
    std::string to_string(std::string) const;
  public:
    explicit constexpr mblas_operation(int y = 0) : mblas_operation{static_cast<mblas_operation_enum>(y)} {}
    constexpr mblas_operation(mblas_operation_enum y) : value(y) {}
    constexpr operator mblas_operation_enum() const { return value; }
    //mblas_operation();
    mblas_operation(std::string instr);
    bool operator==(const mblas_operation& other) const;
    bool operator!=(const mblas_operation& other) const;
    bool operator<(const mblas_operation& other) const;
    bool operator>(const mblas_operation& other) const;
    bool operator<=(const mblas_operation& other) const;
    bool operator>=(const mblas_operation& other) const;
    virtual std::string to_string() const { return to_string("MBLAS"); }
    std::string to_string_short();
};

constexpr const mblas_operation mblas_operation::MBLAS_OP_N{mblas_operation_enum::MBLAS_OP_N};
constexpr const mblas_operation mblas_operation::MBLAS_OP_T{mblas_operation_enum::MBLAS_OP_T};
constexpr const mblas_operation mblas_operation::MBLAS_OP_C{mblas_operation_enum::MBLAS_OP_C};
constexpr const mblas_operation mblas_operation::MBLAS_OP_CONJG{mblas_operation_enum::MBLAS_OP_CONJG};
constexpr const mblas_operation mblas_operation::MBLAS_OP_NULL{mblas_operation_enum::MBLAS_OP_NULL};


