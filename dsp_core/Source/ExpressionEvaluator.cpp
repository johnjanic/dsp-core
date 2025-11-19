/*
  ==============================================================================

    TransferFunctionEvaluator.cpp
    Created: 14 May 2025 10:58:31am
    Author:  janic

  ==============================================================================
*/

#include "ExpressionEvaluator.h"

namespace dsp_core {

ExpressionEvaluator::ExpressionEvaluator() {
    symbolTable.add_variable("x", xVar);
    symbolTable.add_constants();
    expression.register_symbol_table(symbolTable);
}

bool ExpressionEvaluator::compile(const std::string& expressionStr) {
    return parser.compile(expressionStr, expression);
}

double ExpressionEvaluator::evaluate(double x) const {
    const_cast<double&>(xVar) = x; // set the input variable
    return expression.value();
}

} // namespace dsp_core
