/*
  ==============================================================================

    TransferFunctionEvaluator.h
    Created: 14 May 2025 10:58:51am
    Author:  janic

  ==============================================================================
*/

#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include "exprtk.hpp"

namespace dsp_core {

class ExpressionEvaluator
{
public:
    ExpressionEvaluator();

    bool compile(const std::string& expression);
    double evaluate(double x) const;

private:
    typedef exprtk::symbol_table<double> SymbolTable;
    typedef exprtk::expression<double> Expression;
    typedef exprtk::parser<double> Parser;

    double xVar = 0.0;
    SymbolTable symbolTable;
    Expression expression;
    Parser parser;
};

} // namespace dsp_core
