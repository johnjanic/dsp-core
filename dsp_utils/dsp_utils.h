/*******************************************************************************
 The block below describes the properties of this module, and is read by
 the Projucer to automatically generate project code that uses it.
 For details about the syntax and how to create or use a module, see the
 JUCE Module Format.md file.


 BEGIN_JUCE_MODULE_DECLARATION

  ID:                 dsp_utils
  vendor:             tbd
  version:            1.0.0
  name:               Transfer Function Editor
  description:        A module for creating and editing transfer functions in audio applications.
  website:            tbd
  license:            proprietary/commercial
  minimumCppStandard: 17

  dependencies:       juce_core, juce_data_structures, juce_audio_processors

 END_JUCE_MODULE_DECLARATION

*******************************************************************************/


#pragma once
#define DSP_UTILS_H_INCLUDED

#include <juce_data_structures/juce_data_structures.h>

#include "Source/AudioHistoryBuffer.h"
