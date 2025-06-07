#pragma once

#include <juce_core/juce_core.h>
#include <juce_data_structures/juce_data_structures.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <functional>

using namespace juce;

namespace Service
{
    class PresetManager : public ValueTree::Listener
    {
    public:
        static const String presetNameProperty;
        static const File defaultDirectory;
        static const String extension;

        // Replace processor reference with state callbacks
        using SaveStateCallback = std::function<void(juce::MemoryBlock&)>;
        using RestoreStateCallback = std::function<void(const void*, int)>;

        PresetManager(SaveStateCallback saveState, RestoreStateCallback restoreState, juce::AudioProcessorValueTreeState& apvts);

        void savePreset(const String& presetName);
        void deletePreset(const String& presetName);
        void loadPreset(const String& presetName);
        int loadNextPreset();
        int loadPreviousPreset();
        StringArray getAllPresets() const;
        String getCurrentPreset() const;

        bool savePreset(const juce::File& presetFile);
        bool loadPreset(const juce::File& presetFile);

        void valueTreeRedirected(ValueTree& treeWhichHasBeenChanged) override;

        // Add setter for the callback
        void setPresetLoadedCallback(std::function<void()> cb) { presetLoadedCallback = std::move(cb); }

    private:
        AudioProcessorValueTreeState& valueTreeState;
        Value currentPreset;
        // Remove processor reference, add callbacks
        SaveStateCallback saveStateCallback;
        RestoreStateCallback restoreStateCallback;

        // Add: callback to notify when a preset is loaded
        std::function<void()> presetLoadedCallback;
    };
}