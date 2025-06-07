#include <juce_data_structures/juce_data_structures.h>
#include <juce_core/juce_core.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include "PresetManager.h"


using namespace juce;

namespace Service
{
	const File PresetManager::defaultDirectory{ File::getSpecialLocation(
		File::SpecialLocationType::commonDocumentsDirectory)
			.getChildFile(ProjectInfo::companyName)
			.getChildFile(ProjectInfo::projectName)
	};
	const String PresetManager::extension{ "preset" };
	const String PresetManager::presetNameProperty{ "presetName" };

	PresetManager::PresetManager(
	    SaveStateCallback saveState,
	    RestoreStateCallback restoreState,
	    juce::AudioProcessorValueTreeState& state)
	    : valueTreeState(state),
	      saveStateCallback(std::move(saveState)),
	      restoreStateCallback(std::move(restoreState))
	{
		// Create a default Preset Directory, if it doesn't exist
		if (!defaultDirectory.exists())
		{
			const auto result = defaultDirectory.createDirectory();
			if (result.failed())
			{
				DBG("Could not create preset directory: " + result.getErrorMessage());
				jassertfalse;
			}
		}

		valueTreeState.state.addListener(this);
		currentPreset.referTo(valueTreeState.state.getPropertyAsValue(presetNameProperty, nullptr));
	}

	void PresetManager::savePreset(const String& presetName)
	{
		if (presetName.isEmpty())
			return;

		currentPreset.setValue(presetName);
		const auto presetFile = defaultDirectory.getChildFile(presetName + "." + extension);

		juce::MemoryBlock stateData;
		if (saveStateCallback)
			saveStateCallback(stateData);
		else
			return;

		if (!presetFile.replaceWithData(stateData.getData(), stateData.getSize()))
		{
			DBG("Could not create preset file: " + presetFile.getFullPathName());
			jassertfalse;
		}
	}

	bool PresetManager::savePreset(const juce::File& presetFile)
	{
		juce::MemoryBlock stateData;
		if (saveStateCallback)
			saveStateCallback(stateData);
		else
			return false;

		if (presetFile.replaceWithData(stateData.getData(), stateData.getSize()))
			return true;

		return false;
	}

	void PresetManager::deletePreset(const String& presetName)
	{
		if (presetName.isEmpty())
			return;

		const auto presetFile = defaultDirectory.getChildFile(presetName + "." + extension);
		if (!presetFile.existsAsFile())
		{
			DBG("Preset file " + presetFile.getFullPathName() + " does not exist");
			jassertfalse;
			return;
		}
		if (!presetFile.deleteFile())
		{
			DBG("Preset file " + presetFile.getFullPathName() + " could not be deleted");
			jassertfalse;
			return;
		}
		currentPreset.setValue("");
	}

	void PresetManager::loadPreset(const String& presetName)
	{
	    if (presetName.isEmpty())
	        return;

	    const auto presetFile = defaultDirectory.getChildFile(presetName + "." + extension);
	    if (!presetFile.existsAsFile())
	    {
	        DBG("Preset file " + presetFile.getFullPathName() + " does not exist");
	        jassertfalse;
	        return;
	    }

	    // Use binary state loading via callback, not XML/ValueTree
	    juce::MemoryBlock stateData;
	    if (!presetFile.loadFileAsData(stateData))
	    {
	        DBG("Could not read preset file: " + presetFile.getFullPathName());
	        jassertfalse;
	        return;
	    }

	    if (restoreStateCallback)
	        restoreStateCallback(stateData.getData(), (int)stateData.getSize());

	    currentPreset.setValue(presetName);

	    // Notify listeners that a preset was loaded
	    if (presetLoadedCallback)
	        presetLoadedCallback();
	}

	bool PresetManager::loadPreset(const juce::File& presetFile)
	{
	    juce::MemoryBlock stateData;
	    if (!presetFile.existsAsFile() || !presetFile.loadFileAsData(stateData))
	        return false;

	    if (restoreStateCallback)
	        restoreStateCallback(stateData.getData(), (int)stateData.getSize());
	    else
	        return false;

	    // Notify listeners that a preset was loaded
	    if (presetLoadedCallback)
	        presetLoadedCallback();

	    return true;
	}

	int PresetManager::loadNextPreset()
	{
		const auto allPresets = getAllPresets();
		if (allPresets.isEmpty())
			return -1;
		const auto currentIndex = allPresets.indexOf(currentPreset.toString());
		const auto nextIndex = currentIndex + 1 > (allPresets.size() - 1) ? 0 : currentIndex + 1;
		loadPreset(allPresets.getReference(nextIndex));
		return nextIndex;
	}

	int PresetManager::loadPreviousPreset()
	{
		const auto allPresets = getAllPresets();
		if (allPresets.isEmpty())
			return -1;
		const auto currentIndex = allPresets.indexOf(currentPreset.toString());
		const auto previousIndex = currentIndex - 1 < 0 ? allPresets.size() - 1 : currentIndex - 1;
		loadPreset(allPresets.getReference(previousIndex));
		return previousIndex;
	}

	StringArray PresetManager::getAllPresets() const
	{
		StringArray presets;
		const auto fileArray = defaultDirectory.findChildFiles(
			File::TypesOfFileToFind::findFiles, false, "*." + extension);
		for (const auto& file : fileArray)
		{
			presets.add(file.getFileNameWithoutExtension());
		}
		return presets;
	}

	String PresetManager::getCurrentPreset() const
	{
		return currentPreset.toString();
	}

	void PresetManager::valueTreeRedirected(ValueTree& treeWhichHasBeenChanged)
	{
		currentPreset.referTo(treeWhichHasBeenChanged.getPropertyAsValue(presetNameProperty, nullptr));
	}
} // namespace Service
