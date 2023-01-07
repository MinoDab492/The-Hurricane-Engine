// Copyright Epic Games, Inc. All Rights Reserved.

#include "WaveformEditorInstantiator.h"

#include "ContentBrowserModule.h"
#include "Framework/MultiBox/MultiBoxBuilder.h"
#include "Sound/SoundWave.h"
#include "WaveformEditor.h"
#include "WaveformEditorLog.h"

#define LOCTEXT_NAMESPACE "WaveformEditorInstantiator"

void FWaveformEditorInstantiator::ExtendContentBrowserSelectionMenu()
{
	FContentBrowserModule& ContentBrowserModule = FModuleManager::GetModuleChecked<FContentBrowserModule>(TEXT("ContentBrowser"));

	TArray<FContentBrowserMenuExtender_SelectedAssets>& ContentBrowserExtenders = ContentBrowserModule.GetAllAssetViewContextMenuExtenders();
	ContentBrowserExtenders.Add(FContentBrowserMenuExtender_SelectedAssets::CreateSP(this, &FWaveformEditorInstantiator::OnExtendContentBrowserAssetSelectionMenu));
}

TSharedRef<FExtender> FWaveformEditorInstantiator::OnExtendContentBrowserAssetSelectionMenu(const TArray<FAssetData>& SelectedAssets)
{
	TSharedRef<FExtender> Extender = MakeShared<FExtender>();

	Extender->AddMenuExtension(
		"GetAssetActions",
		EExtensionHook::After,
		nullptr,
		FMenuExtensionDelegate::CreateSP(this, &FWaveformEditorInstantiator::AddWaveformEditorMenuEntry, SelectedAssets)
	);

	return Extender;
}

void FWaveformEditorInstantiator::AddWaveformEditorMenuEntry(FMenuBuilder& MenuBuilder, TArray<FAssetData> SelectedAssets)
{
	if (SelectedAssets.Num() > 0)
	{
		// check that all selected assets are USoundWaves.
		TArray<USoundWave*> SelectedSoundWaves;
		for (const FAssetData& SelectedAsset : SelectedAssets)
		{
			if (SelectedAsset.GetClass() != USoundWave::StaticClass())
			{
				return;
			}

			SelectedSoundWaves.Add(static_cast<USoundWave*>(SelectedAsset.GetAsset()));
		}

		MenuBuilder.AddMenuEntry(
			LOCTEXT("SoundWave_WaveformEditor", "Edit Waveform"),
			LOCTEXT("SoundWave_WaveformEditor_Tooltip", "Open waveform editor"),
			FSlateIcon(),
			FUIAction(
				FExecuteAction::CreateSP(this, &FWaveformEditorInstantiator::CreateWaveformEditor, SelectedSoundWaves),
				FCanExecuteAction()
			)
		);
	}
}

bool FWaveformEditorInstantiator::CanSoundWaveBeOpenedInEditor(const USoundWave* SoundWaveToEdit)
{
	if (SoundWaveToEdit == nullptr)
	{
		UE_LOG(LogWaveformEditor, Warning, TEXT("Could not open waveform editor from null SoundWave"))
		return false;
	}
	
	if (SoundWaveToEdit->GetDuration() == 0.f)
	{
		UE_LOG(LogWaveformEditor, Warning, TEXT("Could not open waveform editor for soundwave %s, duration is 0"), *(SoundWaveToEdit->GetName()))
		return false;
	}
	else if (SoundWaveToEdit->NumChannels == 0.f)
	{
		UE_LOG(LogWaveformEditor, Warning, TEXT("Could not open waveform editor for soundwave %s, channel count is 0"), *(SoundWaveToEdit->GetName()))
		return false;
	}
		
	return true;
}

void FWaveformEditorInstantiator::CreateWaveformEditor(TArray<USoundWave*> SoundWavesToEdit)
{
	for (USoundWave* SoundWavePtr : SoundWavesToEdit)
	{
		if (CanSoundWaveBeOpenedInEditor(SoundWavePtr))
		{
			TSharedRef<FWaveformEditor> WaveformEditor = MakeShared<FWaveformEditor>();

			if (!WaveformEditor->Init(EToolkitMode::Standalone, nullptr, SoundWavePtr))
			{
				UE_LOG(LogWaveformEditor, Warning, TEXT("Could not open waveform editor for soundwave %s, initialization failed"), *(SoundWavePtr->GetName()))
				WaveformEditor->CloseWindow();
			}
		}
	}
}

#undef LOCTEXT_NAMESPACE