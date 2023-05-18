#pragma once

// Returns the index of the first DirectX adapter detected that meetings either high performance or 
// minimum power preference. This function is useful when creating the DirectML EP to select a preferred 
// type of GPU on multi-GPU devices (e.g. laptop with integrated and discrete).
extern "C" __declspec(dllexport) int __cdecl SelectDirectXAdapterId(bool preferHighPerformance);