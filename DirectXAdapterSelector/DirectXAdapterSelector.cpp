// dllmain.cpp : Defines the entry point for the DLL application.
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <vector>

#pragma comment(lib, "dxguid.lib")

// Failure really shouldn't occur, but if it does then returning 0 is 
// safe because it's the default adapter ID.
#define RETURN_IF_FAILED(hr) if (FAILED((hr))) { return 0; }

using Microsoft::WRL::ComPtr;

class DxgiModule
{
    using CreateFactoryFunc = decltype(CreateDXGIFactory);
    HMODULE m_module = nullptr;
    CreateFactoryFunc* m_createFactoryFunc = nullptr;

public:
    DxgiModule()
    {
        m_module = LoadLibraryA("dxgi.dll");
        if (m_module)
        {
            auto funcAddr = GetProcAddress(m_module, "CreateDXGIFactory");
            if (funcAddr)
            {
                m_createFactoryFunc = reinterpret_cast<CreateFactoryFunc*>(funcAddr);
            }
        }
    }
    ~DxgiModule() { if (m_module) { FreeModule(m_module); } }

    ComPtr<IDXGIFactory6> CreateFactory()
    {
        ComPtr<IDXGIFactory6> adapterFactory;
        m_createFactoryFunc(IID_PPV_ARGS(&adapterFactory));
        return adapterFactory;
    }
};

extern "C" __declspec(dllexport) int __cdecl SelectDirectXAdapterId(bool preferHighPerformance)
{
    DxgiModule dxgi;

    ComPtr<IDXGIFactory6> factory = dxgi.CreateFactory();;
    if (!factory)
    {
        return 0;
    }

    ComPtr<IDXGIAdapter1> adapter;

    // Store LUIDs for hardware adapters in original unsorted order.
    std::vector<std::pair<int, LUID>> adaptersUnsortedIndexAndLuid;
    for (int i = 0; factory->EnumAdapters1(i, adapter.ReleaseAndGetAddressOf()) != DXGI_ERROR_NOT_FOUND; i++)
    {
        DXGI_ADAPTER_DESC desc = {};
        RETURN_IF_FAILED(adapter->GetDesc(&desc));
        adaptersUnsortedIndexAndLuid.emplace_back(i, desc.AdapterLuid);
    }

    // Find the first adapter meeting GPU preference.
    DXGI_ADAPTER_DESC preferredAdapterDesc = {};
    {
        DXGI_GPU_PREFERENCE gpuPreference = preferHighPerformance ? DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE : DXGI_GPU_PREFERENCE_MINIMUM_POWER;
        RETURN_IF_FAILED(factory->EnumAdapterByGpuPreference(0, gpuPreference, IID_PPV_ARGS(adapter.ReleaseAndGetAddressOf())));
        RETURN_IF_FAILED(adapter->GetDesc(&preferredAdapterDesc));
    }

    // Look up the index of the preferred adapter in the unsorted list order.
    for (auto& hardwareAdapterEntry : adaptersUnsortedIndexAndLuid)
    {
        if (hardwareAdapterEntry.second.HighPart == preferredAdapterDesc.AdapterLuid.HighPart &&
            hardwareAdapterEntry.second.LowPart == preferredAdapterDesc.AdapterLuid.LowPart)
        {
            return hardwareAdapterEntry.first;
        }
    }

    return 0;
}