
# Stable Diffusion v1.5 Olive Optimized

1. To get started install the python requirements.

    ``` python
    pip install -r Requierements.txt
    ```

2. You will now need a HuggingFace token please go here to retrieve it [Tokens](https://huggingface.co/settings/tokens)
3. Login with huggingface CLI

    ``` python
    huggingface-cli login
    ```
    
4. Once downloading the models from HuggingFace follow the optimization tutorial with [Olive](https://github.com/microsoft/Olive/tree/main/examples/directml/stable_diffusion)

5. After optimizing the model using Olive copy the outputs of the optimization to .\Onnx\fp16\ directory for the build to pick them up.

6. Open the solution and build the project.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
