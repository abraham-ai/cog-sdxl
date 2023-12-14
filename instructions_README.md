## Exporting LoRas for use in other tools

Eden concepts are trained using the LoRA technique, a widely used extension to Stable Diffusion, and is fully compatible with the many other tools that support it. You may export your concepts as LoRas to use in other tools, such as [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) or [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

To export your concept, you can download it from the app. The concept comes as a .tar file which contains two main files: one with the token embeddings and one with the LoRA weights. Full documentation is available [here](https://docs.eden.art/docs/guides/concepts/#exporting-loras-for-use-in-other-tools)

### AUTOMATIC1111 stable-diffusion-webui

To use your concept in [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui), follow these steps:

1. Download the concept.
2. Extract (untar) the content.
3. Put the `..._lora.safetensors` file in the `stable-diffusion-webui/models/Lora` folder.
4. Put the ``..._embeddings.safetensors` file in the `stable-diffusion-webui/embeddings` folder.
5. Eden LoRAs are currently trained using the [**JuggernautXL_v6** checkpoint](https://civitai.com/models/133005/juggernaut-xl). For best results, use that same model as your base checkpoint.
6. **Make sure to load both the embedding *and* the lora weights by triggering them in your prompt**

:::tip
For "face" and "object" modes, refer to your concept directly by using *..._embeddings* in the prompt. For style concepts, you should include *"... in the style of ..._embeddings"* in your prompt.
:::tip

### ComfyUI

1. Download the concept.
2. Extract (untar) the content.
3. Put the `..._lora.safetensors` file in the `ComfyUI/models/loras` folder.
4. Put the `..._embeddings.safetensors` file in the `ComfyUI/models/embeddings` folder.
5. Eden LoRAs are currently trained using the [**JuggernautXL_v6** checkpoint](https://civitai.com/models/133005/juggernaut-xl). For best results, use that same model as your base checkpoint.
6. Load the LoRA weights using a *"Load LoRA"* node in your pipeline and adjust the strength as needed.
6. Trigger the concept in your prompt by refering to it with *embedding..._embeddings*.

:::tip
For "face" and "object" modes you refer to your concept directly by using in the prompt, for style concepts you should add *"... in the style of embedding:..._embeddings"* somewhere in your prompt.
:::tip

:::note
You may notice that the LoRA strength has a relatively small effect on the final output. This is because Eden concepts optimize towards using the token embedding to learn most of the concept, rather than the LoRA matrices.
:::note