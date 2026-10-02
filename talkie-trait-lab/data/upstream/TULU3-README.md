---
annotations_creators:
- crowdsourced
- expert-generated
- machine-generated
language:
- amh
- arb
- ary
- ars
- acq
- arz
- apc
- ben
- ceb
- dan
- deu
- ell
- eng
- eus
- fil
- fin
- fra
- gle
- guj
- hat
- hau
- hin
- hun
- ibo
- ind
- ita
- jav
- jpn
- kan
- kir
- kor
- kur
- lit
- mal
- mar
- mlg
- msa
- mya
- nep
- nld
- nso
- nya
- pan
- pes
- pol
- por
- pus
- rus
- sin
- sna
- snd
- som
- spa
- sqi
- srp
- sun
- swa
- swe
- tam
- tel
- tha
- tur
- ukr
- urd
- vie
- wol
- xho
- yor
- zho
- zul
license: odc-by
multilinguality:
- multilingual
size_categories:
- 100K<n<1M
source_datasets:
- allenai/coconot
- ai2-adapt-dev/flan_v2_converted
- HuggingFaceH4/no_robots
- OpenAssistant/oasst1
- allenai/tulu-3-personas-math
- allenai/tulu-3-sft-personas-math-grade
- allenai/tulu-3-sft-personas-code
- allenai/tulu-3-personas-algebra
- allenai/tulu-3-sft-personas-instruction-following
- AI-MO/NuminaMath-TIR
- allenai/wildguardmix
- allenai/wildjailbreak
- allenai/tulu-3-hard-coded
- CohereForAI/aya_dataset
- allenai/WildChat-1M
- LipengCS/Table-GPT
- allenai/SciRIFF
- theblackcat102/evol-codealpaca-v1
task_categories:
- other
dataset_info:
  features:
  - name: id
    dtype: string
  - name: messages
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  - name: source
    dtype: string
  splits:
  - name: train
    num_bytes: 2914250826.5647593
    num_examples: 939343
  download_size: 1412954868
  dataset_size: 2914250826.5647593
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

<img src="https://huggingface.co/datasets/allenai/blog-images/resolve/main/tulu-3/Tulu3-logo.png" alt="Tulu3 banner" width="400" style="margin-left:'auto' margin-right:'auto' display:'block'"/>

# Tulu 3 SFT Mixture

*Note that this collection is licensed under ODC-BY-1.0 license; different licenses apply to subsets of the data. Some portions of the dataset are non-commercial. We present the mixture as a research artifact.*

The Tulu 3 SFT mixture was used to train the [Tulu 3 series of models](https://huggingface.co/collections/allenai/tulu-3-models-673b8e0dc3512e30e7dc54f5).
It contains 939,344 samples from the following sets:

- [CoCoNot](https://huggingface.co/datasets/allenai/coconot) (ODC-BY-1.0), 10,983 prompts (Brahman et al., 2024)
- [FLAN v2](https://github.com/google-research/FLAN/tree/main) via [`ai2-adapt-dev/flan_v2_converted`](https://huggingface.co/datasets/ai2-adapt-dev/flan_v2_converted), 89,982 prompts (Longpre et al., 2023)
- [No Robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) (CC-BY-NC-4.0), 9,500 prompts (Rajani et al. 2023)
- [OpenAssistant Guanaco](https://huggingface.co/datasets/OpenAssistant/oasst1) (Apache 2.0), 7,132 prompts (Kopf et al., 2024)
- [Tulu 3 Persona MATH](https://huggingface.co/datasets/allenai/tulu-3-personas-math) (ODC-BY-1.0), 149,960 prompts
- [Tulu 3 Persona GSM](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-math-grade) (ODC-BY-1.0), 49,980 prompts
- [Tulu 3 Persona Python](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-code) (ODC-BY-1.0), 34,999 prompts
- [Tulu 3 Persona Algebra](https://huggingface.co/datasets/allenai/tulu-3-personas-algebra) (ODC-BY-1.0), 20,000 prompts 
- [Tulu 3 Persona IF](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following) (ODC-BY-1.0), 29,980 prompts
- [NuminaMath-TIR](https://huggingface.co/datasets/AI-MO/NuminaMath-TIR) (Apache 2.0), 64,312 prompts (Beeching et al. 2024)
- [Tulu 3 WildGuardMix](https://huggingface.co/datasets/allenai/wildguardmix) (Apache 2.0), 50,000 prompts (Han et al., 2024)
- [Tulu 3 WildJailbreak](https://huggingface.co/datasets/allenai/wildjailbreak) (ODC-BY-1.0), 50,000 prompts (Wildteaming, 2024)
- [Tulu 3 Hardcoded](https://huggingface.co/datasets/allenai/tulu-3-hard-coded) (CC-BY-4.0), 240 prompts
- [Aya](https://huggingface.co/datasets/CohereForAI/aya_dataset) (Apache 2.0), 100,000 prompts (Singh et al., 2024)
- [WildChat GPT-4](https://huggingface.co/datasets/allenai/WildChat-1M) (ODC-BY-1.0), 100,000 prompts (Zhao et al., 2024)
- [TableGPT](https://huggingface.co/datasets/LipengCS/Table-GPT) (MIT), 5,000 prompts (Zha et al., 2023)
- [SciRIFF](https://huggingface.co/datasets/allenai/SciRIFF) (ODC-BY-1.0), 10,000 prompts (Wadden et al., 2024)
- [Evol CodeAlpaca](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1) (Apache 2.0), 107,276 prompts (Luo et al., 2023)



## Dataset Structure
Each example in the dataset contains the standard instruction-tuning data points as follow:

- `id` (str): a unique identifier
- `messages` (list): message format used for supervised fine-tuning (this contains user prompt and assistant responses)
- `source` (str): the source dataset for the given sample

### Model Family

| **Stage**           | **Llama 3.1 8B**                                                                                          | **Llama 3.1 70B**                                                                                         |
|----------------------|----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| **Base Model**       | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)                                | [meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B)                              |
| **SFT**              | [allenai/Llama-3.1-Tulu-3-8B-SFT](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-SFT)                | [allenai/Llama-3.1-Tulu-3-70B-SFT](https://huggingface.co/allenai/Llama-3.1-Tulu-3-70B-SFT)              |
| **DPO**              | [allenai/Llama-3.1-Tulu-3-8B-DPO](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-DPO)                | [allenai/Llama-3.1-Tulu-3-70B-DPO](https://huggingface.co/allenai/Llama-3.1-Tulu-3-70B-DPO)              |
| **Final Models (RLVR)**     | [allenai/Llama-3.1-Tulu-3-8B](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B)                        | [allenai/Llama-3.1-Tulu-3-70B](https://huggingface.co/allenai/Llama-3.1-Tulu-3-70B)                      |
| **Reward Model (RM)**| [allenai/Llama-3.1-Tulu-3-8B-RM](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-RM)                                                     | (Same as 8B)                                                     |


## License

This dataset is licensed under ODC-BY-1.0. It is intended for research and educational use in accordance with Ai2's [Responsible Use Guidelines](https://allenai.org/responsible-use). This dataset includes output data generated from third party models that are subject to separate terms governing their use. For more information on license and terms, consult each subset linked above.

## Citation

If Tülu3 or any of the related materials were helpful to your work, please cite:
```
@article{lambert2024tulu3,
  title = {Tülu 3: Pushing Frontiers in Open Language Model Post-Training},
  author = {
    Nathan Lambert and 
    Jacob Morrison and 
    Valentina Pyatkin and 
    Shengyi Huang and 
    Hamish Ivison and 
    Faeze Brahman and 
    Lester James V. Miranda and 
    Alisa Liu and 
    Nouha Dziri and 
    Shane Lyu and 
    Yuling Gu and 
    Saumya Malik and 
    Victoria Graf and 
    Jena D. Hwang and 
    Jiangjiang Yang and
    Ronan Le Bras and
    Oyvind Tafjord and
    Chris Wilhelm and
    Luca Soldaini and 
    Noah A. Smith and 
    Yizhong Wang and 
    Pradeep Dasigi and 
    Hannaneh Hajishirzi
  },
  year = {2024},
  email = {tulu@allenai.org}
}
```