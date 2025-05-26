---
license: cc-by-4.0
task_categories:
- text-generation
- text2text-generation
language:
- en
- zh
size_categories:
- 10K<n<100K
pretty_name: conversa
---


# ChatHaruhi

# Reviving Anime Character in Reality via Large Language Model

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)]()
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)]()

github repo: https://github.com/LC1332/Chat-Haruhi-Suzumiya



**Chat-Haruhi-Suzumiya**is a language model that imitates the tone, personality and storylines of characters like Haruhi Suzumiya,


<details>
  <summary> The project was developed by Cheng Li, Ziang Leng, Chenxi Yan, Xiaoyang Feng, HaoSheng Wang, Junyi Shen, Hao Wang, Weishi Mi, Aria Fei, Song Yan, Linkang Zhan, Yaokai Jia, Pingyu Wu, and Haozhen Sun,etc. </summary>

This is an open source project and the members were recruited from open source communities like DataWhale.

Lulu Li( [Cheng Li@SenseTime](https://github.com/LC1332) )initiated the whole project and designed and implemented most of the features.
 
Ziang Leng( [Ziang Leng@SenseTime](https://blairleng.github.io) )designed and implemented the training, data generation and backend architecture for ChatHaruhi 1.0.

Chenxi Yan( [Chenxi Yan@Chengdu University of Information Technology](https://github.com/todochenxi) )implemented and maintained the backend for ChatHaruhi 1.0.

Junyi Shen( [Junyi Shen@Zhejiang University](https://github.com/J1shen) )implemented the training code and participated in generating the training dataset.

Hao Wang( [Hao Wang](https://github.com/wanghao07456) )collected script data for a TV series and participated in data augmentation.

Weishi Mi( [Weishi MI@Tsinghua University](https://github.com/hhhwmws0117) )participated in data augmentation.
 
Aria Fei( [Aria Fei@BJUT](https://ariafyy.github.io/) )implemented the ASR feature for the script tool and participated in the Openness-Aware Personality paper project.

Xiaoyang Feng( [Xiaoyang Feng@Nanjing Agricultural University](https://github.com/fengyunzaidushi) )integrated the script recognition tool and participated in the Openness-Aware Personality paper project.

Yue Leng ( [Song Yan](https://github.com/zealot52099) )Collected data from The Big Bang Theory. Implemented script format conversion.

scixing(HaoSheng Wang)( [HaoSheng Wang](https://github.com/ssccinng) ) implemented voiceprint recognition in the script tool and tts-vits speech synthesis.

Linkang Zhan( [JunityZhan@Case Western Reserve University](https://github.com/JunityZhan) ) collected Genshin Impact's system prompts and story data.

Yaokai Jia( [Yaokai Jia](https://github.com/KaiJiaBrother) )implemented the Vue frontend and practiced GPU extraction of Bert in a psychology project.

Pingyu Wu( [Pingyu Wu@Juncai Shuyun](https://github.com/wpydcr) )helped deploy the first version of the training code. 

Haozhen Sun( [Haozhen Sun@Tianjin University] )plot the character figures for ChatHaruhi. 



</details>

## transfer into input-target format

If you want to convert this data into an input-output format

check the link here

https://huggingface.co/datasets/silk-road/ChatHaruhi-Expand-118K


### Citation

Please cite the repo if you use the data or code in this repo.
```
@misc{li2023chatharuhi,
      title={ChatHaruhi: Reviving Anime Character in Reality via Large Language Model}, 
      author={Cheng Li and Ziang Leng and Chenxi Yan and Junyi Shen and Hao Wang and Weishi MI and Yaying Fei and Xiaoyang Feng and Song Yan and HaoSheng Wang and Linkang Zhan and Yaokai Jia and Pingyu Wu and Haozhen Sun},
      year={2023},
      eprint={2308.09597},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```