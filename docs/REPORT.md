# PictureImages Larme Debug — report card

Generated 2026-10-17 20:21 UTC from the checkpoint itself.

## Measured facts

| item | value |
|---|---|
| parameters | 46,572,544 (46.6 M) |
| layers / heads / width | 10 / 8 / 512 |
| context (tokens) | 512 |
| trained steps | 824 |
| checkpoint size | 177.8 MiB |
| LM loss on pretraining validation set | 4.8288 |
| LM perplexity on the same set | 125.06 |
| instruction tuning | 8 epochs, 824 steps |
| weights | [sft_v2.pt](https://gitopen.we-strikes.top/pictureimages/pictureimages-light-series-ai/releases/2/download/larme-debug-fp16.pt) |

## Pretraining curve (validation loss)

| step | tokens seen | val loss |
|---:|---:|---:|
| 200 | 614,400 | 5.3841 |
| 400 | 1,228,800 | 4.9592 |
| 600 | 1,843,200 | 4.6060 |
| 800 | 2,457,600 | 4.2938 |
| 1000 | 3,072,000 | 4.2358 |
| 1200 | 3,686,400 | 4.1549 |
| 1400 | 4,300,800 | 4.0870 |
| 1600 | 4,915,200 | 3.9793 |
| 1800 | 5,529,600 | 3.9862 |
| 2000 | 6,144,000 | 3.7943 |
| 2200 | 6,758,400 | 3.7411 |
| 2400 | 7,372,800 | 3.8162 |
| 2600 | 7,987,200 | 3.8019 |

## Held-out instruction set

| metric | value |
|---|---|
| pairs | 15 |
| answer loss | 4.007 |
| answer perplexity | 55.0 |
| character F1 | 0.319 |

## Series sizes (architecture only, no weights)

| config | parameters | FP16 weights |
|---|---:|---:|
| `series_7b.json` | 6,967,140,352 | 13.0 GiB |
| `series_8b.json` | 8,040,947,712 | 15.0 GiB |
| `series_9b.json` | 9,114,755,072 | 17.0 GiB |

This build carries 46,572,544 parameters — the debug baseline, not a series size.

## What this build is

This is the pipeline-validation build. It was trained and evaluated on a 4-core CPU host
with no GPU, which caps it at tens of millions of parameters. It is **not** the 7B/8B/9B
series models: those configs ship in `configs/` and require GPU hardware to train.

## Limitations, stated plainly

| limitation | detail |
|---|---|
| training tokens | about 8 M characters of pretraining text, roughly 100× less than a model this size needs |
| instruction data | a few hundred pairs; style is learned, not knowledge |
| answer quality | partially fluent; factual accuracy is not reliable |
| modalities | text only |

Read the samples below as a measurement of where the pipeline is, not as a product demo.

## Sample answers (verbatim model output)

**Q: 今天上班很累，回家怎么缓缓？**

> 天地方再加班，先别急着再轮，到后只做一这种点就意味。今天先别靠留一小时，就决定不完是了，可以把所有债券也留到晚上。下次就别反复加点回来，、走不给自己得多回到了。下次就认定有没有多少，也可以安排两次，如果有变就可以的情况。
1. 说白答应也可

**Q: 光合作用是什么？**

> 比如小音的能替代表有血型的相同，还要考虑作用和对象标识的一位置。同时应提供可以交付、相关的位置等机制，通常都会在测试中所属等条件任务；在中国大陆法基础上，为一个对象是否仍有相同的为要开支，正面的组合：在特殊形成时，氧气和城市物。合并非著作用

**Q: 写一个 Python 函数判断一个字符串是不是回文**

> 。例如 Caddy 用 carine：
1. seen reim、senter.com 、 in two_starce_samsents -
2. 合适复数是单复数 tode sterday。
3.  tort in the lost of

**Q: 为什么夏天南方比北方更闷热？**

> 关键在于：夏季节的气候工业、和物业工业编码，主要区别在“块边界”中心区”。
1. 夏季节奏比中场约1万元，抗原海洋和海岸线：钱发入地规模、地形10处，与还受季风影响，编程度为海地主要范围。
2. 下两条线：夏季风、海岸线和海水文化发展示，对

**Q: 推荐一部适合周末看的科幻电影**

> 不是。电影无论定在电影中持续重要。可以先按“真实上，因此在电影里最佳片尾字节”，是与现在的人生出现的时制作品，并不是当期。特别人放在一个身中，更常用于名词、可能文艺复兴。关键不是当地一般过去了，而是还没找到，不中心字证明自己的理由。通常是缓

