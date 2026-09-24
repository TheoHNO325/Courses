---
license: cc-by-nc-sa-4.0
language:
- zh

extra_gated_prompt: "
## 📜  DongbaMIE's Terms of Use: 

**1. The copyright of the DongbaMIE dataset is owned by the Key Laboratory of Ethnic Language Intelligent Analysis and Security Governance of MOE at Minzu University of China.**

**2. All samples of DongbaMIE dataset are provided exclusively to designated applicants for research purposes only. The resources must not be disclosed or disseminated to third parties in any form. None of the samples may be used for any commercial purposes.**

**3. Appropriate acknowledgment must be given to the Key Laboratory of Ethnic Language Intelligent Analysis and Security Governance of MOE at Minzu University of China in all instances where this dataset is used.**

## ⚠️ Access Verification Required

**In order to verify your identity and ensure responsible use of the DongbaMIE dataset, please send a confirmation email from your institutional account (e.g., ending with `.edu`, `.ac.cn`, or your organization’s domain) to:**

📧   **thinklis@hrbeu.edu.cn**

**Subject: `DongbaMIE Dataset Access Request`**

**In the body, please include:**

*- Your Hugging Face username*

*- The organization you are affiliated with*

*- A brief description of your intended use*

❗ **Access requests without email verification may not be approved.**
"



extra_gated_fields:
  Full Name: text
  Email Address (⚠️ must be a valid institutional email): text
  Company: text
  Job Title: text
  Country: country
  Specific date: date_picker
  I want to use this dataset for:
    type: select
    options: 
      - Research
  I agree to use the DongbaMIE dataset for academic research under the terms of use, and I am willing to bear the corresponding responsibilities if I violate them: checkbox
---

## Overview

This is the dataset from the paper "**DongbaMIE: A Multimodal Information Extraction Dataset for Evaluating Semantic Understanding of Dongba Pictograms**" ([arxiv](https://arxiv.org/abs/2503.03644))

Please refer to this [github](https://github.com/thinklis/DongbaMIE) repo **for details.**


## Citation
If you find our data useful, please consider citing:
```
@inproceedings{bi-etal-2025-dongbamie,
    title = "{D}ongba{MIE}: A Multimodal Information Extraction Dataset for Evaluating Semantic Understanding of Dongba Pictograms",
    author = "Bi, Xiaojun  and
      Li, Shuo  and
      Xing, Junyao  and
      Wang, Ziyue  and
      Luo, Fuwen  and
      Qiao, Weizheng  and
      Han, Lu  and
      Sun, Ziwei  and
      Li, Peng  and
      Liu, Yang",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-emnlp.51/",
    doi = "10.18653/v1/2025.findings-emnlp.51",
    pages = "976--990",
    ISBN = "979-8-89176-335-7"
}
```