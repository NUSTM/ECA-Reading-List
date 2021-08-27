# Emotion-Cause-Analysis
This is a reading list on the task of Emotion Cause Analysis maintained by Zixiang Ding, Fanfan Wang, and Jiaming An. Last update time: 20210827.


### Contents

* [1. New Tasks](#1-New-Tasks)
* [2. Emotion-Cause Pair Extraction (ECPE)](#2-Emotion-Cause-Pair-Extraction-(ECPE))
* [3. Emotion Cause Extraction (ECE)](#3-Emotion-Cause-Extraction-(ECE))
	* [3.1 Word-level ECE](#31-Word-level-ECE)
	* [3.2 Span-level ECE](#32-Span-level-ECE)
	* [3.3 Clause-level ECE](#33-Clause-level-ECE)
	* [3.4 Multi-level ECE](#34-Multi-level-ECE)
	* [3.5 Current/Original-subtweet-based ECE](#35-Current/Original-subtweet-based-ECE)
	* [3.6 Dataset](#36-Dataset)
* [4. Emotion-Cause Related Tasks](#4-Emotion-Cause-Related-Tasks)

## 1. New Tasks
1. Xinhong Chen, Qing Li and JianpingWang. **Conditional Causal Relationships between Emotions and Causes in Texts**. EMNLP 2020. [[paper](https://www.aclweb.org/anthology/2020.emnlp-main.252.pdf)] [[code](https://github.com/mark-xhchen/Conditional-ECPE)]
1. Hongliang Bi and Pengyuan Liu. **ECSP: A New Task for Emotion-Cause Span-Pair Extraction and Classification**. ArXiv 2020. [[paper](https://arxiv.org/pdf/2003.03507)]
1. SoujanSoujanya Poria, Navonil Majumder, Devamanyu Hazarika, Deepanway Ghosal, Rishabh Bhardwaj, Samson Yu Bai Jian, Pengfei Hong, Romila Ghosh, Abhinaba Roy, Niyati Chhaya, Alexander Gelbukh, Rada Mihalcea. **Recognizing Emotion Cause in Conversations**. Cognitive Computation 2021. [[paper](https://arxiv.org/pdf/2012.11820.pdf)] [[code](https://github.com/declare-lab/conv-emotion/tree/master/emotion-cause-extraction)]

## 2. Emotion-Cause Pair Extraction (ECPE)
2. Chuang Fan, Chaofa Yuan, Lin Gui, Yue Zhang, Ruifeng Xu. **Multi-task Sequence Tagging for Emotion-Cause Pair Extraction via Tag Distribution Refinement**. TASLP 2021. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9457144)] 
3. Aaditya Singh, Shreeshail Hingane, Saim Wani, and Ashutosh Modi. **An End-to-End Network for Emotion-Cause Pair Extraction**. EACL 2021. [[paper](https://www.aclweb.org/anthology/2021.wassa-1.9/)] [[code](https://github.com/Aaditya-Singh/E2E-ECPE)]
4. Jiaxin Yu, Wenyuan Liu, Yongjun He, Chunyue Zhang. **A Mutually Auxiliary Multitask Model With Self-Distillation for Emotion-Cause Pair Extraction**. IEEE Access 2021.[[paper](https://doi.org/10.1109/ACCESS.2021.3057880)]
5. Qixuan Sun, Yaqi Yin, Hong Yu. **A Dual-Questioning Attention Network for Emotion-Cause Pair Extraction with Context Awareness**. Arxiv 2021.[[paper](https://arxiv.org/abs/2104.07221)]
6. Zixiang Ding, Rui Xia, and Jianfei Yu. **ECPE-2D: Emotion-Cause Pair Extraction based on Joint Two-Dimensional Representation, Interaction and Prediction**. ACL 2020. [[paper](https://www.aclweb.org/anthology/2020.acl-main.288.pdf)] [[code](https://github.com/NUSTM/ECPE-2D)]
7. Penghui Wei, Jiahao Zhao, and Wenji Mao. **Effective Inter-Clause Modeling for End-to-End Emotion-Cause Pair Extraction**. ACL 2020. [[paper](https://www.aclweb.org/anthology/2020.acl-main.289.pdf)] [[code](https://github.com/Determined22/Rank-Emotion-Cause)]
8. Chuang Fan, Chaofa Yuan, Jiachen Du, Lin Gui, Min Yang, and Ruifeng Xu. **Transition-based Directed Graph Construction for Emotion-Cause Pair Extraction**. ACL 2020. [[paper](https://www.aclweb.org/anthology/2020.acl-main.342.pdf)] [[code](https://github.com/HLT-HITSZ/TransECPE)]

1. Chaofa Yuan, Chuang Fan, Jianzhu Bao, and Ruifeng Xu. **Emotion-Cause Pair Extraction as Sequence Labeling Based on A Novel Tagging Scheme**. EMNLP 2020. [[paper](https://www.aclweb.org/anthology/2020.emnlp-main.289.pdf)]
1. Zixiang Ding, Rui Xia, and Jianfei Yu. **End-to-End Emotion-Cause Pair Extraction Based on Sliding Window Multi-Label Learning**. EMNLP 2020. [[paper](https://www.aclweb.org/anthology/2020.emnlp-main.290.pdf)] [[code](https://github.com/NUSTM/ECPE-MLL)]
1. Zifeng Cheng, Zhiwei Jiang, Yafeng Yin, Hua Yu, and Qing Gu. **A Symmetric Local Search Network for Emotion-Cause Pair Extraction**. COLING 2020. [[paper](https://www.aclweb.org/anthology/2020.coling-main.12.pdf)]
1. Xinhong Chen, Qing Li, and Jianping Wang. **A Unified Sequence Labeling Model for Emotion Cause Pair Extraction**. COLING 2020. [[paper](https://www.aclweb.org/anthology/2020.coling-main.18.pdf)]
1. Ying Chen, Wenjun Hou, Shoushan Li, Caicong Wu, and Xiaoqiang Zhang. **End-to-End Emotion-Cause Pair Extraction with Graph Convolutional Network**. COLING 2020. [[paper](https://www.aclweb.org/anthology/2020.coling-main.17.pdf)]

1. Sixing Wu, Fang Chen, Fangzhao Wu, Yongfeng Huang, and Xing Li. **A Multi-Task Learning Neural Network for Emotion-Cause Pair Extraction**. ECAI 2020. [[paper](http://ecai2020.eu/papers/583_paper.pdf)]

1. Hao Tang, Donghong Ji, and Qiji Zhou. **Joint Multi-level Attentional Model for Emotion Detection and Emotion-cause Pair Extraction**. Neurocomputing 2020. [[paper](https://www.sciencedirect.com/science/article/pii/S092523122030566X)]

1. Rui Fan, Yufan Wang, Tingting He. **An End-to-End Multi-task Learning Network with Scope Controller for Emotion-Cause Pair Extraction**. NLPCC 2020. [[paper](https://doi.org/10.1007/978-3-030-60450-9_60)]

1. Haolin Song, Chen Zhang, Qiuchi Li, and Dawei Song. **End-to-end Emotion-Cause Pair Extraction via Learning to Link**. ArXiv 2020. [[paper](https://arxiv.org/pdf/2002.10710)]

18. Rui Xia and Zixiang Ding. **Emotion-Cause Pair Extraction: A New Task to Emotion Analysis in Texts**. ACL 2019. [[paper](https://arxiv.org/abs/1906.01267)] [[code](https://github.com/NUSTM/ECPE)]



### 2.1 Dataset
- ECPE Dataset [[paper](https://arxiv.org/abs/1906.01267)] [[data](https://github.com/NUSTM/ECPE/tree/master/data_combine)]
   - This dataset was constructed based on the benchmark ECE corpus (Gui et al., 2016a);
   - This dataset consists of 1,945 Chinese city news documents from SINA NEWS website.


&nbsp;
## 3. Emotion Cause Extraction (ECE)
### 3.1 Word-level ECE
1. Shuntaro Yada, Kazushi Ikeda, Keiichiro Hoashi, and Kyo Kageura. **A bootstrap method for automatic rule acquisition on emotion cause extraction**. ICDMW 2017. [[paper](https://sentic.net/sentire2017yada.pdf)]
1. Shuangyong Song and Yao Meng. **Detecting concept-level emotion cause in microblogging**. WWW 2015. [[paper](https://arxiv.org/pdf/1504.08050)]
1. Diman Ghazi, Diana Inkpen, and Stan Szpakowicz. **Detecting emotion stimuli in emotion-bearing sentences**. CICLing 2015. [[paper](http://www.site.uottawa.ca/~diana/publications/90420152.pdf)] 
1. Kai Gao, Hua Xu, and Jiushuo Wang. **Emotion cause detection for chinese micro-blogs based on ecocc model**. PAKDD 2015. [[paper](https://link.springer.com/chapter/10.1007/978-3-319-18032-8_1)]
1. Kai Gao, Hua Xu, and Jiushuo Wang. **A rule-based approach to emotion cause detection for chinese micro-blogs**. Expert Systems with Applications 2015. [[paper](https://www.sciencedirect.com/science/article/pii/S0957417415000871)]
1. Weiyuan Li and Hua Xu. **Text-based emotion classification using emotion cause extraction**. Expert Systems with Applications 2014. [[paper](https://www.sciencedirect.com/science/article/pii/S0957417413006945)]
1. Alena Neviarouskaya and Masaki Aono. **Extracting causes of emotions from text**. IJCNLP 2013. [[paper](https://www.aclweb.org/anthology/I13-1121)]
1. Sophia Yat Mei Lee, Ying Chen, Chu-Ren Huang, and Shoushan Li. **Detecting emotion causes with a linguistic rule-based approach**. Computational Intelligence 2013. [[paper](http://www.academia.edu/download/39679985/DETECTING_EMOTION_CAUSES_WITH_A_LINGUIST20151104-12559-z8qt0h.pdf)]
1. Sophia Yat Mei Lee, Ying Chen, and Chu-Ren Huang. **A text-driven rule-based system for emotion cause detection**. NAACL-HLT 2010. [[paper](https://www.aclweb.org/anthology/W/W10/W10-0206.pdf)]

### 3.2 Span-level ECE
1. Xiangju Li, Wei Gao, Shi Feng, Yifei Zhang, Daling Wang. **Boundary Detection with BERT for Span-level Emotion Cause Analysis**. ACL (Findings) 2021. [[paper](https://aclanthology.org/2021.findings-acl.60.pdf)] 
1. Elsbeth Turcan, Shuai Wang, Rishita Anubhai, Kasturi Bhattacharjee, Yaser Al-Onaizan, Smaranda Muresan. **Multi-Task Learning and Adapted Knowledge Models for Emotion-Cause Extraction**. ACL (Findings) 2021. [[paper](https://aclanthology.org/2021.findings-acl.348.pdf)] 
1. Xiangju Li, Wei Gao, Shi Feng, Wang Daling, and Shafiq Joty. **Span-Level Emotion Cause Analysis by BERT-based Graph Attention Network**. CIKM 2021. 
1. Xiangju Li, Wei Gao, Shi Feng, Wang Daling, and Shafiq Joty. **Span-level Emotion Cause Analysis with Neural Sequence Tagging**. CIKM 2021. 
1. Min Li, Hui Zhao, Hao Su, YuRong Qian and Ping Li. **Emotion-cause span extraction: a new task to emotion cause identification in texts**. Applied Intelligence 2021. [[paper](https://link.springer.com/article/10.1007/s10489-021-02188-7)]

### 3.3 Clause-level ECE
1. Hanqi Yan, Lin Gui, Gabriele Pergola, Yulan He. **Position Bias Mitigation: A Knowledge-Aware Graph Model for Emotion Cause Extraction**. ACL 2021 . [[paper](https://aclanthology.org/2021.acl-long.261.pdf)] 
1. Wenhui Yu, Chongyang Shi. **Emotion Cause Extraction by Combining Intra-clause Sentiment-enhanced Attention and Inter-clause Consistency Interaction**. ICCCS 2021. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9449281)] 
1. Guimin Hu, Guangming Lu and Yi Zhao. **FSS-GCN: A graph convolutional networks with fusion of semantic and structure for emotion cause analysis**. Knowledge-Based Systems 2021. [[paper](https://www.sciencedirect.com/science/article/pii/S0950705120307139)] [[code](https://github.com/LeMei/FSS-GCN)]
1. Yufeng Diao, Hongfei Lin, Liang Yang, Xiaochao Fan, Yonghe Chu, Di Wu, Kan Xu. **Emotion cause detection with enhanced-representation attention convolutional-context network**. Soft Comput 2021. [[paper](https://doi.org/10.1007/s00500-020-05223-w)]
1. Guimin Hu, Guangming Lu, Yi Zhao. **Emotion-Cause Joint Detection: A Unified Network with Dual Interaction for Emotion Cause Analysis**. NLPCC 2020. [[paper](https://doi.org/10.1007/978-3-030-60450-9_45)]
1. Xinglin Xiao, Lei Wang, Qingchao Kong, Wenji Mao. **Social Emotion Cause Extraction from Online Texts**. ISI 2020.[[paper](https://doi.org/10.1109/ISI49825.2020.9280532)]
1. Yufeng Diao, Hongfei Lin, Liang Yang, Xiaochao Fan, Yonghe Chu, Di Wu, Kan Xu, Bo Xu. **Multi-granularity bidirectional attention stream machine comprehension method for emotion cause extraction**. Neural Comput 2020. [[paper](https://doi.org/10.1007/s00521-019-04308-4)]
1. Jiayuan Ding, Mayank Kejriwal. **An Experimental Study of The Effects of Position Bias on Emotion Cause Extraction**. Arxiv 2020. [[paper](https://arxiv.org/abs/2007.15066)]
1. Rui Xia, Mengran Zhang, and Zixiang Ding. **RTHN: A RNN-Transformer Hierarchical Network for Emotion Cause Extraction**. IJCAI 2019. [[paper](https://arxiv.org/abs/1906.01236)] [[code](https://github.com/NUSTM/RTHN)]
1. Zixiang Ding, Huihui He, Mengran Zhang, and Rui Xia. **From Independent Prediction to Reordered Prediction: Integrating Relative Position and Global Label Information to Emotion Cause Identification**. AAAI 2019. [[paper](https://aaai.org/ojs/index.php/AAAI/article/view/4596/4474)] [[code](https://github.com/NUSTM/PAEDGL)]
1. Chuang Fan, Hongyu Yan, Jiachen Du, Lin Gui, Lidong Bing, Min Yang, Ruifeng Xu, and Ruibin Mao. **A Knowledge Regularized Hierarchical Approach for Emotion Cause Analysis**. EMNLP-IJCNLP 2019. [[paper](https://www.aclweb.org/anthology/D19-1563.pdf)]
1. Xiangju Li, Shi Feng, Daling Wang, and Yifei Zhang. **Context-aware emotion cause analysis with multi-attention-based neural network**. Knowledge-Based Systems 2019. [[paper](https://www.sciencedirect.com/science/article/pii/S0950705119301273)]
1. Xinglin Xiao, Penghui Wei, Wenji Mao, and Lei Wang. **Context-Aware Multi-View Attention Networks for Emotion Cause Extraction**. ISI 2019. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8823225)]
1. Xinyi Yu, Wenge Rong, Zhuo Zhang, Yuanxin Ouyang, and Zhang Xiong. **Multiple level hierarchical network-based clause selection for emotion cause extraction**. IEEE Access 2019. [[paper](https://ieeexplore.ieee.org/iel7/6287639/6514899/08598785.pdf)] [[code](https://github.com/deardelia/ECextraction)]
1. Bo Xu, Hongfei Lin, Yuan Lin, Yufeng Diao, Liang Yang, and Kan Xu. **Extracting emotion causes using learning to rank methods from an information retrieval perspective**. IEEE Access 2019. [[paper](https://ieeexplore.ieee.org/iel7/6287639/6514899/08625499.pdf)]
1. Jiaxing Hu, Shumin Shi, and Heyan Huang. **Combining External Sentiment Knowledge for Emotion Cause Detection**. NLPCC 2019. [[paper](http://tcci.ccf.org.cn/conference/2019/papers/345.pdf)]
1. Xiangju Li, Kaisong Song, Shi Feng, DalingWang, and Yifei Zhang. **A co-attention neural network model for emotion cause analysis with emotional context awareness**. EMNLP 2018. [[paper](https://www.aclweb.org/anthology/D18-1506)]
1. 慕永利, 李旸, 王素格. **基于 E-CNN 的情绪原因识别方法**. 中文信息学报 2018. [[paper](http://jcip.cipsc.org.cn/CN/article/downloadArticleFile.do?attachType=PDF&id=2523)]
1. 张晨, 钱涛, 姬东鸿. **基于神经网络的微博情绪识别与诱因抽取联合模型**. 计算机应用 2018. [[paper](http://www.joca.cn/CN/article/downloadArticleFile.do?attachType=PDF&id=21853)]
1. Lin Gui, Jiannan Hu, Yulan He, Ruifeng Xu, Qin Lu, and Jiachen Du. **A question answering approach to emotion cause extraction**. EMNLP 2017. [[paper](https://arxiv.org/pdf/1708.05482)]
1. Ruifeng Xu, Jiannan Hu, Qin Lu, Dongyin Wu, and Lin Gui. **An ensemble approach for emotion cause detection with event extraction and multikernel svms**. Tsinghua Science and Technology 2017. [[paper](https://ieeexplore.ieee.org/iel7/5971803/8195337/08195347.pdf)]
1. Qinghong Gao, Jiannan Hu, Ruifeng Xu, Lin Gui, Yulan He, Qin Lu, and Kam-Fai Wong. **Overview of NTCIR-13 ECA task**. NTCIR-13 2017. [[paper](https://research.nii.ac.jp/ntcir/workshop/OnlineProceedings13/pdf/ntcir/01-NTCIR13-OV-ECA-GaoQ.pdf)]
1. Xiangju Li, Shi Feng, Daling Wang, and Yifei Zhang. **Decision Tree Method for the NTCIR-13 ECA Task**. NTCIR-13 2017. [[paper](https://research.nii.ac.jp/ntcir/workshop/OnlineProceedings13/pdf/ntcir/03-NTCIR13-ECA-LiX.pdf)]
1. Han Ren, Yafeng Ren, and Jing Wan. **The GDUFS System in NTCIR-13 ECA Task**. NTCIR-13 2017. [[paper](http://research.nii.ac.jp/ntcir/workshop/OnlineProceedings13/pdf/ntcir/04-NTCIR13-ECA-RenH.pdf)]
1. Maofu Liu, Linxu Xia, Zhenlian Zhang, and Yang Fu. **WUST CRF-Based System at NTCIR-13 ECA Task**. NTCIR-13 2017. [[paper](http://research.nii.ac.jp/ntcir/workshop/OnlineProceedings13/pdf/ntcir/02-NTCIR13-ECA-LiuM.pdf)]
1. Lin Gui, Dongyin Wu, Ruifeng Xu, Qin Lu, and Yu Zhou. **Event-driven emotion cause extraction with corpus construction**. EMNLP 2016. [[paper](https://www.aclweb.org/anthology/D16-1170.pdf)]
1. Lin Gui, Ruifeng Xu, Qin Lu, Dongyin Wu, and Yu Zhou. **Emotion cause extraction, a challenging task with corpus construction**. In Chinese National Conference on Social Media Processing, 2016. [[paper](https://link.springer.com/chapter/10.1007/978-981-10-2993-6_8)]
1. Lin Gui, Li Yuan, Ruifeng Xu, Bin Liu, Qin Lu, and Yu Zhou. **Emotion cause detection with linguistic construction in chinese weibo text**. NLPCC 2014. [[paper](https://pdfs.semanticscholar.org/2da2/0f0afa18bf2dd2e89978f612e22cd1359fdb.pdf)]
1. 李逸薇, 李寿山, 黄居仁, 高伟. **基于序列标注模型的情绪原因识别方法**. 中文信息学报 2013. [[paper](https://core.ac.uk/download/pdf/78761760.pdf)]
1. Irene Russo, Tommaso Caselli, Francesco Rubino, Ester Boldrini, and Patricio Mart´ınez-Barco. **Emocause: an easy-adaptable approach to emotion cause contexts**. WASSA 2011. [[paper](https://rua.ua.es/dspace/bitstream/10045/22495/1/2011_Russo_HLT.pdf)]
1. Ying Chen, Sophia Yat Mei Lee, Shoushan Li, and Chu-Ren Huang. **Emotion cause detection with linguistic constructions**. COLING 2010. [[paper](https://www.aclweb.org/anthology/C10-1021)]

### 3.4 Multi-level ECE
1. Xiangju Li, Shi Feng, Yifei Zhang, Daling Wang. **Multi-level Emotion Cause Analysis by Multi-head Attention Based Multi-task Learning**. CCL 2021. [[paper](https://link.springer.com/chapter/10.1007%2F978-3-030-84186-7_6)] 

### 3.5 Current/Original-subtweet-based ECE
1. Ying Chen, Wenjun Hou, Xiyao Cheng, and Shoushan Li. **Joint learning for emotion classification and emotion cause detection**. EMNLP 2018. [[paper](https://www.aclweb.org/anthology/D18-1066)]
1. Ying Chen, Wenjun Hou, and Xiyao Cheng. **Hierarchical convolution neural network for emotion cause detection on microblogs**. ICANN 2018. [[paper](https://link.springer.com/chapter/10.1007/978-3-030-01418-6_12)]
1. Xiyao Cheng, Ying Chen, and Bixiao Cheng. **An Emotion Cause Corpus for Chinese Microblogs with Multiple-User Structures**. TALLIP 2017. [[paper](http://ir.nsfc.gov.cn/paperDownload/ZD4360642.pdf)]

### 3.6 Dataset
| Dataset | Emotion Categories | Annotated Roles | Size | Description | Source |
| ----- | ----- | ----- | ----- | ----- | ----- |
| Chinese EC Corpus | 5 <br />Turner: {happiness, sadness, fear, anger, surprise} | emotion cue, cause | 5,964 | Each **Chinese** instance contains 3 sentences extracted from the Sinica Corpus and 3,460 instances are annotated with emotion cause. | Lee et al., 2010b<br />[[paper](https://www.researchgate.net/profile/Chu-Ren_Huang/publication/220746716_Emotion_Cause_Events_Corpus_Construction_and_Analysis/links/0912f508ff080541ac000000/Emotion-Cause-Events-Corpus-Construction-and-Analysis.pdf)] |
| It-EmoContext | 7<br />Ekman: {happiness, sadness, fear, anger, surprise, disgust} + {underspecified} | emotion cue, cause | 6,000 | Each instance contains 3 sentences which are extracted from the La Repubblica Corpus based on the **Italian** emotion keywords. | Russo et al., 2011<br />[[paper](https://rua.ua.es/dspace/bitstream/10045/22495/1/2011_Russo_HLT.pdf)] |
| Corpus of Emotion Causes | 22 | emotion cue, experiencer, cause, polarity of cause events, linguistic relation between EC | 532 | The dataset contains 22 sentences from (Ortony et al., 1988) for 22 emotion type and 510 sentences with emotion tokens and explicitly mentioned causes from online ABBYY Lingvo dictionary. | Neviarouskaya and Aono, 2013<br />[[paper](https://www.aclweb.org/anthology/I13-1121)]|
| Weibo EC Corpus 1 | 7<br />Ekman + {like} | emotion cue, cause | 1,333 | This is the first corpus for ECE from **Chinese** Weibo text based on NLPCC13 dataset. | Gui et al., 2014<br />[[paper](https://pdfs.semanticscholar.org/2da2/0f0afa18bf2dd2e89978f612e22cd1359fdb.pdf)] |
| Weibo EC Corpus 2 | 6<br />Ekman | emotion cue, cause | 16,485 | The dataset contains 16,485 **Chinese** Weibo posts, only 1,305 of which are labeled with emotion causes. | Li and Xu, 2014<br />[[paper](https://www.sciencedirect.com/science/article/pii/S0957417413006945)]|
| Electoral-Tweets | 8<br />Plutchik: {happiness, sadness, fear, anger, surprise, disgust, trust, anticipation} | emotion stimulus, experiencer | 4,058 | This dataset consists of annotated tweets pertaining to the 2012 US presidential elections. |Mohammad et al., 2014<br />[[paper](https://www.aclweb.org/anthology/W14-2607.pdf)]<br />[[data](http://www.purl.org/net/PoliticalTweets2012)] |
| Emotion-Stimulus | 7<br />Ekman + {shame} | emotion cause | 2,414 | This dataset consists of 820 sentences which are annotated with emotions and causes, and 1,594 sentences which are marked only with emotion. | Ghazi et al., 2015<br />[[paper](http://www.site.uottawa.ca/~diana/publications/90420152.pdf)]<br />[[data](http://www.site.uottawa.ca/~diana/resources/emotion_stimulus_data/)] |
| SINA-City-News<br />(EC benchmark corpus) | 6<br />Ekman | emotion cue, cause | 2,105 | The dataset consists of **Chinese** city news from SINA NEWS website. Each instance contains only one emotion keyword and at least one emotion cause. | Gui et al., 2016a<br />[[paper](https://www.aclweb.org/anthology/D16-1170.pdf)]<br />[[data](http://119.23.18.63/?page%20id=694)] |
| Weibo EC Corpus 3 | 4<br />{joy, angry, sad, fearful} | emotion cue, cause, experiencer | 4,000 | This dataset contains 7,000 tweets sampled from **Chinese** Sina microblog and their 10,700 subtweets, many of which have the multiple-user structure. | Cheng et al., 2017<br />[[paper](http://ir.nsfc.gov.cn/paperDownload/ZD4360642.pdf)]<br />[[data](https://github.com/Wenjun-Hou/Weibo-Emotion-Corpus)] |
| Japanese EC Corpora | / | emotion word, cause, cue phrase between emotion and cause | 1,000 * 3 | Three datasets containing 1,000 sentences with emotion words were sampled from **Japanese** newspaper articles, web news articles and Q&A site texts. | Yada et al., 2017<br />[[paper](https://sentic.net/sentire2017yada.pdf)] |
| NTCIR-13_ECA Corpora<br />(Chines city news /<br />English novel) | 6<br />Ekman | emotion cue, cause | 2,619/<br />2,403 | The two corpora are constructed from **Chinese** SINA news and English novel text respectively, in which each instance contains multiple clauses. | Gao et al., 2017<br />[[paper](https://research.nii.ac.jp/ntcir/workshop/OnlineProceedings13/pdf/ntcir/01-NTCIR13-OV-ECA-GaoQ.pdf)]<br />[[data](http://119.23.18.63/ECA.html)] |
| Weibo EC Corpus 4 | 6<br />Ekman | emotion cause | 6,771 | The **Chinese** posts all contain emojis which are labeled with emotions and corresponding clause-level emotion causes are also annotated. | 张晨 et al., 2018<br />[[paper](http://www.joca.cn/CN/article/downloadArticleFile.do?attachType=PDF&id=21853)] |
| REMAN | 9<br />Plutchik + {other} | emotion cue, cause, experiencer, target | 1,720 | Each instance contains consecutive triples of sentences which are sampled from literature (200 books from Project Gutenberg). | Kim and Klinger, 2018<br />[[paper](https://www.aclweb.org/anthology/C18-1114.pdf)]<br />[[data](http://www.ims.uni-stuttgart.de/data/reman)] |
| GoodNewsEveryone | 15<br />extended Plutchik: {anger, annoyance, disgust, fear, guilt, joy, love, pessimism, negative surprise, optimism, positive surprise, pride, sadness, shame, trust} | emotion cue, cause, experiencer, target, intensity, reader emotion | 5,000 | The dataset consists of annotated English news headlines. | Bostan et al., 2019<br />[[paper](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.194.pdf)]<br />[[data](http://www.romanklinger.de/data-sets/GoodNewsEveryone.zip)] |

## 4. Emotion-Cause Related Tasks
 (这个题目是不是有点大）
1. Yanran Li, Ke Li, Hongke Ning, Xiaoqiang Xia, Yalong Guo, Chen Wei, Jianwei Cui, Bin Wang. **Towards an Online Empathetic Chatbot with Emotion Causes**. SIGIR 2021. [[paper](https://dl.acm.org/doi/pdf/10.1145/3404835.3463042)] 
