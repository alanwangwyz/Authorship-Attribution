# Authorship-Attribution
https://github.com/analyticascent/stylext/blob/master/Stylometry%20Capstone.pdf
http://cs229.stanford.edu/proj2017/final-reports/5241953.pdf
http://cs229.stanford.edu/proj2012/CastroLindauer-AuthorIdentificationOnTwitter.pdf
https://github.com/lindauer/twitauth
https://www.kaggle.com/srkirkland/author-identification-with-tensorflow

-- SML
52 个有用的机器学习与预测接口盘点
https://www.infoq.cn/article/52-useful-machine-learning-prediction-apis

-- Project
利用机器学习算法进行特朗普twitter的主题分析：
這篇的思路可以當作project參考。
https://blog.csdn.net/Aaronji1222/article/details/78153269

Context
Scrapped from twitters from 2016-01-01 to 2019-03-29, Collecting Tweets containing Bitcoin or BTC
Content
User, fullname, tweet-id,timestamp, url, likes,replies,retweets, text, html
https://www.kaggle.com/alaix14/bitcoin-tweets-20160101-to-20190329/downloads/bitcoin-tweets-20160101-to-20190329.zip/1

推特情感分析归类https://pdfs.semanticscholar.org/f7e5/18c99a0ffd997e6d7b385dda1675c3f3ef3c.pdf
Pandas使用入门指南：https://codingpy.com/article/a-quick-intro-to-pandas/
spaCy的主要操作：https://www.jianshu.com/p/74e6c5376bc0
Scikit-learn：https://www.jianshu.com/p/e0844e7cdba5

中文定制的小范本文本分类：https://blog.csdn.net/qq_28626909/article/details/80382029
---HINT 可以想想都对与文本直接进行x朴素贝叶斯等分类的方法并不适合这个模型。而且大多数基于naive bayes以及random forest的分类都是判别模型，即按0-1，是否分类，不适用于这种。可能得自己
按提取的关键词来按照频次来判断（目前收获不一定正确，让我在看看，如果不对的话就无视了哈）
