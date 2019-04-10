# hw2
1.**Trajectory**,表示一個**episode**中所有的**observation**和**action**

2.基本任務則是尋找到一種**agent**的行為模式**(policy)**使得**reward**最大化,**policy**是一個網路所以我們要找到最佳的參數使得**reward**最大

3.**reinforce**算法是使用N次採樣來近似期望值,所以PG的算法實際上是採樣**reward**最大的動作作為**label**,然後**policy network**對這動作做**supervised learning**

4.因為回歸任務多個完美的**policy**,可能都能使得**reward**最大,所以以分類問題來看這變成單一物件卻多類別的意思,所以造成強化學習不穩定且**high variance**,因為網絡不知道應該最大化哪個類別的輸出概率

5.降低**high variance**
* reward to go : 
