# hw2
1.**Trajectory**,表示一個**episode**中所有的**observation**和**action**

2.基本任務則是尋找到一種**agent**的行為模式**(policy)**使得**reward**最大化,**policy**是一個網路所以我們要找到最佳的參數使得**reward**最大

3.**reinforce**算法是使用N次採樣來近似期望值,所以PG的算法實際上是採樣**reward**最大的動作作為**label**,然後**policy network**對這動作做**supervised learning**

4.因為回歸任務多個完美的**policy**,可能都能使得**reward**最大,所以以分類問題來看這變成單一物件卻多類別的意思,所以造成強化學習不穩定且**high variance**,因為網絡不知道應該最大化哪個類別的輸出概率

5.降低**high variance**
> -   reward to go : 因為原式會考慮t時刻動作會與之前的reward有關,所以直觀的將 它修改,數學上也證明是可行的
* Baseline : 使用baseline,可以減少方差有數學的證明,以直觀的解釋為在進行採樣時可能沒有辦法每個動作都採樣到,造成該動作的機率降低,所以增加baseline可以減少有一些action沒被考慮的可能性

6.增加 Discount Factor 
> * Discount Factor為小於1的數,用Discount的t次方,保證Q函數的收歛性,讓他不會無窮大
* 鼓勵agent 盡快完成任務
* 用Discount的t次方,代表時間越近,就影響越大

