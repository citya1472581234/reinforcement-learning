#HW3 Actor-Critic

Actor-Critic 算法與REINFORCE (with baseline)算法最大的不同在於，REINFORCE (with baseline)算法擬合value function 的方式是透過採樣，但是沒有採用到value function間的關係。而Actor-Critic 算法使用了value function的迭代性，代表是基於就的value function來更新value function。

###總結
以下是比較的結果，大迭代100小迭代1取得了最好的表現。
![](https://github.com/citya1472581234/reinforcement-learning/blob/master/cs294-homework/hw3/result.png?raw=true) 

在網路上看到的資料是lb_rtg_na花費的時間會比較長，但我自己實測是較短，還需要研究一下為何會有這種情況。
![](https://github.com/citya1472581234/reinforcement-learning/blob/master/cs294-homework/hw3/Time_walker2d.png?raw=true) 

