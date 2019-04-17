# HW3  
## Actor-Critic

Actor-Critic(TD tempral diff) 算法與REINFORCE (with baseline)（MC 蒙地卡羅）算法最大的不同在於，REINFORCE (with baseline)算法擬合value function 的方式是透過採樣，但是沒有採用到value function間的關係。而Actor-Critic 算法使用了value function的迭代性還有baseline，代表是基於舊的value function來更新value function。

MC算法的variance較大，TD算法的variance較小，但TD的value function估計若不準也會迭代影響，所以個偶優缺點，使用TD算法較多。

### 總結
以下是比較的結果，大迭代100小迭代1取得了最好的表現。
![](https://github.com/citya1472581234/reinforcement-learning/blob/master/cs294-homework/hw3/result.png?raw=true) 

在網路上看到的資料是lb_rtg_na花費的時間會比較長，但我自己實測是較短，還需要研究一下為何會有這種情況。
![](https://github.com/citya1472581234/reinforcement-learning/blob/master/cs294-homework/hw3/Time_walker2d.png?raw=true) 

---------------
##  Q-learning
簡易的說 Actor-Critic 是採用TD去算出value function，Advantage Actor Critic則稱為(A2C)或者REINFORCE算法都是Policy Gradients 的方法(迭代或採樣出value function在套入Policy Gradients)，而Q-learning是利用Bellman的迭代式使得現在的Q function逼近未來Q function(迭代產生的)（計算現在的Q function和未來Q function之間的差值產生loss），不需要Policy Gradients。

* dueling Q 只有更改network架構，改為state-value 和 the advantages for each actions分別輸出

* noise net 增加noise 可以使得結果更好

* distributional Q function :也不一定要學習期望值，我們可以學習reward的分佈，會較為準確。（因為期望值相同，分佈可能差很多）

* double Q : 將target Q function 得到的action，再次輸入得到新的Q function，可以解決Q learning 估計錯誤的情形。

* replay buffer: 在這個問題中，之所以加入experience replay是因為樣本是從遊戲中的連續幀獲得的，這與簡單的reinforcement learning問題（比如maze）相比，樣本的關聯性大了很多，如果沒有experience replay，算法在連續一段時間內基本朝著同一個方向做gradient descent，那麼同樣的步長下這樣直接計算gradient就有可能不收斂。因此experience replay是從一個memory pool中隨機選取了一些expeirence，然後再求梯度，從而避免了這個問題。


