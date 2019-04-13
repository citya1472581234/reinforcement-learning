#HW3 Actor-Critic

Actor-Critic 算法與REINFORCE (with baseline)算法最大的不同在於，REINFORCE (with baseline)算法擬合value function 的方式是透過採樣，但是沒有採用到value function間的關係。而Actor-Critic 算法使用了value function的迭代性，代表是基於就的value function來更新value function。
