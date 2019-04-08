# CS294-112 HW 1: Imitation Learning

Dependencies:
 * Python **3.6.8**
 * TensorFlow version **1.13.1**
 * MuJoCo version **2.0**  mujoco_py**2.0.2.0**
 * OpenAI Gym version **0.12.1**
 
The only file that you need to look at is `run_expert.py`, which is code to load up an expert policy, run a specified number of roll-outs, and save out data.
run_expert.py 利用給定的 expert.pkl 去獲得 expert_data(專家的資料) 來做 **behavior** **cloning** 

In `experts/`, the provided expert policies are:
* Ant-v2.pkl
* HalfCheetah-v2.pkl
* Hopper-v2.pkl
* Humanoid-v2.pkl
* Reacher-v2.pkl
* Walker2d-v2.pkl

The name of the pickle file corresponds to the name of the gym environment.

Imitation Learning ，分了兩個分支：Behavioral Cloning (BC) 和 Data Aggregation (DAgger)。

**behavior** **cloning** 
觀察專家的行為產生data,然後做監督式學習
**data** **aggregation** 
原本BC只有參照專家產生data,所以只會模仿專家行為無法舉一反三,因為參照專家產生的data是做sample,不可能全部模仿.利用aggregation,在每次iteration迭代過程中在將觀察到的observation,給專家網路做action,(obs,act)當作新的data,來擴增資料可以增加資料的可靠度.

以下是使用HalfCheetah環境的測試

env :HalfCheetah| BC | DAGGER | Expert
--------------|:-----:|-----:| ----:
Mean    | 382 |  1952 | 4172   | 
Std    | 602 |  35 |  77 | 


