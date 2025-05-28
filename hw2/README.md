코드 구현이 쉽지 않았다.
처음에 접근을 Imitation Learning에서 그랬듯 Supervised Learning에서 사용하는 출력층 활성화 함수를 사용했는데,
이게 문제였는지 제대로 learning이 수행되지 않았다.
결국 다른 사람 코드를 참고하였다.
참고한 코드에선 sampling 데이터를 확률 분포 내에서 랜덤하게 추출한 값을 이용했다.
이는 강화학습 특성상, 다양한 환경에 직면하여 어떤 상황에서 어떤 행동을 했을때 효율적인지 추출하기 위한 것이다.

처음 확률 분포에 따른 랜덤한 sampling이 아닌 결정론적인 sampling을 이용했을때 return의 결과가 하락하는 결과가 나왔는데,
추측으론, hw2에서 구현한 policy gredient 특성상 return의 결과에 따라 정도가 달라질 뿐이지 sampling에서의 행동이 좋은 결과를 초래하든 나쁜 결과를 초래하든 해당 행동을 할 확률이 조금씩 증가하기 때문일 것이라 생각한다.
(그에대한 반증으로, nomalize 적용 시 학습이 정상적으로 수행되는것을 확인하였다.)

# section 3
CartPole을 오래 유지하도록 강화학습을 수행한다.
해당 실험에선 각 모델마다 'batch size', 'reward to go', 'nomalize'를 변수로 두었다.

통상 trajectory의 길이인 batch size가 증가하면, 모델의 예측 정확도는 상승하되 편차가 증가한다.

reward to go는 강화학습 시나리오 처럼, 학습이 한 시간축에 의한 맥락적 상황에 대해 수행될때 현재의 행동이 과거의 보상에 영향을 미치지 않도록 조절하는 것이다. 이를 적용할 경우 편차가 줄어든다.

nomalize는 학습 데이터가 [-1, 1]의 값을 가지고 평균이 0이되도록 조절하여 학습 성능을 향상시키는 것을 의미한다.

참고로 policy gredient에서는 high variance로 학습이 불안정한 문제가 존재해, 이를 해결하기 위한 다양한 방법이 제안되었다.

## batch size 1000
 
betch 사이즈는 1000으로 하고, 'reward to go', 'nomalize' 여부에 따라 총 4가지 경우에 대해 실험을 진행하였다.

`python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole`

`python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name cartpole_rtg`
 
 `python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -na --exp_name cartpole_na`
 
 `python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -na --exp_name cartpole_rtg_na `

 아래의 그래프에서 
 
 black line은 'reward to go', 'nomalize'를 모두 적용하지 않은 버전,

 pink line은 'nomalize' 만을 적용한 버전,

 blue line은 'reward to go' 만을 적용한 버전,

 yellow는 'reward to go', 'nomalize'를 모두 적용한 버전이다.

 아래의 그림은 평균 Reward를 나타낸 그래프이다.
 nomalize를 적용한 pink line과 yellow line에서 뛰어난 학습 정도를 보여줌을 확인 할 수 있다.

![Image](https://github.com/user-attachments/assets/5a0e6f91-ef45-47c5-bc88-d82f309bfe34)

다음으로 분산을 비교한 것이다.
예상과 다르게 이론적으로 분산 문제를 해결하는 reward to go(blue line, yellow line)이 적지 않은 분산을 보이긴 하지만,
전반적으로 보았을때 아무것도 적용하지 않은 black line보다 낮은 분산 정도를 보인다.
![Image](https://github.com/user-attachments/assets/71a65f20-2218-493f-84f3-5c71cac6bfc5)

## batch size 4000

위에서 보다 batch 사이즈를 4배 증가시켰다.

` python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 --exp_name cartpole_lb`

` python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -rtg --exp_name cartpole_lb_rtg `

` python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -na --exp_name cartpole_lb_na `

` python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -rtg -na --exp_name cartpole_lb_rtg_na `

 purple line은 'reward to go', 'nomalize'를 모두 적용하지 않은 버전,

 orange line은 'nomalize' 만을 적용한 버전,

 green line은 'reward to go' 만을 적용한 버전,

 black line는 'reward to go', 'nomalize'를 모두 적용한 버전이다.
 
![Image](https://github.com/user-attachments/assets/86588a98-0c5f-4628-b5f0-b5ffe7e4098b)

batch size가 1000일때 보다 분산의 변동률이 안정적이다.

![Image](https://github.com/user-attachments/assets/f68f1557-49e4-4949-8c75-b418a490c1a8)

# section 4
HalfCheetah가 높은 보상을 주는 방향으로 학습한다.
HalfCheetah는 앞으로 나아갈때 점수를 얻는다.

이번 Section에선 Base line 개념이 도입된다.
Base line은 Reward의 합에 Base line을 빼는 것으로 전체 데이터의 평균값을 조절하는 기술이다.(개인적으로 nomalize와 유사하다고 생각한다)
Base line은 여러가지 방식으로 구현될 수 있지만, 여기선 Value Function으로 base line을 구현한다.
value function은 특정 상태로 부터 시작된 미래 보상의 합을 의미한다.
구현에서 value function은 state와 reward의 합으로 훈련시킨 모델을 이용하여 value function을 구현한다.

`python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --exp_name cheetah `

`python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline`

`python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.007 -bgs 5 --exp_name cheetah_baseline_blr_0.005_bgs_3`

black line은 base line을 적용하지 않은 버전,

blue line은 value function 모델의 learning rate가 0.01, 한번의 trejectory 당 훈련 횟수가 5인 버전,

green line은 value function 모델의 learning rate가 0.007, 한번의 trejectory 당 훈련 횟수가 5인 버전을 의미한다.

왼쪽은 각각의 보상을 나타낸 것이고, 오른쪽은 value function 모델의 loss 함수를 나타낸 것이다.
학습률을 적절히 조절한 green line에서 안정적인 학습이 이루어지고 있다.

![Image](https://github.com/user-attachments/assets/4902f69b-16dd-4cff-a474-798602ae79fa)
![Image](https://github.com/user-attachments/assets/b63b5c17-5ed3-4277-a1a3-956f02b3a902)

# section 5
LunarLander를 이용하여 실험을 진행했다.
Generalized Advantage Estimation 방식에서 람다가 각각 0. 0.95, 0.98, 0.99, 1 일때의 성능을 비교해 보았다.
GAE는 본래의 보상의 합과 base line의 차를 확장한 개념으로, 람다가 높을 수록 모델의 정확도가 오르는 대신 학습의 안정성이 불안정해지고, 낮을 수록 정확도가 낮은 대신 학습이 안정화 된다.

black line은 람다가 0, blue line은 람다가 0.95, pink line은 람다가 0.98, yellow line은 람다가 0.99, purple line은 람다가 1인 버전을 의미한다.

전체적인 평균 보상 그래프는 다음과 같다.

![Image](https://github.com/user-attachments/assets/8d09d092-f361-476e-a623-fd71942bed6d)

가장 좋은 평균 보상 그래프를 보인 람다가 1인 버전의 평균 보상 그래프와 분산 그래프이다.

![Image](https://github.com/user-attachments/assets/bb4f15b1-66b1-4009-a2b0-41b4ca789295)
![Image](https://github.com/user-attachments/assets/748c041d-a5e9-45f8-aae4-079db42127f2)

극단적인 예를 보이기 위해 람다가 0인 버전의 평균 보상 그래프와 분산 그래프이다.

![Image](https://github.com/user-attachments/assets/f754b0ad-c932-40a0-b2d0-e9da2452b7d7)
![Image](https://github.com/user-attachments/assets/2f7f2701-fa4c-4dce-886a-8113f1f1ea5e) 

구현의 문제인진 모르겠으나, 이론과 다르게 람다가 큰 버전에서 작은 버전에 비해 분산이 작게 나왔다.

# section 6
Inverted Pendulum을 이용하여 실험을 진행하였다.
강화 학습은 운의 영향을 많이 받는다.
sampling 시에 어떠한 데이터를 수집하는지에 따라 학습 속도가 달라질 수 있다.
파라미터로 인한 개선을 명확히 확인하기 위해 random seed를 변경하며 얻은 결과의 평균을 확인하였다.`
edit에선 discount 옵션과 gae_lamda 옵션을 다음과 같이 설정했다

`--discount 0.98 --gae_lambda 0.99`

![Image](https://github.com/user-attachments/assets/9eb7a570-15a1-4b79-a282-5ef15c70645d)
