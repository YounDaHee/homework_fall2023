코드 구현이 쉽지 않았다.
처음에 접근을 Imitation Learning에서 그랬듯 Supervised Learning에서 사용하는 출력층 활성화 함수를 사용했는데,
이게 문제였는지 제대로 learning이 수행되지 않았다.
결국 다른 사람 코드를 참고하였다.
참고한 코드에선 sampling 데이터를 확률 분포 내에서 랜덤하게 추출한 값을 이용했다.
이는 강화학습 특성상, 다양한 환경에 직면하여 어떤 상황에서 어떤 행동을 했을때 효율적인지 추출하기 위한 것이다.

처음 확률 분포에 따른 랜덤한 sampling이 아닌 결정론적인 sampling을 이용했을때 return의 결과가 하락하는 결과가 나왔는데,
추측으론, hw2에서 구현한 policy gredient 특성상 return의 결과에 따라 정도가 달라질 뿐이지 sampling에서의 행동이 좋은 결과를 초래하든 나쁜 결과를 초래하든 해당 행동을 할 확률이 조금씩 증가하기 때문일 것이라 생각한다.
(그에대한 반증으로, nomalize 적용 시 학습이 정상적으로 수행되는것을 확인하였다.)

## section 3
CartPole을 오래 유지하도록 강화학습을 수행한다.
해당 실험에선 각 모델마다 'batch size', 'reward to go', 'nomalize'를 변수로 두었다.

통상 trajectory의 길이인 batch size가 증가하면, 모델의 예측 정확도는 상승하되 편차가 증가한다.

reward to go는 강화학습 시나리오 처럼, 학습이 한 시간축에 의한 맥락적 상황에 대해 수행될때 현재의 행동이 과거의 보상에 영향을 미치지 않도록 조절하는 것이다. 이를 적용할 경우 편차가 줄어든다.

nomalize는 학습 데이터가 [-1, 1]의 값을 가지고 평균이 0이되도록 조절하여 학습 성능을 향상시키는 것을 의미한다.

참고로 policy gredient에서는 high variance로 학습이 불안정한 문제가 존재해, 이를 해결하기 위한 다양한 방법이 제안되었다.
 
betch 사이즈는 1000으로 하고, 'reward to go', 'nomalize' 여부에 따라 총 4가지 경우에 대해 실험을 진행하였다.

`python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole 
 python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name cartpole_rtg
 python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -na --exp_name cartpole_na 
 python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -na --exp_name cartpole_rtg_na `

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

위에서 보다 batch 사이즈를 4배 증가시켰다.
` python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 --exp_name cartpole_lb 
 python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -rtg --exp_name cartpole_lb_rtg 
 python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -na --exp_name cartpole_lb_na 
 python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -rtg -na --exp_name cartpole_lb_rtg_na `

 
 
![Image](https://github.com/user-attachments/assets/86588a98-0c5f-4628-b5f0-b5ffe7e4098b)
![Image](https://github.com/user-attachments/assets/f68f1557-49e4-4949-8c75-b418a490c1a8)
