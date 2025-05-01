# 개요
해당 코드는 CS285 코드를 주어진 조건에 맞추어 수정한 것입니다.

Imitation Learning 기법인 Behavior Cloning과 DAgger를 구현하였습니다.
현재 코드에서는 DAgger의 Beta 설정을 파라미터가 없는(처음에만 학습에 있어 expert가 개입하고 이후에는 개입하지 않는)
버전입니다만, cs285/scripts/run_hw1.py에 있는 주석을 제거하면 파라미터가 있는 버전으로 변환할 수 있습니다.

학습 과정에서, 학습 기준이 되는 데이터에서 무작위로 임의의 갯수의 데이터를 뽑아 학습하도록 설계하였습니다.
loss 함수로는 F.mse_loss()를 사용하였습니다.

### 지원하는 gym 환경
"Ant-v4", "Walker2d-v4", "HalfCheetah-v4", "Hopper-v4"

### 실행 방법
```
# expert 실행
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 env PYTHONPATH=. python cs285/scripts/run_expert.py --expert_policy_file cs285/policies/experts/[실행시킬 환경].pkl --env_name [실행시킬 환경]-v4 --expert_data cs285/expert_data/expert_data_[실행시킬 환경]-v4.pkl --video_log_freq 1 --eval_batch_size 5000

#BC 실행
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 env PYTHONPATH=. python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/[실행시킬 환경].pkl --env_name [실행시킬 환경]-v4 --exp_name bc --n_iter 1 --expert_data cs285/expert_data/expert_data_[실행시킬 환경]-v4.pkl --video_log_freq 5 --num_agent_train_steps_per_iter 200 --eval_batch_size 5000

#DAgger 실행
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 env PYTHONPATH=. python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/[실행시킬 환경].pkl --env_name [실행시킬 환경]-v4 --exp_name dagger --n_iter 10 --do_dagger --expert_data cs285/expert_data/expert_data_[실행시킬 환경]-v4.pkl --video_log_freq 5 --num_agent_train_steps_per_iter 200

```
# 실험

Expert, BC, DAgger에 대해 5000개의 데이터 셋을 모아 성능을 비교하였습니다.
(한번의 실행마다 최대 1000개의 데이터 셋을 모읍니다. 행동 불능 상태가 되어 조기 종료되는 경우가 생기지 않으면 5번 실행 됩니다.)
비교는 gym에서 제공하는 reward를 가지고 진행됩니다.
BC와 DAgger는 매 반복마다 200번의 training을 수행합니다.

Expert와 BC의 성능을 비교한 것입니다.
대부분의 환경에서 BC는 Expert의 15% 미만의 성능을 보이지만, HalfCheetah-v4에서는 70% 가량의 성능을 보이고 있습니다.
이는 HalfCheetah의 경우, "상태 불능"의 경우가 없기에 조기 종료 되지 않은 덕이라 생각합니다.

![Image](https://github.com/user-attachments/assets/8649fccf-e333-4ebf-8746-37e883157823)

다음으로 분산입니다. 

![Image](https://github.com/user-attachments/assets/2dc03bf7-390c-46cf-ac6c-7653890551ed)

BC에서 training 횟수를 1000으로 늘렸을땐, 좋지 않은 성능을 보였던 Walker2d-v4 환경에서 906.45를 보여 대략 4.5배(3.84%->17.26%)의 성능을 보였습니다.

BC에서 가장 좋은 성능을 보였던 HalfCheetah-v4와 가장 나쁜 성능을 보였던 Walker2d-v4을 DAgger를 통해 실행하였습니다.

이때 사용한 Beta는 Parameter가 있는 버전으로, 초기에는 expert 데이터 만을 이용하여 학습, 이후로는 0.9씩 기하급수적으로 비율을 줄여나갔습니다.(1 -> 0.9 -> 0.81 ...)

![Image](https://github.com/user-attachments/assets/3240b931-2073-4f1e-bca2-e892320058ae)

Parameter가 없는 버전의 Beta로도 동일하게 학습을 진행하였습니다.(처음에는 expert 데이터만을 이용하여 학습, 이후로는 자신만을 이용하여 학습)

![Image](https://github.com/user-attachments/assets/ab083108-e00f-4b55-8718-2ad8a9d9a410)

당연한 결과지만, 해당 버전의 경우 학습이 제대로 진행되지 않음을 확인할 수 있습니다.
BC에서도 준수한 성능을 보였던 HalfCheetah-v4의 경우 iteration을 반복할 수록 성능이 조금씩 개선됨을 알 수 있지만, Walker2d-v4의 성능은 iteration의 수행에도 성능이 거의 개선되지 않습니다.
논문에서 해당 beta 설정의 경우에도 준수한 성능을 보였다고 하여 기대하고 실험을 진행했지만 예상과 다른 결과나 나와 아쉽습니다.
논문에서 학습 데이터 셋을 어떻게 뽑았는지가 나와있지 않아 이를 구현하긴 힘들거 같습니다.

HalfCheetah-v4와 Walker2d-v4의 과정을 도식으로 나타낸 것입니다.

![Image](https://github.com/user-attachments/assets/82427baf-8fa0-4d89-ac1f-c50c802bf654)
![Image](https://github.com/user-attachments/assets/9eba8147-5b1e-4855-8663-82edbca6a247)
