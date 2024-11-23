# Sumo_DQN_Project
## 실험한 도로 환경
![image](https://github.com/user-attachments/assets/c4821473-9d0f-4b14-8bcb-7d1f85eb9a2e)
다음과 같은 T자형 교차로에서 DQN적용
차량은 1시간마다 0~300대를 랜덤하게 생성하게 하였다
## 코드 설명
### Class Qnet
다음과 같이 신경망을 구성하였다.
- 400개의 노드를 fully connected 방식으로 5층 연결하여 구성
- 은닉층의 활성화 함수로는 ReLU적용, 출력층의 활성화 함수로는 sigmoid적용
### sample_action
- 신경망을 토대로 action을 선택하는 부분
- epsilon greedy방법을 적용하였다.
- 신호별로 최소시간을 가지게 두어서 신경망의 출력이 1이어도 최소시간조건을 충족못하면 0을 반환하게 구현
### overwrite_route_file
- Sumo에서 차량을 생성하는 파일은 rou.xml을 작성하는 부분
- 시간마다(3600초마다) 차량이 0~300대를 생성되도록 파일을 write하였다.
### get_halted_vehicles_vector
- MDP정의 중 상태값에서 Qt를 구하기 위한 함수
- 차선별로 정지한 차량이 몇대인지 구하는 함수이다.
### get_traffic_light_vector
- MDP정의 중 상태값에서 Pt를 구하기 위한 함수
- 각 신호의 상태를 벡터값으로 반환한다.
### get_halted_vehicle_count
- MDP정의 중 보상함수를 구하기 위한 메서드
- 모든 차선에 정지한 차량의 수를 가져온다.
- 그리고 그 값에 음수를 부여한다.
### change_traffic_light_phase
- 신호를 바꿔주는 메소드이다.
### train
- 학습을 시키는 로직
- 정답 값을 r + gamma*max(Q(s_prime, a_prime)) [s_prime : 다음 상태값, a_prime : 다음 액션 값] 으로 두었다.
- 손실함수는 CrossEntropy를 채택했다.
### main
- 각 episode별로 10000번씩 진행하였다.
- 논문에 나와있는데로 1step마다 학습을 진행하였다.

## 결과
30에피소드씩 구현했을 때 결과는 다음과 같다.
![KakaoTalk_20241123_181637312](https://github.com/user-attachments/assets/131845fd-5269-48d8-a4b0-94eea4b2441b)
가로축은 수행한 에피소드, 세로축은 누적대기시간이다.
처음에는 30000정도로 높은 값이 나왔지만 2번째 부터는 5000정도에 머물러 있는 것을 확인할 수 있다.
