import torch
import torch.nn as nn
import torch.optim as optim
import traci
import numpy as np
import xml.etree.ElementTree as ET
import random
import torch.nn.functional as F

gamma = 0.99

class Qnet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Qnet, self).__init__()
        # 입력층부터 은닉층 5층 구성
        self.fc1 = nn.Linear(state_dim, 400)  # 입력
        self.fc2 = nn.Linear(400, 400)       # 두 번째 은닉층
        self.fc3 = nn.Linear(400, 400)       # 세 번째 은닉층
        self.fc4 = nn.Linear(400, 400)       # 네 번째 은닉층
        self.fc5 = nn.Linear(400, 400)       # 다섯 번째 은닉층
        self.output = nn.Linear(400, action_dim)  # 출력층

    def forward(self, x):
        # 은닉층 활성화 함수: ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        # 출력층 활성화 함수: Sigmoid
        x = torch.sigmoid(self.output(x))
        return x

    def sample_action(self, obs, traffic_time, phase_min_time, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 1)  # 랜덤 행동 선택
        else:
            out = self.forward(obs).argmax().item()  # 최대 Q-값을 갖는 행동 선택
            if out == 1 and traffic_time < phase_min_time:
                out = 0
            return out



def overwrite_route_file(file_path, num_repeats):
    with open(file_path, "w", encoding="UTF-8") as file:
        file.write("""<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <!-- Vehicles, persons and containers (sorted by depart) -->
""")
        begin = 0
        step = 3600
        # flow 항목 반복하여 생성
        for i in range(num_repeats):
            end = (i + 1) * step
            flow_number_1 = random.randint(0, 300)
            flow_number_2 = random.randint(0, 300)
            flow_number_3 = random.randint(0, 300)
            flow_number_4 = random.randint(0, 300)
            flow = f"""    <flow id="F{i*4+1}" begin="{begin}" departLane="random" departSpeed="1" fromTaz="taz_0" toTaz="taz_1" end="{end}" number="{flow_number_1}"/>
    <flow id="F{i*4+2}" begin="{begin}" departLane="random" fromTaz="taz_1" departSpeed="1" toTaz="taz_0" end="{end}" number="{flow_number_2}"/>
    <flow id="F{i*4+3}" begin="{begin}" departLane="random" fromTaz="taz_2" departSpeed="1" toTaz="taz_0" end="{end}" number="{flow_number_3}"/>
    <flow id="F{i*4+4}" begin="{begin}" departLane="random" fromTaz="taz_1" departSpeed="1" toTaz="taz_2" end="{end}" number="{flow_number_4}"/>"""
            file.write(flow + "\n")
            begin = end + 1

        # XML 종료 태그
        file.write("</routes>\n")

def get_halted_vehicles_vector(lane_ids):
    
    halted_counts = []
    for lane_id in lane_ids:
        halted_count = 0
        vehicle_ids_in_lane = traci.lane.getLastStepVehicleIDs(lane_id)  # 차선에 있는 차량 ID 리스트 가져오기

        for vehicle_id in vehicle_ids_in_lane:
            speed = traci.vehicle.getSpeed(vehicle_id)  # 차량의 속도를 확인
            if speed == 0:  # 차량이 정지한 경우
                halted_count += 1

        halted_counts.append(halted_count)

    return halted_counts

def get_traffic_light_vector(tls_id):
    
    # 현재 신호 상태 문자열 가져오기 (예: "rGrG" 등)
    light_state = traci.trafficlight.getRedYellowGreenState(tls_id)

    # 신호 상태를 벡터로 변환 (초록: 1, 빨강/노랑: 0)
    state_vector = [1 if light == 'G' or light == 'g' else 0 for light in light_state]

    return state_vector

def get_halted_vehicle_count():
    
    total_halted_vehicles = 0
    lane_ids = traci.lane.getIDList()  # 모든 차선 ID 가져오기

    for lane_id in lane_ids:
        stopped_cars = traci.lane.getLastStepHaltingNumber(lane_id)
        total_halted_vehicles -= stopped_cars  # 보상에 추가 (음수 값)

    return total_halted_vehicles

def change_traffic_light_phase(tls_id, phase_index):
    
    traci.trafficlight.setPhase(tls_id, phase_index)

def train(q, memory, optimizer):
      
    s, a, r, s_prime = memory

    # NumPy 배열을 PyTorch 텐서로 변환
    s = torch.from_numpy(s).float()
    r = torch.tensor(r).float()
    s_prime = torch.from_numpy(s_prime).float()

    # 네트워크 통과
    q_out = q(s)  # 모든 액션에 대한 Q값
    q_a = q_out[a]  # 선택된 액션의 Q값

    max_q_prime = torch.max(q(s_prime))
    target = r + gamma * max_q_prime  # Target Q값

    loss = torch.nn.functional.mse_loss(q_a, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # print(f"loss: {loss.item():.4f}")


def main():
  sumo_binary = "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo-gui.exe" # or sumo-gui
  sumocfg_dir = "C:/sumo_img/case_a/case_a.sumocfg"
  route_dir = "C:/sumo_img/case_a/route.rou.xml"

  q = Qnet(18,2) 
  optimizer = optim.Adam(q.parameters(), lr=0.00025)

  for episode in range(100):
     #차량 수요 랜덤으로 설정
    overwrite_route_file(route_dir, 3)
    sumo_cmd = [sumo_binary, "-c", sumocfg_dir, "-r", route_dir, "--junction-taz", "--no-warnings", "--random"]
    traci.start(sumo_cmd)

    num_steps = 10000

    #dt 측정하기 위함
    traffic_time = 0

    #신호 phase
    phase = 0

    #신호 phase 당 최소 시간
    phase_min_time = [30, 3, 10, 3]

    #관찰값 초기화
    s = [0] * 18
    s = np.array(s)

    #누적 대기시간
    cumulate_waitingTime = 0

    epsilon = max(0.01, 0.08 - 0.01 * (episode/200))

    for step in range(num_steps):
        if phase == 0 or phase == 2: # 노란불은 고정으로 3초로 해야하니깐
            a = q.sample_action(torch.from_numpy(s).float(), traffic_time, phase_min_time[phase], epsilon)

        #신호 바꾸기
        if a == 1 :
          phase = (phase + 1) % 4
          traffic_time = 0
          change_traffic_light_phase("J6", phase)

        traci.simulationStep()
        traffic_time += 1

        s_prime = np.array([])
        s_prime = np.concatenate([
          get_halted_vehicles_vector(traci.lane.getIDList()),
          get_traffic_light_vector("J6")
        ])
        s_prime = np.append(s_prime, traffic_time)

        r = get_halted_vehicle_count()

        train(q, (s,a,r,s_prime), optimizer)

        s = s_prime

        cumulate_waitingTime += r * (-1)
        
        if step % 1000 == 0 and step != 0:
            print(f"Step: {step}")
            print(f"reward: {r}")
            print(f"culmulate_waitingTime: {cumulate_waitingTime}")

    
            
    traci.close()

if __name__ == '__main__':
    main()


