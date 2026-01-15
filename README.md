## 0. 팀 소개
📌 팀명 : 엥?비디아 <br>
📌 팀원 : 강지윤, 김민지, 임은석, 함성민

<table>
  <tr>
    <td align="center">
     <img width="120" height="120" alt="image" src="https://github.com/user-attachments/assets/c56595d9-1d25-4f18-8447-4bbe2ca82bce" /><br/>강지윤
    </td>
    <td align="center">
      <img width="100" height="100" alt="image" src="https://github.com/user-attachments/assets/db8a1282-671b-4ef2-87e0-840c08b2b650" /><br/>김민지
    </td>
    <td align="center">
      <img width="100" height="100" alt="image" src="https://github.com/user-attachments/assets/af8934e3-3616-4b2a-b9e7-25129c1c3208" /><br/>임은석
    </td>
    <td align="center">
     <img width="100" height="100" alt="image" src="https://github.com/user-attachments/assets/3c7dcd17-638b-42e7-988a-afb9ed2432e8" />
<br/>함성민
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/jiyun-kang12"><img src="https://img.shields.io/badge/GitHub-jiyun--kang12-1F1F1F?logo=github" alt="강지윤 GitHub"/></a>
    </td>
    <td align="center">
      <a href="https://github.com/lhjjsh8-sketch"><img src="https://img.shields.io/badge/GitHub-lhjjsh8--sketch-1F1F1F?logo=github" alt="김민지 GitHub"/></a>
    </td>
    <td align="center">
      <a href="https://github.com/mjmjkkk"><img src="https://img.shields.io/badge/GitHub-mjmjkkk-1F1F1F?logo=github" alt="임은석 GitHub"/></a>
    </td>
    <td align="center">
      <a href="https://github.com/raretomato"><img src="https://img.shields.io/badge/GitHub-raretomato-1F1F1F?logo=github" alt="한성민 GitHub"/></a>
    </td>
</table>


## 📍 1. 프로젝트 개요

>주제:시각적 행동 상태 정보와 기하학적 지표를 결합한 카메라 기반 고신뢰도 전방 충돌 경고<br>(Forward Collision Warning, FCW) 시스템 구현 <br>

> 수행기간 : 2026.01.09 ~2026.01.16

### 📌 1.1. 주제 선정 배경 
- **기존 기술의 한계**: 현재 주류인 센서 퓨전(레이더+카메라) 방식은 높은 원가 부담이 있으나 카메라 단독 방식은<br> 고속/장거리 조건에서 신뢰도 확보가 어려운 한계가 있음
- **단순 지표의 맹점**: 거리와 속도만 계산하는 기존 TTC(충돌 예상 시간) 방식은 앞차의 급제동이나 <br>끼어들기 전조 증상(브레이크등, 깜빡이)을 미리 읽지 못해 정교한 위험 판단에 한계가 있음
- **기술적 필요성**: 저성능 연산 환경에서도 지연 없이 동작하면서, 물리적 지표와 차량의 의도(행동 변화)를<br> 동시에 포착하는 경제적이고 신뢰도 높은 시스템이 필요함

### 📌 1.2. 프로젝트 목표
- **단일 센서 활용**: 고가의 LiDAR나 레이더 없이 **전방 RGB 카메라**만 사용하여 시스템 구현
- **복합 판단 로직**:
    1. **물리적 안전선 확보**: 거리·상대속도 기반의 기하학적 지표(TTC) 산출
    2. **조기 위험 감지**: 차량의 상태 정보(브레이크등, 방향지시등, 위치 변화)를 추출 및 통합
- **실시간성 및 신뢰도**: 제한된 연산 자원에서도 지연 없이 동작하며, 단순 기하학적 수치보다 더 빠르고<br> 정교하게 위험 상황을 식별하여 경고의 정확도 향상

## 📍 2. 역할분담 🧑‍🤝‍🧑
<table>
  <tr>
    <th align="center">Role</th>
    <th align="center">주요 내용</th>
    <th align="center">담당자</th>
  </tr>

  <tr>
    <td align="center"><b>#1</b><br/>📊 데이터</td>
    <td>
      • GT(txt/mask) 파싱<br/>
      • train/val split<br/>
      • 데이터 증강<br/>
      • 분포 리포트
    </td>
    <td align="center">
      강지윤<br/>함성민<br/>임은석<br/>김민지
    </td>
  </tr>

  <tr>
    <td align="center"><b>#2</b><br/>🚗 Detection</td>
    <td>
      • YOLO 기반 2D detector<br/>
      • mAP 평가<br/>
      • checkpoint / inference
    </td>
    <td align="center">
      강지윤<br/>김민지
    </td>
  </tr>

  <tr>
    <td align="center"><b>#3</b><br/>📍 Action</td>
    <td>
      • 2-stage crop classifier<br/>
      • Location(5-way)<br/>
      • Action(4-way, BCE)
    </td>
    <td align="center">
      함성민
    </td>
  </tr>

  <tr>
    <td align="center"><b>#4</b><br/>⚠️ FCW</td>
    <td>
      • SORT tracking<br/>
      • TTC / FCW 로직<br/>
      • 데모 영상
    </td>
    <td align="center">
      임은석<br/>강지윤<br/>김민지
    </td>
  </tr>

</table>

## 📍 3. 프로젝트 내용
### 📌 3.1. 데이터셋
- [자율주행 인공지능 챌린지] 객체 복합상태인식 및 인스턴스 세그멘테이션 데이터셋
<src="https://nanum.etri.re.kr/share/kimjy/ObjectStateDetectionAIchallenge2024?lang=ko_KR">
- 165개 환경에서 수집된 33,187개의 전방 RGB 단안 카메라 이미지 (1280*480)

- 데이터셋 구성
  - 전방 RBG 데이터 (.png)
  - 레이블(bbox, 분류{차/버스}, 위치{주행차로, 진행방향 차로, 맞은편 차로, 교차로, 주차장}, 후미등 상태{브레이크등, 좌/우 깜박이, 비상등}) 파일 (.txt)
  - 인스턴스 마스크 파일 (.png)
<img width="300" height="284" alt="image" src="https://github.com/user-attachments/assets/6dac5d9f-44ef-4e58-8e44-54136c446ba0" />

## 📌 3.2. 선정 모델 및 전체 파이프라인 흐름

### 3.2.1. 파이프라인 단계별 사용 모델

| 파이프라인 단계                         | 사용 모델                         |
| -------------------------------- | ----------------------------- |
| Object Detection                 | YOLO v8, YOLO v11n, YOLO v11m |
| Location / Action Classification | ResNet50 + FPN + ROIAlign     |
| TTC Proxy + FCW 로직               | Meng et al. (arXiv, 2023) 기반  |

---

### 3.2.2. 전체 파이프라인 흐름

<p align="center">
  <img width="300" height="500" alt="pipeline" src="https://github.com/user-attachments/assets/9a342eea-adde-4eab-988f-c7f7259e9af4" />
</p>

---

### (1) Object Detection

* YOLOv11n 기반 전방 차량(Car / Bus) 객체 검출
* Bounding Box 및 Confidence 추출
* SORT / DeepSORT 기반 다중 객체 추적 및 Track ID 할당

---

### (2) Location / Action Classification

* Bounding Box 단위 ROI 기반 분류
* **Location (5-class, single-label)**

  * 주행차로 / 진행차로 / 반대차로 / 교차로 / 주차차로
* **Action (multi-label)**

  * 브레이크등 / 좌·우 방향지시등 / 비상등

---

### (3) TTC Proxy + FCW 로직

* Bounding Box 크기 변화 기반 TTC Proxy 계산
* Location / Action 정보를 반영한 위험도 보정
* 최종 위험 점수 계산

$$
Score_{final} = \min(Score_{base} \times \omega + Bonus_{semantic},; 100)
$$

---

### (4) FCW 시각화

* 위험 점수에 따른 Bounding Box 색상 표시

| 위험 점수        | 위험 단계   |
| ------------ | ------- |
| $> 50$       | DANGER  |
| $30 \sim 50$ | CAUTION |
| $< 30$       | SAFE    |


## 📌 6. 파이프라인별 실험 결과

### 6.1. Object Detection 속도 비교 실험

* **YOLO 버전별 성능 비교**

```text
Version   Class  Images  Instances   P      R      mAP50  mAP50-95
YOLOv8    all    3662    26343      0.916  0.814   0.901   0.741
YOLOv11   all    3662    26343      0.894  0.777   0.870   0.684
```

* **추론 속도 비교**

```text
YOLOv11m : 24.03 ms | 41.6 FPS
YOLOv11n : 23.21 ms | 43.1 FPS
→ 약 1.04배 속도 향상
```

* mAP 성능이 유사한 조건에서 **YOLOv11n이 더 빠른 추론 속도**를 보여
  **YOLOv11n을 최종 Detection 모델로 선정**

### 6.2. Location / Action Classification 성능 개선

#### (1) ROI 크기 조절 실험 (box_expand_ratio)

* Bounding Box를 축소하여 **차량 영역에 집중**하도록 설정
* 주차 차로(Location 4)를 제외한 대부분의 위치 클래스에서 **80% 이상 성능 확보**
* 주차 차량 및 교차로 차량은 FCW 대상 비중이 낮아
  전체 시스템 성능에 미치는 영향은 제한적이라고 판단

#### (2) Action Threshold 조정 실험

* 학습 시 threshold = 0.5로 통일
* 실제 환경에서는 **action별 threshold 분리 필요**
* Validation set 기반 threshold sweep 수행
* **F1-score가 최대가 되는 threshold를 action별로 설정**
  
### 6.3. Forward Collision Warning (FCW) 로직 설계

#### 6.3.1. 최종 로직: 연속 프레임 기반 위험도 판단

* Meng et al. (2023)의 **궤적 기반 TTC 계산 방식**을 참고
* 물리적 위험 점수와 의미적 위험 증폭을 결합한 로직 설계

**(1) 객체 주행 궤적 기반 TTC 계산**

* 최근 6프레임 동안의 객체 중심점 $(c_x, c_y)$ 유지
* 자차(Ego vehicle)는 영상 하단 중앙에 위치한다고 가정
* 접근 중인 객체에 대해서만 TTC 계산 수행

**(2) 물리적 위험 점수 (Base Risk)**

* TTC가 임계값 이하일 경우에만 위험 점수 부여
* TTC가 짧아질수록 비선형적으로 위험 증가
* 먼 객체는 자연스럽게 위험 평가에서 제외

**(3) 의미적 위험 증폭 (Semantic Risk)**

* **도로 문맥 필터링**

  * 반대 차로 / 교차로 외부 / 주차 차량 → 제외

* **주행 차로(Ego Lane)**

  * TTC < 2s & 고속 접근 → *Rapid Approach*
  * TTC < 2s & 저속 접근 → *Close Following*

* **인접 차로(Adjacent Lane)**

  * 방향지시등 활성화 → *Dangerous Cut-in*
  * 그 외 → *Approaching*

**(4) 최종 위험 점수**

$$
Score_{final}
= \min(Score_{base} \times \omega + Bonus_{semantic},; 100)
$$



