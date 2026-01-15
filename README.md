## 0. 팀 소개
📌 팀명 : 엥?비디아 <br>
📌 팀원 : 강지윤, 김민지, 임은석, 함성민

<table>
  <tr>
    <td align="center">
     <img width="160" height="160" alt="image" src="https://github.com/user-attachments/assets/c56595d9-1d25-4f18-8447-4bbe2ca82bce" /><br/>강지윤
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
      <a href="https://github.com/ki-student"><img src="https://img.shields.io/badge/GitHub-ki--student-1F1F1F?logo=github" alt="강지윤 GitHub"/></a>
    </td>
    <td align="center">
      <a href="https://github.com/Jinhyeok33"><img src="https://img.shields.io/badge/GitHub-Jinhyeok33-1F1F1F?logo=github" alt="김민지 GitHub"/></a>
    </td>
    <td align="center">
      <a href="https://github.com/jiyun-kang12"><img src="https://img.shields.io/badge/GitHub-jiyun--kang12-1F1F1F?logo=github" alt="임은석 GitHub"/></a>
    </td>
    <td align="center">
      <a href="https://github.com/oowixj819"><img src="https://img.shields.io/badge/GitHub-oowixj819-1F1F1F?logo=github" alt="한성민 GitHub"/></a>
    </td>
</table>

<img width="600" height="337" alt="image" src="https://github.com/user-attachments/assets/9afb326b-6196-4a78-8da6-2254c3404bef" />

## 📍 1. 프로젝트 개요

>주제:시각적 행동 상태 정보와 기하학적 지표를 결합한 카메라 기반 고신뢰도 전방 충돌 경고(Forward Collision Warning, FCW) 시스템 구현 <br>

> 수행기간 : 2026.01.09 ~2026.01.16

### 📌 1.1. 주제 선정 배경 
- **기존 기술의 한계**: 현재 주류인 센서 퓨전(레이더+카메라) 방식은 높은 원가 부담이 있으나 카메라 단독 방식은 고속/장거리 조건에서 신뢰도 확보가 어려운 한계가 있음
- **단순 지표의 맹점**: 거리와 속도만 계산하는 기존 TTC(충돌 예상 시간) 방식은 앞차의 급제동이나 끼어들기 전조 증상(브레이크등, 깜빡이)을 미리 읽지 못해 정교한 위험 판단에 한계가 있음
- **기술적 필요성**: 저성능 연산 환경에서도 지연 없이 동작하면서, 물리적 지표와 차량의 의도(행동 변화)를 동시에 포착하는 경제적이고 신뢰도 높은 시스템이 필요함

### 📌 1.2. 프로젝트 목표
- **단일 센서 활용**: 고가의 LiDAR나 레이더 없이 **전방 RGB 카메라**만 사용하여 시스템 구현
- **복합 판단 로직**:
    1. **물리적 안전선 확보**: 거리·상대속도 기반의 기하학적 지표(TTC) 산출
    2. **조기 위험 감지**: 차량의 상태 정보(브레이크등, 방향지시등, 위치 변화)를 추출 및 통합
- **실시간성 및 신뢰도**: 제한된 연산 자원에서도 지연 없이 동작하며, 단순 기하학적 수치보다 더 빠르고 정교하게 위험 상황을 식별하여 경고의 정확도 향상

## 📍 2. 역할분담 🧑‍🤝‍🧑
<table> <tr> <th align="center">Role</th> <th align="center">담당 영역</th> <th align="center">주요 수행 내용</th> <th align="center">담당자</th> </tr> <tr> <td align="center"><b>Role #1</b><br/>📊 데이터 / GT 파이프라인</td> <td align="center">데이터 전처리 및 분석</td> <td> • txt 라벨 파싱<br/> • instance mask 해석<br/> • train / val split<br/> • 데이터 증강(기초)<br/> • 데이터 분포 리포트 </td> <td align="center"> 강지윤<br/> 함성민<br/> 임은석<br/> 김민지 </td> </tr> <tr> <td align="center"><b>Role #2</b><br/>🚗 Object Detection 모델</td> <td align="center">객체 검출 모델 학습</td> <td> • Agent(차량/버스) 2D detector 학습/추론 파이프라인<br/> • 성능 지표(mAP) 산출<br/><br/> <b>산출물</b><br/> • checkpoint<br/> • inference script<br/> • PR curve / 성능 표 </td> <td align="center"> 강지윤<br/> 김민지 </td> </tr> <tr> <td align="center"><b>Role #3</b><br/>📍 Location / Action 모델</td> <td align="center">객체 상태 인식</td> <td> • 2-stage crop classifier 구현<br/> • Location: 5-way softmax<br/> • Action: 4-way sigmoid (BCE, multi-label)<br/><br/> <b>산출물</b><br/> • attribute predictor<br/> • confusion matrix / micro-F1<br/> • threshold 튜닝 결과 </td> <td align="center"> 함성민 </td> </tr> <tr> <td align="center"><b>Role #4</b><br/>⚠️ Tracking + TTC + FCW</td> <td align="center">위험도 판단 및 서비스 데모</td> <td> • SORT 기반 객체 추적 연동<br/> • TTC(Time-To-Collision) 계산<br/> • FCW 로직 설계<br/> • 영상 오버레이(UI/UX) 구현<br/><br/> <b>산출물</b><br/> • 데모 영상<br/> • 경고 로그<br/> • 실패 케이스 분석 </td> <td align="center"> 임은석<br/> 강지윤<br/> 김민지 </td> </tr> </table>

