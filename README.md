# rpb2023_18조
## 기계시스템 설계와 로봇 프로그래밍의 기초 18 조

기계시스템설계 및 로봇프로그래밍 기초: ROS 실습
Task 0: Download data files & git clone
다음의 두 개의 bag 파일을 다운로드 받습니다. 다운로드에 시간이 걸리므로 수업을 시작하기 전에
다운로드를 시작합니다.
1) ros-turtle.bag
https://drive.google.com/file/d/14gh9gflHGWWIoJHxImFg7gckDgAcDOYn/view?
usp=share_link
2)대회
nttps://drive.google.com/file/d/1qIFn2Nfw_ZUB9q3E8AZXfAZ_ 5pAatGfi/view?usp-share_link
Task 1: Install ROS
ROS를 설치합니다. 다음 홈페이지로 이동하여 R0S1 버전명 Neotic Ninjemys를 설치합니다.
https://www.ros.org!
설치 페이지에서 한 줄씩복사/붙여넣기로 터미널에서 실행하면 됩니다. 설치가 완료되면 다음 명령
어를 실행시켜서 설치 완료를 확인합니다.
roscore
Task 2: Implement talker & listener
팀별 gitrepo을 clone을 받습니다. 나중에 push 할 수 있도록 token도 준비합니다. 오늘 실습은 git
repo안에 ros 라는 폴더를 만들고 그 안에서 실습하겠습니다.
scd <내 깃 리포 폴더>
mkdir ros
cd ros
Task 2-1 Elice에서 실습했 던 talker.py를 local 컴퓨터에서 실행 시켜보겠습니다
여러분의 깃리포 폴더 안에 ros 폴더 안에 talker.py 를 만들고 gedit 으로 파일을 엽니다.
S touch talker.py
gedit talker.py
talker.py 에 여러분이 미리 실습한 Eice 내용을 복사/붙여넣기 합니다.
